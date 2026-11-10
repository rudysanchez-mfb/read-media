#!/usr/bin/env python3
"""read-media — turn images, videos, and audio into TEXT a text-only model can act on.

Images  -> one call to a vision-capable model over an OpenAI-compatible
           chat-completions API.
Videos  -> ffprobe metadata + scene-change keyframes (ffmpeg) described by the
           vision model with timestamps + whisper.cpp transcript of the audio
           track + an optional synthesis pass on a text model.
Audio   -> whisper.cpp transcript (+ optional synthesis).

Only image/audio payloads go to the configured API base; whisper.cpp runs
locally. Frames land in a scratch directory (default ~/scratch/read-media/<slug>/).

Environment:
    READ_MEDIA_API_KEY       API key for the chat-completions endpoint
                             (fallback: OPENAI_API_KEY)
    READ_MEDIA_API_BASE      base URL (default https://api.openai.com)
    READ_MEDIA_VISION_MODEL  vision model id (default gpt-4o-mini)
    READ_MEDIA_TEXT_MODEL    synthesis model id (default gpt-4o-mini)
    WHISPER_CPP_BIN          path to the whisper.cpp CLI
                             (default: whisper-cli on PATH)
    WHISPER_MODEL_PATH       path to a ggml whisper model
                             (default: a ggml-*.bin under ~/.cache/whisper-models/)
    READ_MEDIA_SCRATCH_DIR   scratch root for frames
                             (default ~/scratch/read-media)

Usage:
    read_media.py PATH... [--prompt "question"] [--max-frames 12]
                  [--scene 0.02] [--min-gap 2] [--every SECONDS] [--no-audio] [--no-synth]
                  [--model VISION_ID] [--text-model TEXT_ID] [--json]
                  [--frames-dir DIR]

Exit codes: 0 ok, 1 gateway/tool error, 2 bad path, 3 no token, 4 empty answer.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

CHAT_PATH = "/v1/chat/completions"
# Base URL of an OpenAI-compatible chat-completions endpoint.
DEFAULT_HOST = os.environ.get("READ_MEDIA_API_BASE", "https://api.openai.com").strip().rstrip("/")

VISION_MODEL = os.environ.get("READ_MEDIA_VISION_MODEL", "gpt-4o-mini")  # vision seat (must accept image_url parts)
TEXT_MODEL = os.environ.get("READ_MEDIA_TEXT_MODEL", "gpt-4o-mini")      # synthesis seat

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic"}
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".caf"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".ts", ".wmv"}

SCRATCH_ROOT = (os.environ.get("READ_MEDIA_SCRATCH_DIR", "").strip()
                or os.path.expanduser("~/scratch/read-media"))
WHISPER_MODEL_GLOB = os.path.expanduser("~/.cache/whisper-models/ggml-*.bin")

FRAME_PROMPT = (
    "You are the eyes for a text-only agent. These are frames from a video, "
    "each labelled with its timestamp. For EACH frame, write a section headed "
    "exactly `### [<timestamp>]` and describe precisely what is on screen: all "
    "visible text verbatim, UI elements, layout, what changed from the previous "
    "frame, and anything an agent would need in order to act on it. Be concrete, "
    "no filler."
)
IMAGE_PROMPT = (
    "You are an image reader for a text-only agent. Describe this image "
    "precisely and completely: every piece of on-screen text, layout, UI "
    "elements, colors, and any notable detail. Give the exact context an "
    "agent needs to act on it."
)


# ----------------------------------------------------------------- helpers --
def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def token() -> str:
    return (os.environ.get("READ_MEDIA_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip())


def hms(t: float) -> str:
    t = max(0.0, float(t))
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def data_uri(path: str) -> str:
    with open(path, "rb") as fh:
        raw = fh.read()
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def need(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise RuntimeError(f"{binary} not on PATH (install ffmpeg)")
    return path


def slugify(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-")[:60] or "media"


# ---------------------------------------------------------------- gateway ---
def chat(model: str, messages: list, max_tokens: int, timeout: int = 180) -> str:
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if "deepseek" in model:
        body["reasoning_effort"] = "none"   # thinking would eat the budget
    req = urllib.request.Request(
        DEFAULT_HOST + CHAT_PATH,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {token()}",
                 "Content-Type": "application/json",
                 "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"gateway HTTP {e.code}: {e.read().decode('utf-8')[:300]}")
    content = data["choices"][0]["message"].get("content") or ""
    return str(content).strip()


def describe_images(paths: list[str], prompt: str, model: str, max_tokens: int) -> str:
    content: list = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": data_uri(p)}} for p in paths]
    return chat(model, [{"role": "user", "content": content}], max_tokens)


# ------------------------------------------------------------------ probe ---
def probe(path: str) -> dict:
    ffprobe = need("ffprobe")
    cp = run([ffprobe, "-v", "error", "-show_entries",
              "format=duration,size:stream=codec_type,codec_name,width,height,"
              "r_frame_rate,nb_frames", "-of", "json", path])
    if cp.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {cp.stderr.strip()[:300]}")
    info = json.loads(cp.stdout or "{}")
    out = {"duration": float(info.get("format", {}).get("duration") or 0.0),
           "size": int(info.get("format", {}).get("size") or 0),
           "video": None, "audio": None}
    for s in info.get("streams", []):
        if s.get("codec_type") == "video" and out["video"] is None:
            num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
            fps = float(num) / float(den or 1) if float(den or 1) else 0.0
            out["video"] = {"codec": s.get("codec_name"), "width": s.get("width"),
                            "height": s.get("height"), "fps": round(fps, 3)}
        elif s.get("codec_type") == "audio" and out["audio"] is None:
            out["audio"] = {"codec": s.get("codec_name")}
    return out


# ----------------------------------------------------------------- frames ---
def scene_times(path: str, threshold: float, workdir: str) -> list[tuple[float, float]]:
    """(pts_time, score) for frames whose scene score exceeds threshold."""
    ffprobe = need("ffprobe")
    # movie= filter chokes on ':' and spaces in paths; go through a safe symlink.
    link = os.path.join(workdir, "src" + os.path.splitext(path)[1].lower())
    if os.path.lexists(link):
        os.unlink(link)
    os.symlink(os.path.abspath(path), link)
    cp = run([ffprobe, "-v", "error", "-f", "lavfi",
              "-i", f"movie={link},select=gt(scene\\,{threshold})",
              "-show_entries", "frame=pts_time:frame_tags=lavfi.scene_score",
              "-of", "json"], timeout=900)
    if cp.returncode != 0:
        log(f"warning: scene detection failed, falling back to uniform sampling: "
            f"{cp.stderr.strip()[:200]}")
        return []
    hits = []
    for fr in json.loads(cp.stdout or "{}").get("frames", []):
        try:
            hits.append((float(fr["pts_time"]),
                         float(fr.get("tags", {}).get("lavfi.scene_score", 0.0))))
        except (KeyError, ValueError):
            continue
    return hits


def pick_times(duration: float, hits: list[tuple[float, float]], max_frames: int,
               every: float | None, min_gap: float = 2.0) -> list[float]:
    """Choose keyframe timestamps.

    Scene scores on screen recordings are tiny (0.02-0.05 — most of the frame
    never moves), so there is no fixed threshold that works for both a screen
    capture and a camera cut. Instead: take every frame above a low floor,
    then greedily keep the highest-scoring ones that are at least `min_gap`
    seconds apart. t=0 is always included. If that yields too few, sample
    uniformly.
    """
    if every:
        times = [0.0]
        t = every
        while t < duration and len(times) < max_frames:
            times.append(round(t, 3))
            t += every
        return times
    chosen: list[float] = [0.0]
    for t, _score in sorted(hits, key=lambda x: -x[1]):
        if len(chosen) >= max_frames:
            break
        if all(abs(t - c) >= min_gap for c in chosen):
            chosen.append(round(t, 3))
    if len(chosen) < min(max_frames, 3) and duration > 0:
        n = max_frames
        chosen = sorted({round(duration * i / n, 3) for i in range(n)})
    return sorted(chosen)[:max_frames]


def extract_frames(path: str, times: list[float], workdir: str, max_width: int) -> list[tuple[float, str]]:
    ffmpeg = need("ffmpeg")
    out = []
    for i, t in enumerate(times):
        dst = os.path.join(workdir, f"frame-{i:03d}-{t:08.2f}s.jpg")
        cp = run([ffmpeg, "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", path,
                  "-frames:v", "1", "-vf", f"scale='min({max_width},iw)':-2",
                  "-q:v", "3", dst])
        if cp.returncode == 0 and os.path.exists(dst):
            out.append((t, dst))
        else:
            log(f"warning: frame at {hms(t)} failed: {cp.stderr.strip()[:120]}")
    return out


def describe_frames(frames: list[tuple[float, str]], question: str | None,
                    model: str, batch: int, per_frame_tokens: int) -> list[dict]:
    notes = []
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        labels = ", ".join(f"frame {j + 1} = [{hms(t)}]" for j, (t, _) in enumerate(chunk))
        prompt = f"{FRAME_PROMPT}\nTimestamps in order: {labels}."
        if question:
            prompt += f"\nThe agent's actual question is: {question!r}. Prioritise details relevant to it."
        try:
            text = describe_images([p for _, p in chunk], prompt, model,
                                   per_frame_tokens * len(chunk))
        except RuntimeError as e:
            log(f"warning: vision batch {i // batch + 1} failed: {e}")
            text = ""
        notes.append({"from": chunk[0][0], "to": chunk[-1][0],
                      "timestamps": [hms(t) for t, _ in chunk], "text": text})
        log(f"  frames {i + 1}-{i + len(chunk)}/{len(frames)} described")
    return notes


# ------------------------------------------------------------------ audio ---
def whisper_model() -> str | None:
    env = os.environ.get("WHISPER_MODEL_PATH", "").strip()
    if env:
        return env if os.path.exists(env) else None
    cands = sorted(glob.glob(WHISPER_MODEL_GLOB))
    if not cands:
        return None
    for pref in ("turbo", "large", "medium", "small", "base"):
        for c in cands:
            if pref in os.path.basename(c):
                return c
    return cands[0]


def transcribe(path: str, workdir: str, language: str) -> dict | None:
    cli = os.environ.get("WHISPER_CPP_BIN", "").strip() or shutil.which("whisper-cli")
    model = whisper_model()
    if not cli or not model:
        log("warning: whisper.cpp CLI or ggml model missing — skipping transcript "
            "(set WHISPER_CPP_BIN and WHISPER_MODEL_PATH, or install whisper.cpp "
            "and place a model under ~/.cache/whisper-models/)")
        return None
    ffmpeg = need("ffmpeg")
    wav = os.path.join(workdir, "audio-16k.wav")
    cp = run([ffmpeg, "-v", "error", "-y", "-i", path, "-vn", "-ac", "1",
              "-ar", "16000", "-f", "wav", wav], timeout=600)
    if cp.returncode != 0 or not os.path.exists(wav):
        log(f"warning: audio extract failed: {cp.stderr.strip()[:160]}")
        return None
    base = os.path.join(workdir, "transcript")
    threads = str(max(2, min(8, (os.cpu_count() or 4) - 2)))
    cp = run([cli, "-m", model, "-f", wav, "-oj", "-of", base, "-np",
              "-t", threads, "-l", language], timeout=1800)
    js = base + ".json"
    if cp.returncode != 0 or not os.path.exists(js):
        log(f"warning: whisper failed: {(cp.stderr or cp.stdout).strip()[:200]}")
        return None
    data = json.load(open(js))
    segs = []
    for s in data.get("transcription", []):
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        off = s.get("offsets", {})
        segs.append({"from": off.get("from", 0) / 1000.0,
                     "to": off.get("to", 0) / 1000.0, "text": txt})
    return {"model": os.path.basename(model), "segments": segs,
            "text": " ".join(s["text"] for s in segs)}


# -------------------------------------------------------------- synthesis ---
def synthesize(meta: dict, transcript: dict | None, notes: list[dict],
               question: str | None, text_model: str) -> str:
    parts = [f"Video metadata: {json.dumps(meta)}"]
    if transcript and transcript["segments"]:
        parts.append("Audio transcript (timestamped):\n" + "\n".join(
            f"[{hms(s['from'])}] {s['text']}" for s in transcript["segments"]))
    if notes:
        parts.append("Frame descriptions from a vision model:\n" + "\n\n".join(
            n["text"] for n in notes if n["text"]))
    ask = question or ("Summarise what happens in this video for a text-only agent: "
                       "purpose, sequence of events with timestamps, every "
                       "concrete fact shown or spoken (names, numbers, error "
                       "text, commands), and anything that looks like a problem.")
    messages = [
        {"role": "system", "content":
            "You synthesise evidence about a video for an agent that cannot see "
            "it. Use ONLY the transcript and frame descriptions provided; never "
            "invent details. Cite timestamps. If the transcript and frames "
            "disagree, say so. Use markdown headings no higher than ### (the "
            "output is embedded under an existing ## section)."},
        {"role": "user", "content": "\n\n".join(parts) + f"\n\nTask: {ask}"},
    ]
    return chat(text_model, messages, max_tokens=1500, timeout=300)


# ------------------------------------------------------------------- main ---
def render_markdown(path: str, kind: str, meta: dict | None, transcript: dict | None,
                    notes: list[dict], synthesis: str | None, frames_dir: str | None,
                    image_text: str | None, models: dict) -> str:
    out = [f"# {os.path.basename(path)}", ""]
    if kind == "image":
        out += [f"_Read by {models['vision']} (proxy vision; not the host model's eyes)_", "",
                image_text or "(empty)", ""]
        return "\n".join(out)
    if meta:
        v = meta.get("video") or {}
        line = f"- duration {hms(meta['duration'])}"
        if v:
            line += f" · {v.get('width')}x{v.get('height')} @ {v.get('fps')} fps"
        line += " · audio: " + ("yes" if meta.get("audio") else "none")
        out += ["## Metadata", line, ""]
    if synthesis:
        out += [f"## Synthesis ({models['text']})", synthesis, ""]
    if transcript is not None:
        out += [f"## Transcript ({transcript['model']}, local whisper.cpp)"]
        if transcript["segments"]:
            out += [f"[{hms(s['from'])}] {s['text']}" for s in transcript["segments"]]
        else:
            out += ["(no speech detected)"]
        out += [""]
    if notes:
        out += [f"## Frames ({len(sum((n['timestamps'] for n in notes), []))} keyframes, {models['vision']})"]
        for n in notes:
            out += [n["text"] or f"(batch {n['timestamps']} returned nothing)", ""]
    if frames_dir:
        out += [f"_Frames kept at {frames_dir} (scratch, auto-pruned after 7 days)_"]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Read images/videos/audio into text via a vision model + local whisper.cpp")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--prompt", default=None, help="specific question to answer")
    ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--min-gap", type=float, default=2.0, help="min seconds between keyframes")
    ap.add_argument("--scene", type=float, default=0.02,
                    help="scene-score floor (screen recordings peak ~0.04; camera cuts ~0.3+)")
    ap.add_argument("--every", type=float, default=None,
                    help="uniform sampling interval in seconds (overrides scene detection)")
    ap.add_argument("--batch", type=int, default=4, help="frames per vision request")
    ap.add_argument("--max-width", type=int, default=1600)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-synth", action="store_true", help="skip the text-model synthesis pass")
    ap.add_argument("--language", default="en")
    ap.add_argument("--model", default=VISION_MODEL, help="vision model id")
    ap.add_argument("--text-model", default=TEXT_MODEL, help="synthesis model id")
    ap.add_argument("--per-frame-tokens", type=int, default=350)
    ap.add_argument("--max-tokens", type=int, default=800, help="image answer budget")
    ap.add_argument("--frames-dir", default=None, help="where to keep frames (default ~/scratch/read-media/<slug>)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = ap.parse_args()

    for p in args.paths:
        if not os.path.exists(p):
            log(f"error: no such file: {p}")
            return 2
    if not token():
        log("error: no API key — set READ_MEDIA_API_KEY (or OPENAI_API_KEY)")
        return 3

    models = {"vision": args.model, "text": args.text_model}
    results = []
    rc = 0
    for path in args.paths:
        ext = os.path.splitext(path)[1].lower()
        kind = "image" if ext in IMAGE_EXT and ext != ".gif" else \
               "audio" if ext in AUDIO_EXT else "video"
        t0 = time.time()
        try:
            if kind == "image":
                text = describe_images([path], args.prompt or IMAGE_PROMPT,
                                       args.model, args.max_tokens)
                if not text:
                    rc = 4
                results.append({"path": path, "kind": kind, "text": text})
                print(render_markdown(path, kind, None, None, [], None, None, text, models)
                      if not args.json else "")
                continue

            workdir = args.frames_dir or os.path.join(SCRATCH_ROOT, f"{slugify(path)}-{int(t0)}")
            os.makedirs(workdir, exist_ok=True)
            meta = probe(path)
            log(f"{os.path.basename(path)}: {kind}, {hms(meta['duration'])}, "
                f"audio={'yes' if meta['audio'] else 'no'}")

            transcript = None
            if meta["audio"] and not args.no_audio:
                log("  transcribing audio (local whisper.cpp)…")
                transcript = transcribe(path, workdir, args.language)

            notes: list[dict] = []
            frames: list[tuple[float, str]] = []
            if kind == "video" and meta["video"]:
                hits = [] if args.every else scene_times(path, args.scene, workdir)
                times = pick_times(meta["duration"], hits, args.max_frames, args.every, args.min_gap)
                log(f"  {len(hits)} scene changes → {len(times)} keyframes")
                frames = extract_frames(path, times, workdir, args.max_width)
                notes = describe_frames(frames, args.prompt, args.model,
                                        args.batch, args.per_frame_tokens)

            synthesis = None
            if not args.no_synth and (notes or (transcript and transcript["segments"])):
                log("  synthesising…")
                try:
                    synthesis = synthesize(meta, transcript, notes, args.prompt, args.text_model)
                except RuntimeError as e:
                    log(f"warning: synthesis failed: {e}")

            if not notes and not (transcript and transcript["segments"]):
                rc = 4
            results.append({"path": path, "kind": kind, "meta": meta,
                            "transcript": transcript, "frames": notes,
                            "synthesis": synthesis, "frames_dir": workdir,
                            "seconds": round(time.time() - t0, 1)})
            if not args.json:
                print(render_markdown(path, kind, meta, transcript, notes, synthesis,
                                      workdir if frames else None, None, models))
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            log(f"error: {path}: {e}")
            rc = 1
    if args.json:
        print(json.dumps(results, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
