---
name: read-media
description: >-
  Converts images, videos, and audio files into TEXT context a text-only model
  can act on. Images go to a vision-capable model over an OpenAI-compatible
  chat API; videos are split into scene-change keyframes (ffmpeg) that the
  vision model describes with timestamps, the audio track is transcribed
  locally with whisper.cpp, and a text model synthesises a timestamped
  narrative. Use whenever the current model cannot see media — a pasted
  screenshot, a screen recording, a demo/PR video, a voice memo — or when the
  user asks "what happens in this video", "what does this say", or wants to act
  on media contents without switching models.
---

# Read Media

Turns a media file into evidence a text-only agent can reason on. A vision
model does the seeing, whisper.cpp the hearing; the calling agent gets
structured text.

## Configuration

Set these environment variables (all optional, with sane defaults):

| Variable | Default | Purpose |
|----------|---------|---------|
| `READ_MEDIA_API_KEY` | — | API key for the chat-completions endpoint (falls back to `OPENAI_API_KEY`) |
| `READ_MEDIA_API_BASE` | `https://api.openai.com` | Base URL of an OpenAI-compatible API |
| `READ_MEDIA_VISION_MODEL` | `gpt-4o-mini` | Model that reads images (must accept `image_url` parts) |
| `READ_MEDIA_TEXT_MODEL` | `gpt-4o-mini` | Model that synthesises the narrative |
| `WHISPER_CPP_BIN` | `whisper-cli` on `PATH` | Path to the whisper.cpp CLI |
| `WHISPER_MODEL_PATH` | a `ggml-*.bin` under `~/.cache/whisper-models/` | Path to the whisper model |
| `READ_MEDIA_SCRATCH_DIR` | `~/scratch/read-media` | Where frames/wav are written |

`--model` and `--text-model` override the two model variables per run.

| Input | Pipeline | Output |
|-------|----------|--------|
| image (`png jpg webp …`) | 1 vision call | description / answer |
| video (`mp4 mov mkv webm gif …`) | ffprobe → keyframes → vision (batched, timestamped) + whisper transcript → synthesis | metadata, synthesis, timestamped transcript, per-frame notes |
| audio (`mp3 wav m4a …`) | whisper transcript → synthesis | transcript + synthesis |

Image/audio payloads go to the configured API base; nothing else leaves the
machine. Whisper runs locally (`whisper.cpp`). Frames and the 16 kHz wav land
in the scratch directory (default `~/scratch/read-media/<slug>-<epoch>/`,
auto-pruned after 7 days — never inside a repo).

## When to use

- `Read` on an image fails or comes back blank because the host model has no
  vision (any text-only seat).
- The user drops a video/screen recording and asks what it shows, what was
  said, what broke, or wants a PR/ticket/summary written from it.
- The user points at an audio file / voice memo.

Not for generating or editing media. It only transcribes what the proxy models
perceive.

## Procedure

1. **Get a local path.** Pasted images have one in the session. URLs/blobs:
   save under a scratch directory first.
2. **Run the helper.** A specific `--prompt` beats the generic one every time;
   it is threaded into both the frame descriptions and the synthesis.

   ```bash
   python3 scripts/read_media.py <path>... [--prompt "question"]
   ```

   Useful knobs:

   | Flag | Default | When |
   |------|---------|------|
   | `--max-frames N` | 12 | longer/denser video → 16–24; quick look → 6 |
   | `--every SECONDS` | off | force uniform sampling (skips scene detection) |
   | `--scene F` / `--min-gap S` | 0.02 / 2.0 | scene floor and min spacing; screen recordings score ~0.04, camera cuts 0.3+ |
   | `--no-audio` | off | silent video, or transcript not needed |
   | `--no-synth` | off | you only want raw transcript + frame notes |
   | `--json` | off | machine-readable result (transcript segments, frame batches, synthesis) |
   | `--model` / `--text-model` | `gpt-4o-mini` / `gpt-4o-mini` | vision seat (reads images) / synthesis seat; any model on the API base works, the vision one must accept `image_url` parts |

3. **Reason on stdout.** Sections in order: Metadata, Synthesis (text model),
   Transcript (timestamped `[m:ss]`), Frames (`### [m:ss]` per keyframe). The
   synthesis is a derived summary — when precision matters (exact error text,
   numbers), quote from the Transcript/Frames sections, which are the primary
   evidence.
4. **Say which model saw it.** The reading is a proxy's, not the host model's.
   Report it that way.

## Timing

Whisper dominates the cost, so a multi-minute screen recording can take a few
minutes wall time end to end; a single image read through the API is usually
seconds to ~1 minute depending on the model. Budget accordingly and run long
videos in the background, then read the output file.

## Placement

stdout only by default. If the user wants the reading kept, redirect it to a
deliverables directory — a deliverable, not scratch.

## Failure handling

| Signal | Meaning / fix |
|--------|---------------|
| exit 2 | path missing |
| exit 3 | `READ_MEDIA_API_KEY` (or `OPENAI_API_KEY`) unset in this environment |
| exit 4 | model returned nothing — larger `--per-frame-tokens`/`--max-tokens`, or a more specific `--prompt` |
| exit 1 `gateway HTTP 4xx` | bad model id or rejected request; the body says which |
| `warning: whisper.cpp CLI or ggml model missing` | transcript skipped; set `WHISPER_CPP_BIN` and `WHISPER_MODEL_PATH`, or install whisper.cpp and add a model under `~/.cache/whisper-models/` |
| `0 scene changes → N keyframes` | static recording; uniform fallback kicked in — normal |

Warnings go to stderr and the run continues with whatever lanes succeeded; the
markdown says which sections are present.
