# read-media

A multimodal **media-to-text bridge** for text-only coding agents.

Modern coding agents are great at reading code and great at calling tools, but
many of the models behind them are **text-only**: they cannot open a screenshot,
watch a screen recording, or listen to a voice memo. `read-media` closes that
gap. Point it at an image, a video, or an audio file and it returns structured,
timestamped Markdown that a text-only model can reason and act on.

It is deliberately small and dependency-light:

- **Video** → `ffprobe` metadata + `ffmpeg` scene-change keyframes, each
  described by a vision model with a timestamp, plus a local `whisper.cpp`
  transcript of the audio track, fused into a timestamped narrative.
- **Image** → a single vision-model call that reads every visible detail.
- **Audio** → a local `whisper.cpp` transcript, optionally synthesised.

The only things that leave the machine are the frames/audio sent to the
configured chat-completions endpoint. Transcription is **local**; no audio is
uploaded for speech-to-text.

## Why

Agent workflows constantly hit media the host model cannot see:

- a user pastes a screenshot of an error dialog;
- a PR or bug report links a Loom / screen recording and asks "what broke?";
- a voice memo carries the actual requirements;
- a design review arrives as a video and you need the on-screen text verbatim.

Switching to a vision-capable model mid-task is disruptive and often not
possible. `read-media` lets the *existing* text-only agent stay in the driver's
seat and consume media as text evidence.

## Pipeline

```
                 ┌──────────────┐
 image ─────────►│ vision model │───────────────► description
                 └──────────────┘

                 ┌─────────┐   keyframes   ┌──────────────┐
 video ─────────►│ ffprobe │──────────────►│ vision model │──► per-frame notes
                 │ ffmpeg  │  (scene cut)  └──────────────┘   (timestamped)
                 └────┬────┘
                      │ audio track
                      ▼
                 ┌──────────────┐
                 │  whisper.cpp │──► timestamped transcript
                 │   (local)    │
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │ text model   │──► synthesised narrative
                 └──────────────┘

 audio ──► whisper.cpp ──► transcript ──► (optional) synthesis
```

Scene detection matters: for a screen recording most of the frame never moves,
so a fixed "cut" threshold would miss everything. `read-media` takes every
frame above a low score floor and then greedily keeps the highest-scoring ones
that are at least `--min-gap` seconds apart, always including `t=0`. If that
yields too few, it falls back to uniform sampling.

## Install

Requires Python 3.10+ and:

- `ffmpeg` / `ffprobe` on `PATH`
- a [whisper.cpp](https://github.com/ggerganov/whisper.cpp) build providing
  `whisper-cli`, plus a ggml model (e.g. `ggml-large-v3-turbo.bin`)
- an OpenAI-compatible chat-completions endpoint with a vision-capable model

No third-party Python packages are needed — everything is the standard library.

```bash
git clone https://github.com/rudysanchez-mfb/read-media.git
cd read-media
python3 scripts/read_media.py --help
```

## Configuration

All configuration is via environment variables. Nothing is hard-coded and no
secrets are embedded.

| Variable | Default | Purpose |
|----------|---------|---------|
| `READ_MEDIA_API_KEY` | — | API key for the chat-completions endpoint (falls back to `OPENAI_API_KEY`) |
| `READ_MEDIA_API_BASE` | `https://api.openai.com` | Base URL of an OpenAI-compatible API |
| `READ_MEDIA_VISION_MODEL` | `gpt-4o-mini` | Model that reads images/frames (must accept `image_url` parts) |
| `READ_MEDIA_TEXT_MODEL` | `gpt-4o-mini` | Model that synthesises the narrative |
| `WHISPER_CPP_BIN` | `whisper-cli` on `PATH` | Path to the whisper.cpp CLI |
| `WHISPER_MODEL_PATH` | a `ggml-*.bin` under `~/.cache/whisper-models/` | Path to the whisper model |
| `READ_MEDIA_SCRATCH_DIR` | `~/scratch/read-media` | Where keyframes and the 16 kHz wav are written |

## Usage

```bash
# Image
python3 scripts/read_media.py screenshot.png

# Video with a specific question
python3 scripts/read_media.py demo.mp4 --prompt "What error appears on screen and what command was run?"

# Audio
python3 scripts/read_media.py memo.m4a

# Machine-readable output
python3 scripts/read_media.py demo.mp4 --json > out.json
```

| Flag | Default | Notes |
|------|---------|-------|
| `--prompt TEXT` | generic | Threaded into both frame description and synthesis |
| `--max-frames N` | 12 | Cap on keyframes |
| `--scene F` | 0.02 | Scene-score floor; screen recordings peak ~0.04, camera cuts 0.3+ |
| `--min-gap S` | 2.0 | Minimum seconds between keyframes |
| `--every S` | off | Force uniform sampling, skipping scene detection |
| `--batch N` | 4 | Frames per vision request |
| `--max-width PX` | 1600 | Downscale keyframes before sending |
| `--no-audio` | off | Skip transcription |
| `--no-synth` | off | Skip the text-model synthesis pass |
| `--language L` | `en` | Whisper language |
| `--model ID` | `READ_MEDIA_VISION_MODEL` | Vision model override |
| `--text-model ID` | `READ_MEDIA_TEXT_MODEL` | Synthesis model override |
| `--per-frame-tokens N` | 350 | Answer budget per frame |
| `--max-tokens N` | 800 | Answer budget for a single image |
| `--frames-dir DIR` | scratch | Where to keep extracted frames |
| `--json` | off | Emit JSON instead of Markdown |

### Output

Markdown sections in order:

- `## Metadata` — duration, resolution, fps, audio present
- `## Synthesis (<model>)` — fused, timestamped narrative (derived summary)
- `## Transcript (<model>, local whisper.cpp)` — `[m:ss]` segments (primary evidence)
- `## Frames (N keyframes, <model>)` — `### [m:ss]` per-keyframe descriptions

For precision, quote from **Transcript** and **Frames** — those are the raw
evidence; the synthesis is a convenience summary.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | success |
| 1 | gateway/tool error |
| 2 | bad path |
| 3 | no API key |
| 4 | model returned nothing |

Warnings (missing whisper model, a failed vision batch, no scene changes) go to
stderr and the run continues with whatever lanes succeeded.

## Using as an agent skill

`SKILL.md` is a drop-in skill definition for agents that support the
`SKILL.md` convention. It carries the frontmatter (`name`, `description`) and
the operational procedure an agent needs to invoke `scripts/read_media.py`
correctly. The script itself is agent-agnostic — anything that can run a shell
command can use it.

## Security and privacy

- **No secrets in the tree.** Credentials come from the environment only.
- **No personal paths in the tree.** All paths are configurable; defaults are
  generic home-relative locations.
- **Local speech-to-text.** Audio is transcribed on-device by `whisper.cpp`;
  only images/keyframes are sent to the configured vision endpoint.
- **Scratch output** is written outside any repository and is safe to delete.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
