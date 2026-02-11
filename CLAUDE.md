# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video animal censoring system that detects animals using YOLOv8 on CUDA and censors them with configurable styles. Two modes:

- **Batch processing** (`censor_animals.py`): Process video files offline
- **Live streaming** (`stream_animals_v3.py`): Multi-stream IPTV processor — auto-discovers up to 3 Hallmark channels from an M3U playlist, each with an independent pipeline, auto-start/stop on client activity, web dashboard

## Running

```bash
# Batch process a video file
py -3 censor_animals.py "input.mkv"
py -3 censor_animals.py "input.mkv" --output censored.mkv --confidence 0.2 --padding 30

# Start the live multi-stream server via tray launcher (recommended)
# Double-click NoAnimals.bat or NoAnimals.vbs

# Or start the server directly (dashboard on port 5000)
py -3 stream_animals_v3.py
```

Note: Use `py -3` on this Windows system, not `python`.

## Architecture

### Batch Pipeline (`censor_animals.py`)
Single-threaded: OpenCV reads frames → YOLO detect → draw black boxes → OpenCV write temp file → FFmpeg mux with original audio/subtitles using NVENC.

### Live Streaming Pipeline (`stream_animals_v3.py`)

Multi-stream architecture. Each discovered channel gets an independent `StreamInstance` with its own pipeline:

```
HLS Source → [Decoder FFmpeg: NVDEC + scale_cuda + fps=30]
                    ↓ TCP:port+1 (raw BGR24 1920x1080)
             [Python: YOLO FP16 @ 480px, every 2nd frame]
                    ↓ TCP:port+2
             [Encoder FFmpeg: video TCP + audio TCP:port+0] → HLS segments
                    ↓
             [Flask :5000] → dashboard + /stream/<slug>/<slug>.m3u8
```

Port allocation per stream: stream 0 = 19876-19878, stream 1 = 19886-19888, stream 2 = 19896-19898. Audio on port+0, decoder video on port+1, encoder video on port+2.

### Launch Chain

`NoAnimals.bat` (or `.vbs`) → `noanimals_tray.pyw` (system tray icon with Start/Stop/Dashboard menu) → `stream_animals_v3.py` (subprocess with `CREATE_NO_WINDOW`). The `.bat` relaunches itself minimized to avoid a console popup. The `.vbs` is a fully silent alternative.

### Key Classes
- `StreamInstance`: Per-stream state (ports, stats, stop_event, pipeline_procs, client tracking). Methods: `start()`, `stop()`, `run_pipeline()`, `touch_client()`.
- `BoxTracker`: IoU-based temporal smoothing — persists boxes for N frames after detection lost, averages coordinates over a sliding window. New track IDs used for unique animal instance counting.

### Shared State (all lock-protected)
- `config` dict: Detection/censoring settings, persisted to `noanimals_settings.json` on every change, loaded on startup. Protected by `config_lock`.
- `streams` dict (`slug → StreamInstance`): Stream registry. Protected by `streams_lock`.
- `m3u_state` dict: M3U playlist URL, discovered channels list, fetch status. Protected by `m3u_lock`.

### Censoring System

Four modes, all with rounded-corner masks (radius scales with box size):
- **black**: Zeros out pixels (default look)
- **blur**: Downscale to 1/40th → upscale → Gaussian blur → 70% flood toward average color. Destroys all shape.
- **pixelate**: Downscale to max 10x10 → 50% flood toward average color → bilinear upscale (soft mosaic)
- **color_match**: Fill with region's average color

Applied via `cv2` (GaussianBlur, resize) and numpy. The `rounded_rect_mask()` helper generates per-region alpha masks.

### Overlay System

Six options, pre-rendered as BGRA numpy arrays at module load (zero per-frame PIL cost):
- **none**: No watermark
- **text**: "NoAnimals" matching dashboard header style (red "No" + white "Animals", Segoe UI Bold)
- **graphic**: Dog emoji (🐕 via Segoe UI Emoji font) + red prohibition sign
- **petfree**: "PET FREE TV" white text in dark rounded pill badge
- **paw**: Paw emoji (🐾) + prohibition sign
- **censored**: Red "CENSORED" rubber stamp

Applied bottom-left (16px margin) via `apply_overlay()` using alpha compositing. Previews served as base64 PNGs via `/api/overlay/preview`.

### Animal Detection & Counting

- `ANIMAL_CLASS_IDS`: COCO classes 14-23 (bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe)
- `ANIMAL_CLASS_NAMES`: Maps class IDs to display names
- Runs YOLO every 2nd frame (`DETECT_EVERY=2`) with FP16, `imgsz=480`
- Unique instance counting: only increments when BoxTracker creates a new track (`next_id` comparison), not on every frame a detection appears
- Per-species counts shown as tag pills in dashboard stream cards

### Key Design Decisions
- **TCP sockets for inter-process video/audio**: Windows anonymous pipes have ~4KB OS buffers, too slow for ~6MB raw 1080p frames. TCP with large SO_RCVBUF/SO_SNDBUF buffers.
- **Audio bypasses Python entirely**: Decoder FFmpeg sends PCM audio (48kHz, 2ch, s16le) directly to encoder FFmpeg via TCP:port+0.
- **Encoder opens audio TCP (input 0) before video TCP (input 1)**: FFmpeg opens inputs sequentially and blocks. Port readiness verified with non-consuming bind check.
- **GPU filter chain on decoder**: `fps=30,scale_cuda=W:H,hwdownload,format=nv12` — frame drop and scale on GPU before CPU transfer.
- **Auto-start/stop**: Streams start when client requests HLS playlist, stop after 30s idle (watchdog thread checks every 5s).
- **M3U fetched synchronously on boot** (so streams exist before Flask starts), then every 3 days or on manual "Fetch Now".
- **Settings persistence**: `noanimals_settings.json` written on every `/api/settings` POST, loaded on startup with fallback to defaults.
- **Model hot-swap**: Pipeline loop reads `config["model"]` each frame; on change, reloads YOLO and resets BoxTracker.

### Shared Helpers
- `iou()`: Intersection-over-union
- `slugify()`: Channel name → URL slug
- `rounded_rect_mask()`: Float32 mask with rounded corners for censor blending
- `apply_overlay()`: BGRA alpha composite onto BGR frame
- `recv_exact()`: Read exactly N bytes from TCP into buffer
- `drain_stderr()`: Daemon thread to prevent subprocess pipe deadlock

## System Dependencies

- **FFmpeg** (full build with NVENC/NVDEC/cuvid): Must be on PATH or set `FFMPEG_PATH` env var
- **NVIDIA GPU with CUDA**: Required for YOLO inference, NVDEC decode, NVENC encode
- **Python packages**: `ultralytics`, `opencv-python`, `numpy`, `flask`, `Pillow`, `pystray`

## Known Constraints

- **NVENC session limit**: Consumer GPUs cap at 2 simultaneous encode sessions. `MAX_STREAMS = 3` but only 2 can run pipelines concurrently without the unlock patch.
- **FFmpeg 8.0.1 quirks**: `-live_start_index` does not exist. Avoid it.
- **YOLO `model.half()`**: Crashes silently. Use `half=True` in `model.predict()` instead.
- **`-hwaccel_output_format cuda`**: Required with `scale_cuda` to keep frames on GPU.
- **Dashboard HTML is embedded** in `stream_animals_v3.py` as `DASHBOARD_HTML` string. Changes require server restart.
- **M3U filter**: Only discovers channels with "hallmark" in the name (case-insensitive).
- **Resolution fixed at compile time**: `W, H = 1920, 1080` is a module-level constant, not runtime configurable.

## Live Stream Server API

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Web dashboard |
| `/noanimals.m3u` | GET | M3U playlist of all censored streams |
| `/stream/<slug>/<file>` | GET | HLS playlist/segments (auto-starts pipeline) |
| `/api/start/<slug>` | POST | Start specific stream |
| `/api/stop/<slug>` | POST | Stop specific stream |
| `/api/settings` | POST | Update shared config (confidence, padding, model, overlay, censor_mode, etc.) |
| `/api/stats` | GET | Stats JSON for all streams + `_config` with current settings |
| `/api/overlay/preview` | GET | Base64 PNG data URIs for all overlay types |
| `/api/m3u` | GET | M3U source config and discovered channels |
| `/api/m3u` | POST | Update M3U URL `{"m3u_url": "...", "fetch": true}` |
| `/api/m3u/fetch` | POST | Trigger M3U re-fetch |
