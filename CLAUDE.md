# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video animal censoring system that detects animals using YOLOv8 on CUDA and censors them with configurable styles. Three modes:

- **Batch processing** (`censor_animals.py`): Process video files offline
- **Live streaming** (`stream_animals_v3.py`): Multi-stream IPTV processor — auto-discovers channels from an M3U playlist using configurable keyword filters (default: "hallmark"), each with an independent pipeline, auto-start/stop on client activity, web dashboard
- **VOD movies** (`stream_animals_v3.py`): On-demand movie censoring — parses ~16K movies from the same M3U, spins up pipeline when IPTV app plays a movie

## Running

```bash
# Batch process a video file
py -3 censor_animals.py "input.mkv"
py -3 censor_animals.py "input.mkv" --output censored.mkv --confidence 0.2 --padding 30

# Start the live multi-stream server via tray launcher (recommended)
# Double-click NoAnimals.bat or NoAnimals.vbs

# Or start the server directly (dashboard on port 8080)
py -3 stream_animals_v3.py
```

Note: Use `py -3` on this Windows system, not `python`. FFmpeg must be on PATH or at the WinGet install path (`%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin`).

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
             [Flask :8080] → dashboard + /stream/<slug>/<slug>.m3u8
```

Port allocation: live stream 0 = 19876-19878, stream 1 = 19886-19888, stream 2 = 19896-19898. VOD pipelines use 19976+ (`VOD_PORT_BASE`). Audio on port+0, decoder video on port+1, encoder video on port+2.

### VOD Pipeline

Same `StreamInstance` class with `is_vod=True`. Key differences from live:
- **HLS flags**: `hls_list_size=0` + `independent_segments` (keeps all segments, no rolling window)
- **Synthetic m3u8**: `_serve_synthetic_m3u8()` generates a full-length playlist listing ALL segments (based on `total_duration` from probe) so IPTV apps show a complete seek bar. Includes `#EXT-X-DISCONTINUITY` tags at seek boundaries.
- **Seeking**: `seek_restart()` kills the current pipeline and restarts with `-ss {offset}` on the decoder and `-output_ts_offset {offset}` on the encoder so PTS timestamps remain continuous. Thread-safe via `_seek_lock`. Probe results are cached (`_cached_probe`) to save ~10s on restarts.
- **No hardcoded decoder**: Omits `-c:v h264_cuvid` to auto-detect codec (movies may be H.265/HEVC)
- **No letterboxing**: Movies are scaled directly to `{W}x{H}` regardless of source aspect ratio
- **Overlay positioned bottom-left with margin**: VOD overlay is at (16, H-180-overlay_height) — 180px from the bottom edge to clear letterbox bars on widescreen movies (2.39:1 bars ≈ 138px). Live streams use bottom-left adjusted for `pad_y`.
- **English audio track**: VOD decoder selects the English audio track (`-map 0:a:{eng_idx}`) based on language tags from probe, falling back to first track. Live streams always use first audio track.
- **No subtitles/CC**: VOD decoder adds `-sn` to strip all subtitle and closed caption streams.
- **EOF handling**: Sets `self.completed = True` and `status = "completed"` on natural stream end
- **Cleanup with generation check**: 5-min delayed cleanup after completion, but checks `_pipeline_gen` counter before deleting — if the pipeline was restarted (e.g. resume), cleanup is cancelled.
- **Auto-restart on resume**: When an IPTV app requests an m3u8 for a VOD whose pipeline is in "listening"/"stopped"/"error" status, the `vod_file()` handler auto-restarts the pipeline (protected by `_seek_lock` to avoid racing with seeks).
- **Segment streaming**: Segment responses use a Flask generator that waits up to 60s for the segment file to appear, keeping the HTTP connection alive during seek delays

### Launch Chain

`NoAnimals.bat` (or `.vbs`) → `noanimals_tray.pyw` (system tray icon with Start/Stop/Dashboard menu) → `stream_animals_v3.py` (subprocess with `CREATE_NO_WINDOW`). The `.bat` relaunches itself minimized to avoid a console popup. The `.vbs` is a fully silent alternative.

### Key Classes
- `StreamInstance`: Per-stream state (ports, stats, stop_event, pipeline_procs, client tracking). Methods: `start()`, `stop()`, `run_pipeline()`, `run_pipeline_stdout()`, `seek_restart()`, `start_stdout()`, `touch_client()`. Flags: `is_vod`, `completed`, `xui_id`. Also stores `last_frame_jpeg` (latest processed frame as JPEG bytes for dashboard preview). `_pipeline_gen` counter tracks pipeline restarts for safe delayed cleanup.
- `BoxTracker`: IoU-based temporal smoothing — persists boxes for N frames after detection lost, averages coordinates over a sliding window. New track IDs used for unique animal instance counting.

### Shared State (all lock-protected)
- `config` dict: Detection/censoring settings, persisted to `noanimals_settings.json` on every change. Protected by `config_lock`. Includes `channel_filters` (keyword list for M3U discovery) and `excluded_channels` (slugs to hide from dashboard).
- `streams` dict (`slug → StreamInstance`): Live stream registry. Protected by `streams_lock`.
- `m3u_state` dict: M3U playlist URL, discovered channels list, fetch status. Protected by `m3u_lock`.
- `vod_catalog` dict (`xui_id → {name, url, logo, group}`): All discovered VOD movies. Protected by `vod_lock`.
- `vod_active` dict (`xui_id → StreamInstance`): Currently running VOD pipelines. Protected by `vod_lock`.

### M3U Parsing (`fetch_m3u`)

Reads the entire hivecast M3U playlist (same URL for live + VOD):
- **Live channels**: Matched by configurable `channel_filters` keywords (default: `["hallmark"]`). Any channel whose EXTINF line contains any filter keyword (case-insensitive) is discovered. Extracts `xui-id`, `tvg-id`, `tvg-name`, `tvg-logo`, `group-title`. Channels in `excluded_channels` config are filtered out by `_sync_streams()`.
- **VOD movies**: Identified by `xui-id` attribute + URL ending in `#.mkv`/`#.mp4`/`#.avi`. TV show episodes (matching `S\d+E\d+` in name) are filtered out. URL fragment (`#.ext`) stripped before storage. Groups keep their `│` prefix (e.g. `│ Thriller`).
- Fetched synchronously on boot, then every 3 days or on manual "Fetch Now".

### Xtream Codes API Compatibility

Full XC API for IPTV app integration (TiviMate, XCIPTV, iPTV Smarters, tvOS apps, etc.):
- `/player_api.php` (GET/POST/JSON): Auth (returns `auth: 1` for any credentials), `get_live_categories`, `get_live_streams`, `get_vod_categories`, `get_vod_streams`, `get_vod_info`, `get_series`, `get_series_categories`, `get_series_info`
- `/panel_api.php`: Alias for `/player_api.php`
- `/get.php`: M3U playlist (calls `noanimals_m3u()`)
- `/xmltv.php`: Empty XMLTV EPG stub
- `/live/<user>/<pass>/<stream_id>.<ext>`: Proxies to live HLS pipeline
- `/movie/<user>/<pass>/<stream_id>.<ext>`: Proxies to VOD HLS pipeline
- Segment paths under `/live/` and `/movie/` are handled by catch-all `<path:extra>` routes
- Auth response includes `server_info.process: true` and fully populated fields (required by tvOS apps)
- Stream objects include `is_adult`, `category_ids` (array), and other fields expected by strict IPTV clients
- Series endpoints return empty lists (no series support, but apps expect the endpoints to exist)
- Unknown actions return empty `{}` instead of 404 (some apps treat 404 as login failure)

HLS playlists served through XC routes have segment URLs rewritten to absolute paths (`http://host:port/stream/...` or `/vod/...`) so players resolve segments correctly regardless of the request URL.

### Censoring System

Five modes, all with rounded-corner masks (radius scales with box size):
- **black**: Zeros out pixels (default look)
- **blur**: Downscale to 1/40th → upscale → Gaussian blur → 70% flood toward average color
- **fine_blur**: Downscale to 1/12th → moderate Gaussian blur → 50% flood toward average color (finer grain than blur, still unrecognizable)
- **pixelate**: Downscale to max 10x10 → 50% flood toward average color → bilinear upscale
- **color_match**: Fill with region's average color

### Overlay System

Six options, pre-rendered as BGRA numpy arrays at module load (zero per-frame PIL cost):
- **none**, **text** ("NoAnimals"), **graphic** (dog emoji + prohibition sign), **petfree** ("PET FREE TV" badge), **paw** (paw emoji + prohibition sign), **censored** (red stamp)

Applied via `apply_overlay()` using alpha compositing. Live streams: bottom-left (16px margin, adjusted for `pad_y` letterbox). VOD: bottom-left, 180px from bottom edge to clear widescreen letterbox bars.

### Dashboard UI

- **Channel cards**: Collapsible — idle channels show compact header only (name, status dot, remove button). Active/probing channels expand to show full stats (FPS, detections, frames, uptime, viewers, resolution), animal counts, preview image, and HLS URL.
- **Channel filters**: Tag/chip UI below channel cards for adding/removing M3U filter keywords. Triggers re-fetch on change.
- **Excluded channels**: Removed channels appear as re-addable tags below the filter bar.
- **Topbar**: Compact M3U URL + Xtream server/user/pass display.

### Animal Detection & Counting

- `ANIMAL_CLASS_IDS`: COCO classes 14-23 (bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe)
- Runs YOLO every 2nd frame (`DETECT_EVERY=2`) with FP16, `imgsz=480`
- Unique instance counting via BoxTracker track IDs, per-species counts in dashboard

## System Dependencies

- **FFmpeg 8.0.1** (full build with NVENC/NVDEC/cuvid): On PATH or at WinGet install path
- **NVIDIA GPU with CUDA**: Required for YOLO inference, NVDEC decode, NVENC encode
- **Python packages**: `ultralytics`, `opencv-python`, `numpy`, `flask`, `Pillow`, `pystray`

## Known Constraints

- **NVENC session limit**: Consumer GPUs cap at 2 simultaneous encode sessions. `MAX_STREAMS = 10` + `MAX_VOD_PIPELINES = 1` but total concurrent encodes limited by GPU.
- **FFmpeg 8.0.1 quirks**: `-live_start_index` does not exist. Avoid it.
- **YOLO `model.half()`**: Crashes silently. Use `half=True` in `model.predict()` instead.
- **`-hwaccel_output_format cuda`**: Required with `scale_cuda` to keep frames on GPU.
- **Dashboard HTML is embedded** in `stream_animals_v3.py` as `DASHBOARD_HTML` string. Changes require server restart.
- **Resolution fixed at compile time**: `W, H = 1920, 1080` is a module-level constant.
- **VOD codec auto-detect**: VOD omits `-c:v h264_cuvid` so FFmpeg auto-selects the right cuvid decoder. Live streams hardcode h264_cuvid.
- **VOD seek latency**: Seeking restarts the full pipeline (~12-22s) because the remote source must be re-opened with `-ss`. Probe caching helps (~10s saved).
- **VOD seek decoder flags**: Do NOT add `-analyzeduration 0`, `-probesize 500000`, or `-fflags +fastseek+nobuffer` to the decoder for seeks — causes zero-frame output. Only `-ss` before `-i` is needed.
- **VOD PTS continuity**: After seek, encoder must use `-output_ts_offset {seek_time}` so PTS timestamps match the synthetic playlist positions. Without this, players see a black screen after seeking.
- **VOD cleanup race condition**: Delayed cleanup threads must check `_pipeline_gen` before deleting output directories. Without this, a cleanup from a previous pipeline run can delete segments from a newly restarted pipeline.
- **VOD overlay positioning**: Must be at least 180px from the bottom edge to avoid landing on player-added letterbox bars for widescreen movies. Do NOT use simple bottom-left (16px margin) for VOD — causes letterboxing artifacts.

## Key Design Decisions
- **TCP sockets for inter-process video/audio**: Windows anonymous pipes have ~4KB OS buffers, too slow for ~6MB raw 1080p frames.
- **Audio bypasses Python entirely**: Decoder FFmpeg sends PCM audio (48kHz, 2ch, s16le) directly to encoder FFmpeg via TCP:port+0.
- **Encoder opens audio TCP (input 0) before video TCP (input 1)**: FFmpeg opens inputs sequentially and blocks.
- **Auto-start/stop**: Streams start when client requests HLS playlist, stop after 30s idle (watchdog thread checks every 5s). Applies to both live and VOD.
- **Single NVENC session for VOD**: `MAX_VOD_PIPELINES = 1` leaves room for live streams.
- **No VOD catalog persistence**: Rebuilt from M3U on each boot (~2s parsing for 16K movies).
- **M3U absolute URL rewriting**: HLS playlists served via XC API routes rewrite relative segment filenames to absolute URLs so players fetch segments from correct `/stream/` or `/vod/` paths.
- **XC API accepts any credentials**: No auth validation — returns `auth: 1` for all username/password combos.
- **XC API compatibility**: Auth response includes `process: true`, `https_port`, `rtmp_port`; stream objects include `is_adult`, `category_ids`; series endpoints return empty lists; unknown actions return `{}` not 404. These fields are required by strict tvOS IPTV apps.
- **Preview frames**: Every 30th processed frame is JPEG-encoded (quality 50) and stored on `StreamInstance.last_frame_jpeg`. Dashboard polls `/api/preview/<slug>` with cache-busting timestamp to show live preview images.
- **Configurable channel filters**: `channel_filters` config array replaces hardcoded "hallmark" check. Dashboard provides tag/chip UI for adding/removing keywords, triggers M3U re-fetch on change.
- **Per-channel exclusion**: `excluded_channels` config array lets users hide individual discovered channels without removing the filter keyword. Channels can be re-added from the dashboard.
- **Pipeline generation counter**: `_pipeline_gen` on `StreamInstance` increments each `run_pipeline()` call. Delayed cleanup threads capture the generation at scheduling time and skip deletion if the pipeline was restarted since.
- **VOD auto-restart on resume**: `vod_file()` detects dead/stale VOD pipelines and restarts them when an IPTV app requests the m3u8 again (e.g. resume playback). Protected by `_seek_lock` to avoid racing with concurrent seeks.
- **VOD English audio preference**: `probe_stream()` identifies the English audio track by language tag (`eng`/`en`/`english`), VOD decoder maps that track specifically. Falls back to first audio track if no English track found. Live streams always use first audio track.
- **VOD subtitle stripping**: VOD decoder uses `-sn` to disable all subtitle/CC streams. Prevents closed captions from appearing in transcoded output.

## Server API

### Dashboard & Playlist
| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Web dashboard |
| `/noanimals.m3u` | GET | Combined M3U playlist (live + VOD, hivecast format with `│` prefix for VOD groups) |
| `/stream/<slug>/<file>` | GET | HLS playlist/segments (auto-starts pipeline) |

### Live Stream Control
| Route | Method | Purpose |
|-------|--------|---------|
| `/api/start/<slug>` | POST | Start specific stream |
| `/api/stop/<slug>` | POST | Stop specific stream |
| `/api/channel/remove/<slug>` | POST | Exclude a channel (stops stream, adds to `excluded_channels`) |
| `/api/channel/add/<slug>` | POST | Re-include an excluded channel (removes from `excluded_channels`, re-syncs streams) |

### VOD
| Route | Method | Purpose |
|-------|--------|---------|
| `/vod/<xui_id>/<filename>` | GET | VOD HLS playlist/segments (auto-starts pipeline, auto-restarts on resume) |
| `/api/vod/catalog` | GET | Paginated movie catalog (`?q=search&group=Thriller&page=1&per_page=50`) |
| `/api/vod/groups` | GET | List of genre groups |
| `/api/vod/active` | GET | Running VOD pipelines |
| `/api/vod/stop/<xui_id>` | POST | Stop a VOD pipeline |
| `/api/vod/preview/<xui_id>` | GET | Latest processed frame as JPEG (204 if unavailable) |

### Settings & Stats
| Route | Method | Purpose |
|-------|--------|---------|
| `/api/settings` | POST | Update shared config (accepts `channel_filters`, `excluded_channels`, detection/censor settings) |
| `/api/stats` | GET | Stats JSON for all streams + `_config` + `_vod` + `_discovered` (all matched channels) |
| `/api/overlay/preview` | GET | Base64 PNG data URIs for all overlay types |
| `/api/preview/<slug>` | GET | Latest processed frame as JPEG for live stream (204 if unavailable) |
| `/api/m3u` | GET/POST | M3U source config / update URL |
| `/api/m3u/fetch` | POST | Trigger M3U re-fetch |

### Xtream Codes API
| Route | Method | Purpose |
|-------|--------|---------|
| `/player_api.php` | GET/POST | XC auth + actions (get_live_streams, get_vod_streams, get_series, etc.) |
| `/panel_api.php` | GET/POST | Alias for player_api.php |
| `/get.php` | GET | XC M3U playlist |
| `/xmltv.php` | GET | Empty XMLTV EPG |
| `/live/<user>/<pass>/<stream_id>.<ext>` | GET | XC live stream proxy |
| `/movie/<user>/<pass>/<stream_id>.<ext>` | GET | XC VOD stream proxy |
