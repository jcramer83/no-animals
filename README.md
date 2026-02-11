![NoAnimals Banner](banner.png)

# NoAnimals

Real-time video animal censoring system. Detects animals using YOLOv8 on CUDA and censors them with configurable styles. Supports both batch file processing and live IPTV stream processing with a web dashboard.

## Features

- **Two processing modes**: Batch (offline file processing) and live (multi-stream IPTV with HLS output)
- **4 censoring styles**: Black box, heavy blur, pixelate, or color-match fill — all with rounded corners
- **6 overlay watermarks**: Text, graphic, "PET FREE TV" badge, paw, "CENSORED" stamp, or none
- **Multi-stream**: Auto-discovers up to 3 channels from an M3U playlist, each with an independent pipeline
- **Auto-start/stop**: Streams start on first client request and stop after 30 seconds of inactivity
- **Web dashboard**: Configure settings, monitor streams, view per-species animal counts
- **GPU-accelerated**: NVDEC decode, CUDA-based YOLO inference, NVENC encode
- **System tray launcher**: Start/stop server and open dashboard from the tray icon

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support
- FFmpeg with NVENC/NVDEC/cuvid support (must be on PATH or set `FFMPEG_PATH` env var)
- Python packages:
  ```
  pip install ultralytics opencv-python numpy flask Pillow pystray
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/no-animals.git
   cd no-animals
   ```

2. Install Python dependencies:
   ```bash
   pip install ultralytics opencv-python numpy flask Pillow pystray
   ```

3. Ensure FFmpeg (with NVENC/NVDEC) is on your PATH. If it's installed elsewhere, set the `FFMPEG_PATH` environment variable to the full path of the `ffmpeg` executable.

4. A YOLO model (e.g. `yolov8n.pt`) will be downloaded automatically on first run, or you can place one in the project directory.

## Usage

### Batch Processing

Process a video file, censoring all detected animals:

```bash
python censor_animals.py "input.mkv"
python censor_animals.py "input.mkv" --output censored.mkv --confidence 0.2 --padding 30
```

### Live Streaming Server

Start the multi-stream server (dashboard at http://localhost:5000):

```bash
# Via system tray launcher (Windows)
# Double-click NoAnimals.bat or NoAnimals.vbs

# Or start directly
python stream_animals_v3.py
```

On first run, configure your M3U playlist URL via the web dashboard. The server will discover Hallmark channels and create censored HLS streams.

Point any video player at `http://localhost:5000/noanimals.m3u` to watch the censored streams.

## API

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Web dashboard |
| `/noanimals.m3u` | GET | M3U playlist of all censored streams |
| `/stream/<slug>/<file>` | GET | HLS playlist/segments (auto-starts pipeline) |
| `/api/start/<slug>` | POST | Start specific stream |
| `/api/stop/<slug>` | POST | Stop specific stream |
| `/api/settings` | POST | Update detection/censoring settings |
| `/api/stats` | GET | Stats JSON for all streams |
| `/api/overlay/preview` | GET | Base64 PNG previews for all overlay types |
| `/api/m3u` | GET | M3U source config and discovered channels |
| `/api/m3u` | POST | Update M3U URL |
| `/api/m3u/fetch` | POST | Trigger M3U re-fetch |

## Configuration

All settings are configurable via the web dashboard and persisted to `noanimals_settings.json`:

- **Confidence threshold**: Minimum detection confidence (0.0-1.0)
- **Padding**: Extra pixels around detected animals
- **Censor mode**: black, blur, pixelate, or color_match
- **Overlay**: Watermark style
- **YOLO model**: Swap between model sizes at runtime

## Architecture

### Live Pipeline (per stream)

```
HLS Source -> [Decoder FFmpeg: NVDEC + scale_cuda]
                    | TCP (raw BGR24 1920x1080)
             [Python: YOLO FP16 @ 480px, every 2nd frame]
                    | TCP
             [Encoder FFmpeg: video + audio] -> HLS segments
                    |
             [Flask :5000] -> dashboard + HLS
```

Audio bypasses Python entirely — decoder FFmpeg sends PCM audio directly to encoder FFmpeg via TCP.

## Known Limitations

- **NVENC session limit**: Consumer NVIDIA GPUs allow max 2 simultaneous encode sessions. Only 2 streams can run concurrently without the NVENC session limit patch.
- **Resolution**: Fixed at 1920x1080.
- **Channel filter**: Only discovers channels with "hallmark" in the name (case-insensitive).
