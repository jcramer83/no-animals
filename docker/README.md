# NoAnimals — Docker (Intel iGPU)

Real-time video animal censoring for IPTV streams, running on Linux with Intel iGPU hardware acceleration.

Uses **VAAPI** for FFmpeg encode/decode and **OpenVINO** for YOLO inference (CPU). Falls back to software encoding if no iGPU is available.

## Quick Start

```bash
docker run -d \
  --name noanimals \
  -p 8080:8080 \
  --device /dev/dri:/dev/dri \
  --group-add video \
  -v /path/to/data:/app/data \
  -v /path/to/models:/app/models \
  ghcr.io/jcramer83/no-animals:latest
```

Dashboard: `http://localhost:8080`

## Unraid Setup

1. **Docker tab → Add Container**

| Field | Value |
|-------|-------|
| Name | `noanimals` |
| Repository | `ghcr.io/jcramer83/no-animals:latest` |
| Network Type | `bridge` |
| Extra Parameters | `--group-add video` |

2. **Add the following mappings** (click "Add another Path, Port, Variable, Label or Device" for each):

| Type | Container | Host |
|------|-----------|------|
| Port | `8080` | `8080` |
| Device | `/dev/dri` | — |
| Path | `/app/data` | `/mnt/user/appdata/noanimals/data` |
| Path | `/app/models` | `/mnt/user/appdata/noanimals/models` |

3. **Apply** and start the container.

4. Open `http://<UNRAID-IP>:8080`, set your M3U URL, and start a stream.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `M3U_URL` | IPTV M3U playlist URL (can also be set in dashboard) | — |
| `PYTHONUNBUFFERED` | Set to `1` for real-time log output | `1` |

## Bundled Model

The image includes **YOLOv8s** (OpenVINO format) — a good balance of speed and accuracy for CPU inference. Runs at 30+ fps on an Intel 13900K.

Other models can be exported inside the container:

```bash
docker exec noanimals python export_model.py --model yolov8n.pt   # fastest
docker exec noanimals python export_model.py --model yolov8m.pt   # more accurate
docker exec noanimals python export_model.py --model yolov8x.pt   # most accurate (may be too slow for real-time)
```

Then select the model from the dashboard dropdown.

## IPTV App Setup (Xtream Codes)

Connect any IPTV app (TiviMate, XCIPTV, Smarters, etc.) using Xtream Codes login:

| Field | Value |
|-------|-------|
| Server | `http://<IP>:8080` |
| Username | anything |
| Password | anything |

## Architecture

```
HLS Source → [Decoder FFmpeg: VAAPI decode + scale]
                  ↓ TCP (raw BGR24 1920x1080)
           [Python: YOLOv8s OpenVINO @ 480px, every 2nd frame]
                  ↓ TCP
           [Encoder FFmpeg: VAAPI h264_vaapi encode] → HLS segments
                  ↓
           [Flask :8080] → Dashboard + /stream/<slug>/*.m3u8
```

- **VAAPI** hardware encode/decode via Intel iGPU (`/dev/dri/renderD128`)
- **OpenVINO** YOLO inference on CPU
- Falls back to `libx264` software encode if no iGPU detected

## Requirements

- Docker
- Intel CPU with integrated graphics (6th gen+) for VAAPI acceleration
- `/dev/dri` device passthrough
- `video` group access

## Differences from Windows Version

| | Windows (root dir) | Docker (`docker/`) |
|---|---|---|
| GPU inference | CUDA | OpenVINO (CPU) |
| Decode | NVDEC (`h264_cuvid`) | VAAPI (`hwaccel vaapi`) |
| Encode | NVENC (`h264_nvenc`) | VAAPI (`h264_vaapi`) / `libx264` fallback |
| Default model | `yolov8x.pt` | `yolov8s_openvino_model` |
| Fonts | Segoe UI / Arial | Liberation Sans / DejaVu Sans |
| Tray launcher | Yes (`noanimals_tray.pyw`) | No |
