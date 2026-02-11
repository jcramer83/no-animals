"""
Real-Time IPTV Animal Censoring Stream Server v3 — multi-stream architecture.

Discovers up to 3 Hallmark channels from M3U playlist, each with its own
independent pipeline (decoder → YOLO → encoder → HLS). Streams auto-start
on first client request and auto-stop after 30s of inactivity.

Pipeline per stream:
  Decoder FFmpeg ──TCP:port+1──▶ Python ──TCP:port+2──▶ Encoder FFmpeg ──▶ HLS
  Decoder FFmpeg ──TCP:port+0──────────────────────────▶ Encoder FFmpeg (audio)

Port allocation: stream 0 = 19876-19878, stream 1 = 19886-19888, stream 2 = 19896-19898

Usage:
    py -3 stream_animals_v3.py
    Open http://localhost:5000
"""

import os
import time
import json
import io
import base64
import socket
import threading
import shutil
import subprocess
import re
import urllib.request
from datetime import datetime
import numpy as np
import cv2
from pathlib import Path
from flask import Flask, Response, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FFMPEG = (
    shutil.which("ffmpeg")
    or os.environ.get("FFMPEG_PATH")
)
if not FFMPEG:
    raise RuntimeError("ffmpeg not found on PATH. Install FFmpeg or set the FFMPEG_PATH environment variable.")
FFPROBE = (
    shutil.which("ffprobe")
    or os.environ.get("FFPROBE_PATH")
    or str(Path(FFMPEG).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe"))
)

W, H   = 1920, 1080
FPS    = 30.0
PIX    = "bgr24"
FRAME_BYTES = W * H * 3   # 2,764,800 bytes per frame

ANIMAL_CLASS_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
ANIMAL_CLASS_NAMES = {
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}

MAX_STREAMS = 3
CLIENT_TIMEOUT = 30  # seconds with no requests before auto-stop

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0


def rounded_rect_mask(h, w, radius):
    """Create a float32 mask (h, w) with rounded corners, 1.0 inside, 0.0 outside."""
    mask = np.ones((h, w), dtype=np.float32)
    r = min(radius, h // 2, w // 2)
    if r <= 0:
        return mask
    # Create corner circles using coordinate grids
    yy, xx = np.ogrid[:r, :r]
    dist = (r - 1 - xx) ** 2 + (r - 1 - yy) ** 2
    corner = (dist <= r * r).astype(np.float32)
    # Top-left
    mask[:r, :r] = corner
    # Top-right
    mask[:r, w-r:] = corner[:, ::-1]
    # Bottom-left
    mask[h-r:, :r] = corner[::-1, :]
    # Bottom-right
    mask[h-r:, w-r:] = corner[::-1, ::-1]
    return mask


class BoxTracker:
    def __init__(self, persist=12, smooth=5):
        self.persist = persist
        self.smooth = smooth
        self.tracks = []
        self.next_id = 0

    def update(self, dets):
        used_t, used_d = set(), set()
        pairs = []
        for ti, t in enumerate(self.tracks):
            avg = self._avg(t["h"])
            for di, d in enumerate(dets):
                s = iou(avg, d)
                if s > 0.15:
                    pairs.append((s, ti, di))
        pairs.sort(reverse=True)
        for s, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            self.tracks[ti]["h"].append(dets[di])
            if len(self.tracks[ti]["h"]) > self.smooth:
                self.tracks[ti]["h"] = self.tracks[ti]["h"][-self.smooth:]
            self.tracks[ti]["gone"] = 0
            used_t.add(ti); used_d.add(di)
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append({"id": self.next_id, "h": [d], "gone": 0})
                self.next_id += 1
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t["gone"] += 1
        self.tracks = [t for t in self.tracks if t["gone"] <= self.persist]
        return [self._avg(t["h"]) for t in self.tracks]

    def _avg(self, h):
        return np.array(h).mean(axis=0).astype(int).tolist()


def slugify(name):
    """'HALLMARK MOVIES & MYSTERIES' -> 'hallmark-movies-mysteries'"""
    s = name.lower().strip()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    return s.strip('-')


def probe_stream(url):
    """Get source fps and whether audio exists."""
    cmd = [FFPROBE, "-v", "quiet", "-print_format", "json", "-show_streams", url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = json.loads(r.stdout)
        vs = next(s for s in info["streams"] if s["codec_type"] == "video")
        num, den = vs["r_frame_rate"].split("/")
        fps = int(num) / int(den)
        has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
        return {"fps": fps, "has_audio": has_audio}
    except Exception as e:
        print(f"[probe] failed: {e}", flush=True)
        return None


def tcp_server(port):
    """Create a reusable TCP server socket."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', port))
    srv.listen(1)
    srv.settimeout(30)
    return srv


def port_is_listening(port):
    """Non-consuming check: can we bind? If not, something is listening."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        try:
            s.bind(('127.0.0.1', port))
            s.close()
            return False
        except OSError:
            s.close()
            return True
    except Exception:
        return False


def wait_for_port(port, timeout=15):
    """Wait until a TCP port is in LISTEN state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_is_listening(port):
            return True
        time.sleep(0.2)
    return False


def recv_exact(conn, n, buf):
    """Read exactly n bytes into buf via TCP. Returns True on success."""
    view = memoryview(buf)
    pos = 0
    while pos < n:
        try:
            nb = conn.recv_into(view[pos:], n - pos)
        except (OSError, ConnectionError):
            return False
        if nb == 0:
            return False
        pos += nb
    return True


def drain_stderr(proc, log_path=None):
    """Daemon thread to drain a process's stderr (prevents pipe deadlock)."""
    def _run():
        try:
            if log_path:
                with open(log_path, "w") as f:
                    for line in proc.stderr:
                        f.write(line.decode("utf-8", errors="replace"))
                        f.flush()
            else:
                for _ in proc.stderr:
                    pass
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

SETTINGS_FILE = Path(__file__).parent / "noanimals_settings.json"

config_lock = threading.Lock()
_default_config = {
    "confidence": 0.4,
    "padding": 30,
    "model": "yolov8x.pt",
    "persist_frames": 2,
    "smooth_window": 5,
    "overlay": "text",
    "censor_mode": "pixelate",
}


def _load_config():
    """Load saved settings from disk, falling back to defaults."""
    cfg = dict(_default_config)
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            for k in _default_config:
                if k in saved:
                    cfg[k] = saved[k]
            print(f"[config] loaded settings from {SETTINGS_FILE}", flush=True)
    except Exception as e:
        print(f"[config] failed to load settings: {e}", flush=True)
    return cfg


def _save_config():
    """Save current config to disk (call while holding config_lock)."""
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"[config] failed to save settings: {e}", flush=True)


config = _load_config()

streams_lock = threading.Lock()
streams = {}  # slug -> StreamInstance

m3u_lock = threading.Lock()
m3u_state = {
    "m3u_url": "",
    "channels": [],       # [{"name": ..., "url": ..., "slug": ...}, ...]
    "last_fetched": None,
    "last_error": "",
    "fetching": False,
}


# ---------------------------------------------------------------------------
# StreamInstance
# ---------------------------------------------------------------------------

class StreamInstance:
    def __init__(self, slug, channel_name, stream_url, port_base):
        self.slug = slug
        self.channel_name = channel_name
        self.stream_url = stream_url
        self.a_port = port_base           # audio bypass
        self.v_in_port = port_base + 1    # decoder video -> Python
        self.v_out_port = port_base + 2   # Python video -> encoder
        self.out_dir = Path(f"./stream_output/{slug}")
        self.logo = ""

        self.stats_lock = threading.Lock()
        self.stats = {
            "status": "listening",
            "fps": 0.0,
            "detections_count": 0,
            "frames_processed": 0,
            "uptime": 0.0,
            "last_error": "",
            "active_clients": 0,
            "animal_counts": {},
        }

        self.stop_event = threading.Event()
        self.pipeline_procs = []
        self.client_ips = {}               # {ip: last_request_time}
        self.last_client_request = 0.0
        self._pipeline_lock = threading.Lock()

    def touch_client(self, ip):
        now = time.time()
        self.last_client_request = now
        self.client_ips[ip] = now

    @property
    def active_client_count(self):
        cutoff = time.time() - 15
        return sum(1 for t in self.client_ips.values() if t > cutoff)

    def start(self):
        with self._pipeline_lock:
            with self.stats_lock:
                st = self.stats["status"]
            if st in ("running", "probing"):
                return
            self.stop_event.clear()
            threading.Thread(target=self.run_pipeline, daemon=True).start()

    def stop(self):
        self.stop_event.set()
        for p in self.pipeline_procs:
            try: p.terminate()
            except Exception: pass

    def run_pipeline(self):
        """Main pipeline: decode -> process -> encode. Runs in a background thread."""
        tag = self.slug

        with self.stats_lock:
            self.stats.update({"status": "probing", "fps": 0, "detections_count": 0,
                               "frames_processed": 0, "uptime": 0, "last_error": ""})

        # --- Probe source ---
        probe = probe_stream(self.stream_url)
        if probe is None:
            with self.stats_lock:
                self.stats["last_error"] = "Failed to probe stream"
                self.stats["status"] = "listening"
            return
        has_audio = probe["has_audio"]
        print(f"[{tag}] probe: src_fps={probe['fps']:.1f} audio={has_audio} out={W}x{H}@{FPS}", flush=True)

        # --- Clean output dir ---
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir, ignore_errors=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.stop_event.clear()
        self.pipeline_procs = []

        # --- Create TCP server for decoder video (Python listens) ---
        v_in_srv = tcp_server(self.v_in_port)

        # --- Start encoder FFmpeg ---
        enc_cmd = [FFMPEG, "-y"]
        if has_audio:
            enc_cmd += [
                "-probesize", "32", "-analyzeduration", "0",
                "-f", "s16le", "-ar", "48000", "-ac", "2",
                "-thread_queue_size", "4096",
                "-i", f"tcp://127.0.0.1:{self.a_port}?listen=1",
            ]
            enc_cmd += [
                "-probesize", "32", "-analyzeduration", "0",
                "-f", "rawvideo", "-pix_fmt", PIX, "-s", f"{W}x{H}",
                "-r", f"{FPS:.2f}", "-thread_queue_size", "512",
                "-i", f"tcp://127.0.0.1:{self.v_out_port}?listen=1",
            ]
            enc_cmd += ["-map", "1:v", "-map", "0:a"]
        else:
            enc_cmd += [
                "-probesize", "32", "-analyzeduration", "0",
                "-f", "rawvideo", "-pix_fmt", PIX, "-s", f"{W}x{H}",
                "-r", f"{FPS:.2f}", "-thread_queue_size", "512",
                "-i", f"tcp://127.0.0.1:{self.v_out_port}?listen=1",
            ]
            enc_cmd += ["-map", "0:v"]
        enc_cmd += [
            "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p",
            "-preset", "p4", "-b:v", "4M", "-maxrate", "6M", "-bufsize", "8M",
            "-g", str(int(FPS * 2)),
        ]
        if has_audio:
            enc_cmd += ["-c:a", "aac", "-b:a", "128k"]
        start_num = int(time.time()) % 100000
        enc_cmd += [
            "-f", "hls", "-hls_time", "4", "-hls_list_size", "8",
            "-start_number", str(start_num),
            "-hls_flags", "delete_segments+omit_endlist",
            "-hls_segment_filename", str(self.out_dir / f"{self.slug}_%05d.ts"),
            str(self.out_dir / f"{self.slug}.m3u8"),
        ]
        print(f"[{tag}] encoder: {' '.join(enc_cmd)}", flush=True)
        enc_proc = subprocess.Popen(
            enc_cmd, stdin=subprocess.DEVNULL, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.ABOVE_NORMAL_PRIORITY_CLASS,
        )
        self.pipeline_procs.append(enc_proc)
        drain_stderr(enc_proc, str(self.out_dir / "encoder.log"))

        # --- Wait for encoder's audio TCP listener ---
        if has_audio:
            if wait_for_port(self.a_port):
                print(f"[{tag}] encoder audio TCP port {self.a_port} ready", flush=True)
            else:
                print(f"[{tag}] WARNING: audio port check timed out", flush=True)

        # --- Start decoder FFmpeg ---
        dec_cmd = [
            FFMPEG, "-y",
            "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-c:v", "h264_cuvid",
            "-i", self.stream_url,
            "-map", "0:v:0",
            "-vf", f"fps={FPS:.2f},scale_cuda={W}:{H},hwdownload,format=nv12",
            "-pix_fmt", PIX, "-f", "rawvideo",
            f"tcp://127.0.0.1:{self.v_in_port}",
        ]
        if has_audio:
            dec_cmd += [
                "-map", "0:a:0", "-f", "s16le", "-ar", "48000", "-ac", "2",
                f"tcp://127.0.0.1:{self.a_port}",
            ]
        print(f"[{tag}] decoder: {' '.join(dec_cmd)}", flush=True)
        dec_proc = subprocess.Popen(
            dec_cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.ABOVE_NORMAL_PRIORITY_CLASS,
        )
        self.pipeline_procs.append(dec_proc)
        drain_stderr(dec_proc, str(self.out_dir / "decoder.log"))

        # --- Accept decoder video connection ---
        try:
            v_in_conn, _ = v_in_srv.accept()
            v_in_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            v_in_conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, FRAME_BYTES * 4)
            print(f"[{tag}] decoder video TCP connected, rcvbuf={v_in_conn.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)}", flush=True)
        except socket.timeout:
            with self.stats_lock:
                self.stats["last_error"] = "Decoder video TCP accept timed out"
                self.stats["status"] = "listening"
            v_in_srv.close()
            return
        finally:
            v_in_srv.close()

        # --- Wait for encoder's video TCP listener ---
        print(f"[{tag}] waiting for encoder video TCP port {self.v_out_port}...", flush=True)
        if not wait_for_port(self.v_out_port, timeout=30):
            with self.stats_lock:
                self.stats["last_error"] = "Encoder video listen port timed out"
                self.stats["status"] = "listening"
            v_in_conn.close()
            return
        print(f"[{tag}] encoder video TCP port {self.v_out_port} ready", flush=True)

        # --- Connect Python to encoder's video port ---
        v_out_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        v_out_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            v_out_conn.connect(('127.0.0.1', self.v_out_port))
            v_out_conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, FRAME_BYTES * 4)
            print(f"[{tag}] encoder video TCP connected, sndbuf={v_out_conn.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)}", flush=True)
        except (ConnectionRefusedError, OSError) as e:
            with self.stats_lock:
                self.stats["last_error"] = f"Encoder video connect failed: {e}"
                self.stats["status"] = "listening"
            v_in_conn.close()
            return

        # --- Load YOLO model ---
        from ultralytics import YOLO

        with config_lock:
            model_name = config["model"]
            conf = config["confidence"]
            padding = config["padding"]
            persist = config["persist_frames"]
            smooth = config["smooth_window"]

        model = YOLO(model_name)
        model.to("cuda")
        tracker = BoxTracker(persist, smooth)
        cur_model = model_name

        frame_buf = bytearray(FRAME_BYTES)
        frame_count = 0
        det_count = 0
        animal_counts = {}
        fps_clock = time.perf_counter()
        start_time = time.time()
        DETECT_EVERY = 2
        cached_boxes = []

        with self.stats_lock:
            self.stats["status"] = "running"
            self.stats["last_error"] = ""

        print(f"[{tag}] main loop started", flush=True)

        # -------------------------------------------------------------------
        # MAIN LOOP: recv -> detect -> draw -> send
        # -------------------------------------------------------------------
        while not self.stop_event.is_set():
            if not recv_exact(v_in_conn, FRAME_BYTES, frame_buf):
                with self.stats_lock:
                    self.stats["last_error"] = "Decoder stream ended"
                    self.stats["status"] = "listening"
                break

            frame = np.frombuffer(frame_buf, dtype=np.uint8).reshape((H, W, 3)).copy()

            with config_lock:
                new_model = config["model"]
                conf = config["confidence"]
                padding = config["padding"]
                persist = config["persist_frames"]
                smooth = config["smooth_window"]
                overlay_mode = config["overlay"]
                censor_mode = config["censor_mode"]

            if new_model != cur_model:
                cur_model = new_model
                model = YOLO(cur_model)
                model.to("cuda")
                tracker = BoxTracker(persist, smooth)
            tracker.persist = persist
            tracker.smooth = smooth

            if frame_count % DETECT_EVERY == 0:
                results = model.predict(frame, conf=conf, verbose=False,
                                        classes=list(ANIMAL_CLASS_IDS),
                                        half=True, imgsz=480)
                raw_boxes = []
                raw_classes = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        raw_boxes.append([
                            max(0, x1 - padding), max(0, y1 - padding),
                            min(W, x2 + padding), min(H, y2 + padding),
                        ])
                        det_count += 1
                        cls_id = int(box.cls[0].item())
                        raw_classes.append(ANIMAL_CLASS_NAMES.get(cls_id, f"class_{cls_id}"))
                next_id_before = tracker.next_id
                cached_boxes = tracker.update(raw_boxes)
                # Count only newly confirmed tracks (unique animal instances)
                for t in tracker.tracks:
                    if t["id"] >= next_id_before:
                        # New track — match to detection by box coords
                        for i, rb in enumerate(raw_boxes):
                            if rb == t["h"][-1]:
                                animal_counts[raw_classes[i]] = animal_counts.get(raw_classes[i], 0) + 1
                                break

            for b in cached_boxes:
                y1c, y2c = max(0, b[1]), min(H, b[3])
                x1c, x2c = max(0, b[0]), min(W, b[2])
                rh, rw = y2c - y1c, x2c - x1c
                if rh <= 0 or rw <= 0:
                    continue
                roi = frame[y1c:y2c, x1c:x2c]
                radius = max(12, min(rh, rw) // 6)
                mask = rounded_rect_mask(rh, rw, radius)[:, :, None]
                if censor_mode == "blur":
                    # Crush detail: downscale to tiny, upscale, then blur smooth
                    tiny = cv2.resize(roi, (max(1, rw // 40), max(1, rh // 40)),
                                      interpolation=cv2.INTER_AREA)
                    blurred = cv2.resize(tiny, (rw, rh), interpolation=cv2.INTER_LINEAR)
                    ksize = max(51, (min(rh, rw) // 3) | 1)
                    blurred = cv2.GaussianBlur(blurred, (ksize, ksize), 0)
                    # Flood toward average color to kill any remaining shape
                    avg = blurred.mean(axis=(0, 1)).astype(np.float32)
                    blurred = (blurred.astype(np.float32) * 0.3 + avg * 0.7).astype(np.uint8)
                    frame[y1c:y2c, x1c:x2c] = (roi * (1 - mask) + blurred * mask).astype(np.uint8)
                elif censor_mode == "pixelate":
                    small = cv2.resize(roi, (max(1, min(10, rw)), max(1, min(10, rh))),
                                       interpolation=cv2.INTER_AREA)
                    avg = small.mean(axis=(0, 1)).astype(np.float32)
                    small = (small.astype(np.float32) * 0.6 + avg * 0.4).astype(np.uint8)
                    pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
                    frame[y1c:y2c, x1c:x2c] = (roi * (1 - mask) + pix * mask).astype(np.uint8)
                elif censor_mode == "color_match":
                    avg = roi.mean(axis=(0, 1)).astype(np.uint8)
                    filled = np.full_like(roi, avg)
                    frame[y1c:y2c, x1c:x2c] = (roi * (1 - mask) + filled * mask).astype(np.uint8)
                else:
                    frame[y1c:y2c, x1c:x2c] = (roi * (1 - mask)).astype(np.uint8)

            ov = OVERLAY_MAP.get(overlay_mode)
            if ov is not None:
                apply_overlay(frame, ov, 16, H - ov.shape[0] - 16)

            try:
                v_out_conn.sendall(frame.tobytes())
            except (BrokenPipeError, OSError, ConnectionError):
                with self.stats_lock:
                    self.stats["last_error"] = "Encoder connection lost"
                    self.stats["status"] = "listening"
                break

            frame_count += 1
            if frame_count % 30 == 0:
                now = time.perf_counter()
                cur_fps = round(30 / max(now - fps_clock, 0.001), 1)
                with self.stats_lock:
                    self.stats["fps"] = cur_fps
                    self.stats["detections_count"] = det_count
                    self.stats["frames_processed"] = frame_count
                    self.stats["uptime"] = round(time.time() - start_time, 1)
                    self.stats["active_clients"] = self.active_client_count
                    self.stats["animal_counts"] = dict(animal_counts)
                print(f"[{tag}] frames={frame_count} fps={cur_fps}", flush=True)
                fps_clock = now

        # --- Cleanup ---
        for c in [v_in_conn, v_out_conn]:
            try: c.close()
            except Exception: pass
        for p in self.pipeline_procs:
            try: p.terminate()
            except Exception: pass
        for p in self.pipeline_procs:
            try: p.wait(timeout=5)
            except Exception:
                try: p.kill()
                except Exception: pass
        self.pipeline_procs = []
        with self.stats_lock:
            self.stats.update({"status": "listening", "fps": 0.0,
                               "detections_count": 0, "frames_processed": 0,
                               "uptime": 0.0, "last_error": "", "animal_counts": {}})
        print(f"[{tag}] pipeline stopped", flush=True)


# ---------------------------------------------------------------------------
# M3U playlist fetching
# ---------------------------------------------------------------------------

def fetch_m3u():
    """Download M3U playlist and extract up to MAX_STREAMS Hallmark channel URLs."""
    with m3u_lock:
        url = m3u_state["m3u_url"]
        if not url:
            m3u_state["last_error"] = "No M3U URL configured"
            return
        m3u_state["fetching"] = True
        m3u_state["last_error"] = ""

    print(f"[m3u] fetching playlist from {url[:80]}...", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            channels = []
            pending_name = None
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if pending_name is not None:
                    slug = slugify(pending_name)
                    # Deduplicate by slug
                    if not any(c["slug"] == slug for c in channels):
                        channels.append({"name": pending_name, "url": line, "slug": slug, "logo": pending_logo})
                    pending_name = None
                    pending_logo = ""
                    if len(channels) >= MAX_STREAMS:
                        break
                elif line.startswith("#EXTINF") and "hallmark" in line.lower():
                    m = re.search(r'tvg-name="([^"]*)"', line, re.IGNORECASE)
                    pending_name = m.group(1) if m else "HALLMARK"
                    logo_m = re.search(r'tvg-logo="([^"]*)"', line, re.IGNORECASE)
                    pending_logo = logo_m.group(1) if logo_m else ""

        with m3u_lock:
            m3u_state["fetching"] = False
            if channels:
                m3u_state["channels"] = channels
                m3u_state["last_fetched"] = datetime.now().isoformat()
                m3u_state["last_error"] = ""
                for ch in channels:
                    print(f"[m3u] found: {ch['name']} [{ch['slug']}] -> {ch['url'][:80]}...", flush=True)
            else:
                m3u_state["last_error"] = "No Hallmark channels found in playlist"
                print("[m3u] no hallmark channels found", flush=True)

        # Create/update StreamInstance objects
        _sync_streams(channels)

    except Exception as e:
        with m3u_lock:
            m3u_state["fetching"] = False
            m3u_state["last_error"] = str(e)
        print(f"[m3u] fetch error: {e}", flush=True)


def _sync_streams(channels):
    """Create StreamInstance objects for discovered channels."""
    with streams_lock:
        for i, ch in enumerate(channels):
            slug = ch["slug"]
            port_base = 19876 + (i * 10)
            if slug not in streams:
                inst = StreamInstance(slug, ch["name"], ch["url"], port_base)
                inst.logo = ch.get("logo", "")
                streams[slug] = inst
                print(f"[streams] created instance '{slug}' ports {port_base}-{port_base+2}", flush=True)
            else:
                # Update URL/metadata in case it changed
                streams[slug].stream_url = ch["url"]
                streams[slug].channel_name = ch["name"]
                streams[slug].logo = ch.get("logo", "")


# ---------------------------------------------------------------------------
# Pre-rendered overlays (created once at import, reused every frame)
# ---------------------------------------------------------------------------

def _build_text_overlay():
    """Render 'NoAnimals' text overlay matching the dashboard header style."""
    try:
        font = ImageFont.truetype("segoeuib.ttf", 28)  # Segoe UI Bold
    except OSError:
        try:
            font = ImageFont.truetype("arialbd.ttf", 28)
        except OSError:
            font = ImageFont.load_default()
    # Measure each part separately: "No" in accent red, "Animals" in white
    tmp = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(tmp)
    no_bbox = draw.textbbox((0, 0), "No", font=font)
    no_w = no_bbox[2] - no_bbox[0]
    full_bbox = draw.textbbox((0, 0), "NoAnimals", font=font)
    full_w, full_h = full_bbox[2] - full_bbox[0], full_bbox[3] - full_bbox[1]
    # Create image with padding
    pad = 10
    w, h = full_w + pad * 2, full_h + pad * 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y_off = pad - full_bbox[1]
    # Dark shadow for readability on video
    draw.text((pad + 1, y_off + 1), "NoAnimals", font=font, fill=(0, 0, 0, 140))
    # "No" in accent red #e94560
    draw.text((pad, y_off), "No", font=font, fill=(233, 69, 96, 240))
    # "Animals" in white, offset by "No" width
    draw.text((pad + no_w, y_off), "Animals", font=font, fill=(255, 255, 255, 240))
    # Convert RGBA to BGRA for OpenCV
    arr = np.array(img)
    arr[:, :, :3] = arr[:, :, 2::-1]  # RGB -> BGR
    return arr


def _build_graphic_overlay():
    """Render dog emoji with prohibition sign as BGRA numpy array."""
    size = 96
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy, r = size // 2, size // 2, 42

    # Render dog emoji using Windows Segoe UI Emoji font
    try:
        font = ImageFont.truetype("seguiemj.ttf", 62)
        dog = "\U0001F415"
        bbox = draw.textbbox((0, 0), dog, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - tw) // 2 - bbox[0]
        y = (size - th) // 2 - bbox[1] + 2
        draw.text((x, y), dog, font=font, embedded_color=True)
    except (OSError, AttributeError):
        # Fallback: simple filled circle with "D" if emoji font unavailable
        draw.ellipse([20, 20, 76, 76], fill=(200, 180, 120, 200))
        try:
            fb = ImageFont.truetype("arial.ttf", 36)
        except OSError:
            fb = ImageFont.load_default()
        draw.text((35, 28), "D", font=fb, fill=(80, 60, 30, 230))

    # Prohibition sign
    ring_w = 5
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=(233, 69, 96, 230), width=ring_w)
    offset = int(r * 0.7071)
    draw.line([(cx - offset, cy - offset), (cx + offset, cy + offset)],
              fill=(233, 69, 96, 230), width=ring_w)

    # Convert RGBA to BGRA
    arr = np.array(img)
    arr[:, :, :3] = arr[:, :, 2::-1]
    return arr


def _build_petfree_overlay():
    """Render 'Pet Free TV' retro badge overlay as BGRA numpy array."""
    try:
        font = ImageFont.truetype("segoeuib.ttf", 22)
    except OSError:
        try:
            font = ImageFont.truetype("arialbd.ttf", 22)
        except OSError:
            font = ImageFont.load_default()
    tmp = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), "PET FREE TV", font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 14, 8
    w, h = tw + pad_x * 2, th + pad_y * 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Rounded dark background pill
    draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=10, fill=(15, 15, 26, 180),
                           outline=(233, 69, 96, 200), width=2)
    # White text with accent star
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), "PET FREE TV", font=font,
              fill=(255, 255, 255, 230))
    arr = np.array(img)
    arr[:, :, :3] = arr[:, :, 2::-1]
    return arr


def _build_paw_overlay():
    """Render paw emoji with prohibition sign as BGRA numpy array."""
    size = 96
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy, r = size // 2, size // 2, 42
    try:
        font = ImageFont.truetype("seguiemj.ttf", 58)
        paw = "\U0001F43E"
        bbox = draw.textbbox((0, 0), paw, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (size - tw) // 2 - bbox[0]
        y = (size - th) // 2 - bbox[1] + 2
        draw.text((x, y), paw, font=font, embedded_color=True)
    except (OSError, AttributeError):
        draw.ellipse([24, 24, 72, 72], fill=(180, 140, 100, 200))
    ring_w = 5
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 outline=(233, 69, 96, 230), width=ring_w)
    offset = int(r * 0.7071)
    draw.line([(cx - offset, cy - offset), (cx + offset, cy + offset)],
              fill=(233, 69, 96, 230), width=ring_w)
    arr = np.array(img)
    arr[:, :, :3] = arr[:, :, 2::-1]
    return arr


def _build_censored_overlay():
    """Render 'CENSORED' red stamp overlay as BGRA numpy array."""
    try:
        font = ImageFont.truetype("segoeuib.ttf", 30)
    except OSError:
        try:
            font = ImageFont.truetype("arialbd.ttf", 30)
        except OSError:
            font = ImageFont.load_default()
    tmp = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), "CENSORED", font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 12, 6
    w, h = tw + pad_x * 2, th + pad_y * 2
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Red border rectangle (stamp look)
    draw.rectangle([0, 0, w - 1, h - 1], outline=(233, 69, 96, 220), width=3)
    # Red text
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), "CENSORED", font=font,
              fill=(233, 69, 96, 220))
    arr = np.array(img)
    arr[:, :, :3] = arr[:, :, 2::-1]
    return arr


def _overlay_to_png_b64(overlay_bgra):
    """Convert a BGRA numpy overlay to a base64-encoded PNG data URI."""
    rgba = overlay_bgra.copy()
    rgba[:, :, :3] = rgba[:, :, 2::-1]  # BGRA -> RGBA
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


TEXT_OVERLAY = _build_text_overlay()
GRAPHIC_OVERLAY = _build_graphic_overlay()
PETFREE_OVERLAY = _build_petfree_overlay()
PAW_OVERLAY = _build_paw_overlay()
CENSORED_OVERLAY = _build_censored_overlay()
OVERLAY_MAP = {
    "text": TEXT_OVERLAY,
    "graphic": GRAPHIC_OVERLAY,
    "petfree": PETFREE_OVERLAY,
    "paw": PAW_OVERLAY,
    "censored": CENSORED_OVERLAY,
}
OVERLAY_PREVIEWS = {k: _overlay_to_png_b64(v) for k, v in OVERLAY_MAP.items()}


def apply_overlay(frame, overlay, x, y):
    """Blend BGRA overlay onto BGR frame at (x, y) using alpha channel."""
    oh, ow = overlay.shape[:2]
    roi = frame[y:y+oh, x:x+ow]
    alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
    frame[y:y+oh, x:x+ow] = (roi * (1 - alpha) + overlay[:, :, :3] * alpha).astype(np.uint8)


def m3u_refresh_loop():
    """Background thread: re-fetch M3U every 3 days."""
    while True:
        time.sleep(3 * 86400)
        fetch_m3u()


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------

def client_watchdog():
    """Background thread: stop streams with no client requests for CLIENT_TIMEOUT seconds."""
    while True:
        time.sleep(5)
        with streams_lock:
            stream_list = list(streams.values())
        for stream in stream_list:
            with stream.stats_lock:
                st = stream.stats["status"]
            if st not in ("running", "probing"):
                continue
            if stream.last_client_request > 0 and (time.time() - stream.last_client_request) > CLIENT_TIMEOUT:
                print(f"[watchdog] [{stream.slug}] no client requests for {CLIENT_TIMEOUT}s, stopping", flush=True)
                stream.stop()
                with stream.stats_lock:
                    stream.stats["status"] = "listening"
                stream.last_client_request = 0.0


# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------

app = Flask(__name__)

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NoAnimals - Stream Server</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#0f0f1a;color:#e0e0e0;min-height:100vh}
.topbar{background:linear-gradient(135deg,#16213e 0%,#1a1a2e 100%);border-bottom:1px solid #0f3460;padding:16px 24px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;backdrop-filter:blur(12px)}
.topbar h1{font-size:1.3em;font-weight:700;color:#fff;letter-spacing:-0.5px}
.topbar h1 span{color:#e94560}
.topbar-right{display:flex;align-items:center;gap:16px}
.topbar-playlist{display:flex;align-items:center;gap:8px;background:#0f0f1a;border:1px solid #0f3460;border-radius:8px;padding:6px 12px}
.topbar-playlist code{font-size:0.82em;color:#53d8fb;white-space:nowrap}
.topbar-playlist .btn-copy{padding:4px 10px;font-size:0.78em}
.container{max-width:960px;margin:0 auto;padding:20px}
.card{background:#16213e;border-radius:12px;padding:20px;margin-bottom:16px;border:1px solid rgba(15,52,96,0.6);box-shadow:0 2px 12px rgba(0,0,0,0.3);transition:border-color 0.3s}
.card:hover{border-color:rgba(83,216,251,0.2)}
.card h2{font-size:0.95em;margin-bottom:14px;color:#a0a0b0;text-transform:uppercase;letter-spacing:1px;font-weight:600}
.collapsible-toggle{cursor:pointer;display:flex;align-items:center;justify-content:space-between;user-select:none}
.collapsible-toggle::after{content:'\25BC';font-size:0.7em;color:#a0a0b0;transition:transform 0.2s}
.collapsible-toggle.collapsed::after{transform:rotate(-90deg)}
.collapsible-body{overflow:hidden;transition:max-height 0.3s ease,opacity 0.3s ease;max-height:500px;opacity:1}
.collapsible-body.hidden{max-height:0;opacity:0;margin:0;padding:0}
.row{display:flex;gap:10px;align-items:center;margin-bottom:10px}
.row label{min-width:120px;font-size:0.85em;color:#a0a0b0;font-weight:500}
.row input[type=text]{flex:1;padding:8px 12px;border-radius:8px;border:1px solid #0f3460;background:#0f0f1a;color:#e0e0e0;font-size:0.9em;transition:border-color 0.2s}
.row input[type=text]:focus{outline:none;border-color:#53d8fb}
.row input[type=range]{flex:1;accent-color:#e94560}
.row select{flex:1;padding:8px 12px;border-radius:8px;border:1px solid #0f3460;background:#0f0f1a;color:#e0e0e0;font-size:0.9em}
.row .val{min-width:44px;text-align:right;font-family:'SF Mono',Monaco,Consolas,monospace;font-size:0.85em;color:#53d8fb}
button{padding:8px 20px;border-radius:8px;border:none;cursor:pointer;font-size:0.85em;font-weight:600;transition:all 0.2s}
button:hover{transform:translateY(-1px);box-shadow:0 2px 8px rgba(0,0,0,0.3)}
button:active{transform:translateY(0)}
.btn-start{background:linear-gradient(135deg,#00b894,#00a884);color:#fff}
.btn-stop{background:linear-gradient(135deg,#e94560,#d63851);color:#fff}
.btn-copy{background:#0f3460;color:#53d8fb;padding:5px 12px;font-size:0.8em}
.btn-copy:hover{background:#1a4a7a}
.btn-save{background:linear-gradient(135deg,#0f3460,#1a4a7a);color:#53d8fb}
.buttons{display:flex;gap:10px;margin-top:8px}
.stream-cards{display:flex;flex-direction:column;gap:16px}
.stream-card{background:linear-gradient(135deg,#16213e 0%,#1a1a30 100%);border-radius:14px;padding:0;border:1px solid rgba(15,52,96,0.6);box-shadow:0 4px 16px rgba(0,0,0,0.3);overflow:hidden;transition:border-color 0.3s,box-shadow 0.3s}
.stream-card:hover{border-color:rgba(83,216,251,0.3);box-shadow:0 4px 20px rgba(83,216,251,0.08)}
.stream-card.is-running{border-color:rgba(0,184,148,0.4)}
.stream-card-header{display:flex;align-items:center;gap:14px;padding:16px 20px;border-bottom:1px solid rgba(15,52,96,0.4)}
.stream-logo{width:42px;height:42px;border-radius:8px;object-fit:contain;background:#0f0f1a;flex-shrink:0}
.stream-logo-placeholder{width:42px;height:42px;border-radius:8px;background:linear-gradient(135deg,#0f3460,#1a1a2e);display:flex;align-items:center;justify-content:center;font-size:1.1em;font-weight:700;color:#53d8fb;flex-shrink:0}
.stream-title{flex:1}
.stream-title .channel-name{font-size:1.05em;font-weight:700;color:#fff;letter-spacing:-0.3px}
.stream-title .stream-status{display:flex;align-items:center;gap:6px;margin-top:3px;font-size:0.8em;color:#a0a0b0}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%}
.status-listening{background:#53d8fb;box-shadow:0 0 6px rgba(83,216,251,0.5)}
.status-running{background:#00b894;box-shadow:0 0 6px rgba(0,184,148,0.5);animation:pulse 2s infinite}
.status-probing{background:#fdcb6e;box-shadow:0 0 6px rgba(253,203,110,0.5);animation:pulse 1s infinite}
.status-stopped{background:#666}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.5}}
.stream-card-body{padding:16px 20px}
.stats-row{display:flex;gap:8px;flex-wrap:wrap}
.mini-stat{flex:1;min-width:80px;background:#0f0f1a;border-radius:8px;padding:10px 8px;text-align:center;border:1px solid rgba(15,52,96,0.4)}
.mini-stat .label{font-size:0.7em;color:#a0a0b0;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:3px}
.mini-stat .value{font-size:1.15em;font-weight:700;font-family:'SF Mono',Monaco,Consolas,monospace;color:#fff}
.mini-stat .value.fps-val{color:#53d8fb}
.mini-stat .value.det-val{color:#e94560}
.mini-stat .value.viewer-val{color:#00b894}
.stream-card-footer{padding:10px 20px 14px;display:flex;align-items:center;gap:8px}
.stream-card-footer code{flex:1;background:#0f0f1a;padding:7px 12px;border-radius:8px;font-size:0.82em;color:#53d8fb;border:1px solid rgba(15,52,96,0.4);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:'SF Mono',Monaco,Consolas,monospace}
.error{color:#e17055;font-size:0.82em;padding:0 20px 8px;min-height:0}
.error:empty{display:none}
.config-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:600px){.config-grid{grid-template-columns:1fr}.topbar{flex-direction:column;gap:10px}.topbar-right{width:100%}}
.section-label{font-size:0.75em;color:#a0a0b0;text-transform:uppercase;letter-spacing:1.5px;font-weight:600;margin-bottom:12px;margin-top:8px}
</style>
</head>
<body>
<div class="topbar">
<h1><span>No</span>Animals</h1>
<div class="topbar-right">
<div class="topbar-playlist">
<code id="playlistUrl"></code>
<button class="btn-copy" onclick="copyText(document.getElementById('playlistUrl').textContent,this)">Copy</button>
</div>
</div>
</div>
<div class="container">

<div class="section-label">Channels</div>
<div class="stream-cards" id="streamCards">
<div class="card" style="text-align:center;color:#a0a0b0;padding:40px">
No channels discovered yet. Configure M3U source below and click Fetch Now.
</div>
</div>

<div class="section-label" style="margin-top:24px">Settings</div>
<div class="config-grid">
<div class="card">
<h2 class="collapsible-toggle" onclick="toggleCollapse(this)">Channel Source</h2>
<div class="collapsible-body">
<div class="row"><label>M3U URL</label>
<input type="text" id="m3uUrl" placeholder="https://example.com/playlist.m3u"></div>
<div class="buttons" style="justify-content:space-between;align-items:center">
<div style="display:flex;gap:8px">
<button class="btn-save" onclick="saveM3u()" id="m3uSaveBtn">Save</button>
<button class="btn-start" onclick="fetchM3u()" id="m3uFetchBtn">Fetch Now</button>
</div>
<span id="m3uLastFetched" style="color:#a0a0b0;font-size:0.78em">Never</span>
</div>
<div class="error" id="m3uError" style="padding:0;margin-top:6px"></div>
</div></div>

<div class="card">
<h2 class="collapsible-toggle" onclick="toggleCollapse(this)">Detection</h2>
<div class="collapsible-body">
<div class="row"><label>Confidence</label><input type="range" id="confidence" min="0.05" max="0.9" step="0.05" value="0.4"><span class="val" id="confidenceVal">0.40</span></div>
<div class="row"><label>Padding</label><input type="range" id="padding" min="0" max="100" step="5" value="30"><span class="val" id="paddingVal">30</span></div>
<div class="row"><label>Persist</label><input type="range" id="persist" min="1" max="30" step="1" value="2"><span class="val" id="persistVal">2</span></div>
<div class="row"><label>Smooth</label><input type="range" id="smooth" min="1" max="15" step="1" value="5"><span class="val" id="smoothVal">5</span></div>
<div class="row"><label>Model</label><select id="model">
<option value="yolov8n.pt">yolov8n (fast)</option>
<option value="yolov8s.pt">yolov8s (balanced)</option>
<option value="yolov8m.pt">yolov8m</option>
<option value="yolov8l.pt">yolov8l</option>
<option value="yolov8x.pt" selected>yolov8x (accurate)</option>
</select></div>
<div class="row"><label>Censor Style</label><select id="censorMode">
<option value="black">Black box</option>
<option value="blur">Blur</option>
<option value="pixelate" selected>Pixelate</option>
<option value="color_match">Color match</option>
</select></div>
</div></div>

<div class="card">
<h2 class="collapsible-toggle" onclick="toggleCollapse(this)">Overlay</h2>
<div class="collapsible-body">
<div class="row"><label>Watermark</label><select id="overlay">
<option value="none">Off</option>
<option value="text" selected>NoAnimals</option>
<option value="graphic">No dogs</option>
<option value="petfree">Pet Free TV</option>
<option value="paw">No paws</option>
<option value="censored">CENSORED</option>
</select></div>
<div id="overlayPreview" style="margin-top:10px;background:#1a1a2e;border:1px solid #0f3460;border-radius:8px;padding:16px;display:flex;align-items:center;justify-content:center;min-height:60px">
<span style="color:#555;font-size:0.82em" id="overlayPreviewOff">No overlay active</span>
<img id="overlayPreviewImg" style="display:none;max-height:80px;image-rendering:pixelated" alt="Overlay preview">
</div>
</div></div>
</div>

</div>
<script>
let dt=null;
function copyText(text,btn){if(navigator.clipboard&&window.isSecureContext){navigator.clipboard.writeText(text)}else{const ta=document.createElement('textarea');ta.value=text;ta.style.cssText='position:fixed;left:-9999px';document.body.appendChild(ta);ta.select();document.execCommand('copy');document.body.removeChild(ta)}if(btn){const orig=btn.textContent;btn.textContent='Copied!';setTimeout(()=>btn.textContent=orig,1500)}}
function post(u,b){return fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})}
function sendSettings(){post('/api/settings',{confidence:parseFloat(document.getElementById('confidence').value),padding:parseInt(document.getElementById('padding').value),persist_frames:parseInt(document.getElementById('persist').value),smooth_window:parseInt(document.getElementById('smooth').value),model:document.getElementById('model').value,overlay:document.getElementById('overlay').value,censor_mode:document.getElementById('censorMode').value})}
function ds(){clearTimeout(dt);dt=setTimeout(sendSettings,500)}
['confidence','padding','persist','smooth'].forEach(id=>{const e=document.getElementById(id),v=document.getElementById(id+'Val');e.addEventListener('input',()=>{v.textContent=id==='confidence'?parseFloat(e.value).toFixed(2):e.value;ds()})});
document.getElementById('model').addEventListener('change',sendSettings);
document.getElementById('censorMode').addEventListener('change',sendSettings);
document.getElementById('overlay').addEventListener('change',function(){sendSettings();updateOverlayPreview()});
let overlayPreviews={};
fetch('/api/overlay/preview').then(r=>r.json()).then(d=>{overlayPreviews=d;updateOverlayPreview()});
function updateOverlayPreview(){const v=document.getElementById('overlay').value,img=document.getElementById('overlayPreviewImg'),off=document.getElementById('overlayPreviewOff');if(v!=='none'&&overlayPreviews[v]){img.src=overlayPreviews[v];img.style.display='block';off.style.display='none'}else{img.style.display='none';off.style.display='block'}}
function fmt(s){s=Math.floor(s);if(s<60)return s+'s';const m=Math.floor(s/60),sec=s%60;if(m<60)return m+'m '+sec+'s';return Math.floor(m/60)+'h '+(m%60)+'m'}
function toggleCollapse(el){el.classList.toggle('collapsed');el.nextElementSibling.classList.toggle('hidden')}

function renderStreamCards(stats){
    const container=document.getElementById('streamCards');
    const slugs=Object.keys(stats);
    if(slugs.length===0){
        container.innerHTML='<div class="card" style="text-align:center;color:#a0a0b0;padding:40px">No channels discovered yet. Configure M3U source below and click Fetch Now.</div>';
        return;
    }
    slugs.forEach(slug=>{
        const d=stats[slug];
        let card=document.getElementById('card-'+slug);
        if(!card){
            card=document.createElement('div');
            card.className='stream-card';
            card.id='card-'+slug;
            const initial=(d.channel_name||slug).charAt(0).toUpperCase();
            card.innerHTML=`
<div class="stream-card-header">
<div class="stream-logo-placeholder" id="logo-${slug}">${initial}</div>
<div class="stream-title">
<div class="channel-name" id="chname-${slug}"></div>
<div class="stream-status"><span class="status-dot" id="dot-${slug}"></span><span id="st-${slug}"></span></div>
</div>
<button class="btn-stop" id="stopbtn-${slug}" style="padding:5px 12px;font-size:0.75em;display:none" onclick="post('/api/stop/${slug}',{})">Stop</button>
</div>
<div class="stream-card-body">
<div class="stats-row">
<div class="mini-stat"><div class="label">FPS</div><div class="value fps-val" id="fps-${slug}">0.0</div></div>
<div class="mini-stat"><div class="label">Detections</div><div class="value det-val" id="det-${slug}">0</div></div>
<div class="mini-stat"><div class="label">Frames</div><div class="value" id="fr-${slug}">0</div></div>
<div class="mini-stat"><div class="label">Uptime</div><div class="value" id="up-${slug}">0s</div></div>
<div class="mini-stat"><div class="label">Viewers</div><div class="value viewer-val" id="vc-${slug}">0</div></div>
<div class="mini-stat"><div class="label">Resolution</div><div class="value" id="res-${slug}" style="font-size:0.85em">-</div></div>
</div>
<div id="animals-${slug}" style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap"></div>
</div>
<div class="error" id="err-${slug}"></div>
<div class="stream-card-footer">
<code id="url-${slug}"></code>
<button class="btn-copy" onclick="copyText(document.getElementById('url-${slug}').textContent,this)">Copy</button>
</div>`;
            container.appendChild(card);
            if(d.logo){const el=document.getElementById('logo-'+slug);const img=document.createElement('img');img.className='stream-logo';img.src=d.logo;img.onerror=function(){this.replaceWith(el)};el.parentNode.replaceChild(img,el)}
        }
        card.className='stream-card'+(d.status==='running'?' is-running':'');
        const stopBtn=document.getElementById('stopbtn-'+slug);if(stopBtn)stopBtn.style.display=(d.status==='running'||d.status==='probing')?'inline-block':'none';
        document.getElementById('chname-'+slug).textContent=d.channel_name||slug;
        document.getElementById('dot-'+slug).className='status-dot status-'+d.status;
        document.getElementById('st-'+slug).textContent=d.status;
        document.getElementById('fps-'+slug).textContent=d.fps.toFixed(1);
        document.getElementById('det-'+slug).textContent=d.detections_count;
        document.getElementById('fr-'+slug).textContent=d.frames_processed;
        document.getElementById('up-'+slug).textContent=fmt(d.uptime);
        document.getElementById('vc-'+slug).textContent=d.active_clients;
        const resEl=document.getElementById('res-'+slug);if(resEl)resEl.textContent=d.status==='running'?_resolution:'-';
        const ac=document.getElementById('animals-'+slug);
        if(ac&&d.animal_counts){const entries=Object.entries(d.animal_counts).sort((a,b)=>b[1]-a[1]);ac.innerHTML=entries.map(([name,count])=>`<span style="background:#0f0f1a;border:1px solid #0f3460;border-radius:6px;padding:3px 8px;font-size:0.75em;color:#e0e0e0"><span style="color:#e94560">${count}</span> ${name}</span>`).join('')}
        document.getElementById('err-'+slug).textContent=d.last_error||'';
        document.getElementById('url-'+slug).textContent='http://'+window.location.host+'/stream/'+slug+'/'+slug+'.m3u8';
    });
    Array.from(container.children).forEach(el=>{
        if(!el.id || !el.id.startsWith('card-') || !stats[el.id.substring(5)]){
            el.remove();
        }
    });
}

function syncSelect(id,val){const el=document.getElementById(id);if(el&&val!=null&&el.value!==String(val))el.value=val}
function syncRange(id,val){const el=document.getElementById(id);if(el&&val!=null&&String(el.value)!==String(val)){el.value=val;const v=document.getElementById(id+'Val');if(v)v.textContent=id==='confidence'?parseFloat(val).toFixed(2):val}}
let _resolution='-';
function poll(){fetch('/api/stats').then(r=>r.json()).then(d=>{if(d._config){syncSelect('overlay',d._config.overlay);syncSelect('censorMode',d._config.censor_mode);syncSelect('model',d._config.model);syncRange('confidence',d._config.confidence);syncRange('padding',d._config.padding);syncRange('persist',d._config.persist_frames);syncRange('smooth',d._config.smooth_window);if(d._config.resolution)_resolution=d._config.resolution;updateOverlayPreview();delete d._config}renderStreamCards(d)}).catch(()=>{})}
setInterval(poll,2000);poll();

function fmtDate(iso){if(!iso)return'Never';const d=new Date(iso);return d.toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'})+' '+d.toLocaleTimeString('en-US',{hour:'numeric',minute:'2-digit'})}
function updateM3u(d){document.getElementById('m3uLastFetched').textContent=fmtDate(d.last_fetched);document.getElementById('m3uError').textContent=d.last_error||'';document.getElementById('m3uFetchBtn').textContent=d.fetching?'Fetching\u2026':'Fetch Now';document.getElementById('m3uFetchBtn').disabled=d.fetching}
function pollM3u(){fetch('/api/m3u').then(r=>r.json()).then(updateM3u).catch(()=>{})}
function fetchM3u(){document.getElementById('m3uFetchBtn').textContent='Fetching\u2026';document.getElementById('m3uFetchBtn').disabled=true;post('/api/m3u/fetch',{}).then(()=>setTimeout(pollM3u,2000)).catch(()=>{document.getElementById('m3uFetchBtn').textContent='Fetch Now';document.getElementById('m3uFetchBtn').disabled=false})}
function saveM3u(){const u=document.getElementById('m3uUrl').value.trim();if(!u)return;post('/api/m3u',{m3u_url:u,fetch:true}).then(()=>{document.getElementById('m3uSaveBtn').textContent='Saved!';setTimeout(()=>{document.getElementById('m3uSaveBtn').textContent='Save';pollM3u()},2000)}).catch(()=>{})}
document.getElementById('playlistUrl').textContent='http://'+window.location.host+'/noanimals.m3u';
fetch('/api/m3u').then(r=>r.json()).then(d=>{if(d.m3u_url)document.getElementById('m3uUrl').value=d.m3u_url;updateM3u(d)}).catch(()=>{});
setInterval(pollM3u,5000);
</script>
</body></html>"""


@app.route("/")
def index():
    return Response(DASHBOARD_HTML, content_type="text/html")


@app.route("/noanimals.m3u")
def noanimals_m3u():
    """Serve an M3U playlist of all discovered censored streams."""
    base = request.host_url.rstrip("/")
    lines = ["#EXTM3U"]
    with streams_lock:
        for slug, stream in streams.items():
            logo_attr = f' tvg-logo="{stream.logo}"' if getattr(stream, 'logo', '') else ''
            lines.append(f'#EXTINF:-1 tvg-name="{stream.channel_name}"{logo_attr},{stream.channel_name}')
            lines.append(f"{base}/stream/{slug}/{slug}.m3u8")
    return Response("\n".join(lines) + "\n", content_type="audio/x-mpegurl",
                    headers={"Content-Disposition": "inline; filename=noanimals.m3u"})


@app.route("/stream/<slug>/<path:filename>")
def stream_file(slug, filename):
    with streams_lock:
        stream = streams.get(slug)
    if not stream:
        return "Unknown stream", 404

    stream.touch_client(request.remote_addr)

    if filename.endswith(".m3u8"):
        fpath = stream.out_dir / filename
        with stream.stats_lock:
            st = stream.stats["status"]
        # Auto-start on first client request
        if st == "listening":
            print(f"[auto-start] [{slug}] client requested {filename}, starting pipeline", flush=True)
            stream.start()
            return Response("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:4\n",
                            content_type="application/vnd.apple.mpegurl")
        if st in ("probing", "running") and not fpath.exists():
            return Response("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:4\n",
                            content_type="application/vnd.apple.mpegurl")

    fpath = stream.out_dir / filename
    if not fpath.exists():
        return "Not found", 404
    ct = "application/vnd.apple.mpegurl" if filename.endswith(".m3u8") else "video/mp2t"
    return send_from_directory(str(stream.out_dir.resolve()), filename, mimetype=ct)


@app.route("/api/start/<slug>", methods=["POST"])
def api_start(slug):
    with streams_lock:
        stream = streams.get(slug)
    if not stream:
        return jsonify({"error": f"Unknown stream '{slug}'"}), 404
    stream.stop()
    time.sleep(0.5)
    stream.stop_event.clear()
    stream.start()
    return jsonify({"ok": True})


@app.route("/api/stop/<slug>", methods=["POST"])
def api_stop(slug):
    with streams_lock:
        stream = streams.get(slug)
    if not stream:
        return jsonify({"error": f"Unknown stream '{slug}'"}), 404
    stream.stop()
    with stream.stats_lock:
        stream.stats["status"] = "listening"
    return jsonify({"ok": True})


@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json(force=True)
    with config_lock:
        for k in ("confidence", "padding", "persist_frames", "smooth_window"):
            if k in data:
                config[k] = float(data[k]) if k == "confidence" else int(data[k])
        if "model" in data:
            config["model"] = str(data["model"])
        if "overlay" in data and data["overlay"] in ("none", "text", "graphic", "petfree", "paw", "censored"):
            config["overlay"] = data["overlay"]
        if "censor_mode" in data and data["censor_mode"] in ("black", "blur", "pixelate", "color_match"):
            config["censor_mode"] = data["censor_mode"]
        _save_config()
    return jsonify({"ok": True})


@app.route("/api/overlay/preview")
def api_overlay_preview():
    return jsonify(OVERLAY_PREVIEWS)


@app.route("/api/stats")
def api_stats():
    with streams_lock:
        stream_list = list(streams.items())
    result = {}
    for slug, stream in stream_list:
        with stream.stats_lock:
            s = dict(stream.stats)
        s["channel_name"] = stream.channel_name
        s["logo"] = stream.logo
        s["active_clients"] = stream.active_client_count
        result[slug] = s
    with config_lock:
        result["_config"] = dict(config)
    result["_config"]["resolution"] = f"{W}x{H}"
    return jsonify(result)


@app.route("/api/m3u")
def api_m3u_get():
    with m3u_lock:
        return jsonify(dict(m3u_state))


@app.route("/api/m3u", methods=["POST"])
def api_m3u_post():
    data = request.get_json(force=True)
    with m3u_lock:
        if "m3u_url" in data:
            m3u_state["m3u_url"] = data["m3u_url"].strip()
    if data.get("fetch", False):
        threading.Thread(target=fetch_m3u, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/m3u/fetch", methods=["POST"])
def api_m3u_fetch():
    with m3u_lock:
        if m3u_state["fetching"]:
            return jsonify({"ok": False, "error": "Already fetching"}), 409
    threading.Thread(target=fetch_m3u, daemon=True).start()
    return jsonify({"ok": True})


if __name__ == "__main__":
    # Clean output directory
    out_root = Path("./stream_output")
    if out_root.exists():
        shutil.rmtree(out_root, ignore_errors=True)

    print("Animal Censor Stream Server v3 — Multi-Stream")

    # Fetch M3U synchronously so streams exist before Flask starts
    fetch_m3u()

    with streams_lock:
        if streams:
            print(f"Discovered {len(streams)} channel(s):")
            for slug, inst in streams.items():
                print(f"  [{slug}] {inst.channel_name} -> http://localhost:5000/stream/{slug}/{slug}.m3u8")
        else:
            print("No channels discovered yet. Configure M3U source in dashboard.")

    print(f"Dashboard: http://localhost:5000")
    threading.Thread(target=m3u_refresh_loop, daemon=True).start()
    threading.Thread(target=client_watchdog, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
