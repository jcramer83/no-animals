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
    Open http://localhost:8080
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
import urllib.error
from datetime import datetime
import numpy as np
import cv2
from pathlib import Path
from flask import Flask, Response, request, jsonify, send_from_directory, redirect
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINGET_FFMPEG = str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-8.0.1-full_build/bin/ffmpeg.exe")
FFMPEG = (
    shutil.which("ffmpeg")
    or os.environ.get("FFMPEG_PATH")
    or (_WINGET_FFMPEG if Path(_WINGET_FFMPEG).exists() else None)
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

MAX_STREAMS = 10
MAX_VOD_PIPELINES = 3       # GPU NVENC session limit is the real constraint
VOD_PORT_BASE = 19976       # well above live range 19876-19898
CLIENT_TIMEOUT = 30      # seconds with no requests before auto-stop (live)
VOD_CLIENT_TIMEOUT = 300  # 5 minutes for VOD (players buffer ahead and go idle)

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
    """Get source fps, audio presence, video dimensions, and duration."""
    cmd = [FFPROBE, "-v", "quiet", "-probesize", "5000000", "-analyzeduration", "5000000",
           "-print_format", "json", "-show_streams", "-show_format", url]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        info = json.loads(r.stdout)
        vs = next(s for s in info["streams"] if s["codec_type"] == "video")
        num, den = vs["r_frame_rate"].split("/")
        fps = int(num) / int(den)
        has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
        src_w = int(vs.get("width", 0))
        src_h = int(vs.get("height", 0))
        duration = float(info.get("format", {}).get("duration", 0))
        return {"fps": fps, "has_audio": has_audio, "width": src_w, "height": src_h, "duration": duration}
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
    "channel_filters": ["hallmark"],
    "excluded_channels": [],
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


def _load_m3u_state():
    """Load saved M3U state (url, channels, last_fetched) from settings file."""
    state = {"m3u_url": "", "channels": [], "last_fetched": None, "last_error": "", "fetching": False}
    try:
        if SETTINGS_FILE.exists():
            with open(SETTINGS_FILE, "r") as f:
                saved = json.load(f)
            if "m3u_url" in saved:
                state["m3u_url"] = saved["m3u_url"]
            if "m3u_channels" in saved:
                state["channels"] = saved["m3u_channels"]
            if "m3u_last_fetched" in saved:
                state["last_fetched"] = saved["m3u_last_fetched"]
            if state["m3u_url"]:
                print(f"[config] loaded M3U URL and {len(state['channels'])} channel(s) from settings", flush=True)
    except Exception as e:
        print(f"[config] failed to load M3U state: {e}", flush=True)
    return state


def _save_config():
    """Save current config + M3U state to disk (call while holding config_lock)."""
    try:
        data = dict(config)
        with m3u_lock:
            data["m3u_url"] = m3u_state["m3u_url"]
            data["m3u_channels"] = m3u_state["channels"]
            data["m3u_last_fetched"] = m3u_state["last_fetched"]
        with open(SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[config] failed to save settings: {e}", flush=True)


config = _load_config()

streams_lock = threading.Lock()
streams = {}  # slug -> StreamInstance

m3u_lock = threading.Lock()
m3u_state = _load_m3u_state()

vod_lock = threading.Lock()
vod_catalog = {}   # xui_id (str) -> {xui_id, name, url, logo, group}
vod_active = {}    # xui_id (str) -> StreamInstance (only running pipelines)


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
        self.is_vod = False
        self.completed = False
        self.xui_id = None

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
        self.encoder_proc = None
        self.pipeline_ready = threading.Event()
        self.stdout_claimed = False
        self.total_duration = 0.0   # movie duration in seconds (from probe)
        self.seek_offset = 0.0      # decoder -ss seek position
        self.hls_start_number = 0   # encoder segment start number
        self.keep_segments = False   # skip rmtree on restart (preserve old segments)
        self._seek_lock = threading.Lock()  # prevent concurrent seek_restart calls
        self.pad_y = 0              # letterbox bottom padding (for overlay placement)
        self._cached_probe = None   # cached probe_stream() result for seek restarts
        self._discontinuities = []  # segment numbers where #EXT-X-DISCONTINUITY is needed
        self.last_frame_jpeg = None # latest processed frame as JPEG bytes (for preview)
        self._pipeline_gen = 0     # incremented each pipeline start, used to cancel stale cleanups

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

    def start_stdout(self):
        """Start the stdout MPEG-TS pipeline in a background thread."""
        with self._pipeline_lock:
            with self.stats_lock:
                st = self.stats["status"]
            if st in ("running", "probing"):
                return
            self.stop_event.clear()
            self.pipeline_ready.clear()
            self.encoder_proc = None
            self.stdout_claimed = False
            threading.Thread(target=self.run_pipeline_stdout, daemon=True).start()

    def seek_restart(self, target_time):
        """Restart pipeline seeking to target_time seconds. Thread-safe via _seek_lock."""
        if not self._seek_lock.acquire(blocking=False):
            # Another seek is already in progress — skip this one
            print(f"[{self.slug}] seek to {target_time:.0f}s skipped — another seek in progress", flush=True)
            return
        try:
            HLS_TIME = 4
            seg_num = int(target_time / HLS_TIME)
            self.seek_offset = target_time
            self.hls_start_number = seg_num
            self.keep_segments = True   # preserve previously processed segments
            if seg_num not in self._discontinuities:
                self._discontinuities.append(seg_num)
            self.stop()
            # Force-kill pipeline procs for faster restart
            for p in self.pipeline_procs:
                try: p.kill()
                except Exception: pass
            # Wait for pipeline to actually stop (shorter timeout)
            for _ in range(40):  # 4s max
                with self.stats_lock:
                    st = self.stats["status"]
                if st not in ("running", "probing"):
                    break
                time.sleep(0.1)
            self.completed = False
            with self.stats_lock:
                self.stats["status"] = "listening"
            self.stop_event.clear()
            self.start()
        finally:
            self._seek_lock.release()

    def run_pipeline(self):
        """Main pipeline: decode -> process -> encode. Runs in a background thread."""
        tag = self.slug
        self._pipeline_gen += 1

        with self.stats_lock:
            self.stats.update({"status": "probing", "fps": 0, "detections_count": 0,
                               "frames_processed": 0, "uptime": 0, "last_error": ""})

        # --- Probe source (use cache on seek restarts to save ~10s) ---
        if self._cached_probe and self.seek_offset > 0:
            probe = self._cached_probe
            print(f"[{tag}] using cached probe (seek restart)", flush=True)
        else:
            probe = probe_stream(self.stream_url)
            if probe is None and self.is_vod:
                time.sleep(3)
                probe = probe_stream(self.stream_url)
            if probe is None:
                with self.stats_lock:
                    self.stats["last_error"] = "Failed to probe stream"
                    self.stats["status"] = "listening"
                return
            self._cached_probe = probe
        has_audio = probe["has_audio"]
        src_w, src_h = probe.get("width", 0), probe.get("height", 0)
        if probe.get("duration", 0) > 0:
            self.total_duration = probe["duration"]
        print(f"[{tag}] probe: src={src_w}x{src_h} fps={probe['fps']:.1f} audio={has_audio} dur={self.total_duration:.0f}s out={W}x{H}@{FPS}", flush=True)

        # --- Clean output dir (skip if keeping segments for seek) ---
        if not self.keep_segments:
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
        # For VOD seek: offset encoder timestamps to match playlist position
        if self.seek_offset > 0:
            enc_cmd += ["-output_ts_offset", f"{self.seek_offset:.3f}"]
        if self.is_vod:
            hls_list_size = "0"
            hls_flags = "independent_segments"
            hls_playlist_type = None      # no type — avoids EVENT confusion in IPTV players
            start_num = self.hls_start_number
        else:
            hls_playlist_type = None
            hls_list_size = "8"
            hls_flags = "delete_segments+omit_endlist"
            start_num = int(time.time()) % 100000
        enc_cmd += [
            "-f", "hls", "-hls_time", "4", "-hls_list_size", hls_list_size,
            "-start_number", str(start_num),
            "-hls_flags", hls_flags,
        ]
        if hls_playlist_type:
            enc_cmd += ["-hls_playlist_type", hls_playlist_type]
        enc_cmd += [
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
        dec_cmd = [FFMPEG, "-y"]
        if self.seek_offset > 0:
            dec_cmd += ["-ss", f"{self.seek_offset:.3f}"]
        dec_cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        if not self.is_vod:
            dec_cmd += ["-c:v", "h264_cuvid"]

        # Build video filter chain — scale to output resolution
        self.pad_y = 0
        vf = f"fps={FPS:.2f},scale_cuda={W}:{H},hwdownload,format=nv12"

        dec_cmd += [
            "-i", self.stream_url,
            "-map", "0:v:0",
            "-vf", vf,
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
                if self.is_vod:
                    self.completed = True
                    with self.stats_lock:
                        self.stats["status"] = "completed"
                        self.stats["last_error"] = ""
                    print(f"[{tag}] VOD playback completed", flush=True)
                else:
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
                elif censor_mode == "fine_blur":
                    # Finer blur: downscale to 1/12th (not 1/40th), lighter Gaussian, moderate color flood
                    tiny = cv2.resize(roi, (max(1, rw // 12), max(1, rh // 12)),
                                      interpolation=cv2.INTER_AREA)
                    blurred = cv2.resize(tiny, (rw, rh), interpolation=cv2.INTER_LINEAR)
                    ksize = max(31, (min(rh, rw) // 5) | 1)
                    blurred = cv2.GaussianBlur(blurred, (ksize, ksize), 0)
                    avg = blurred.mean(axis=(0, 1)).astype(np.float32)
                    blurred = (blurred.astype(np.float32) * 0.5 + avg * 0.5).astype(np.uint8)
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
                if self.is_vod:
                    apply_overlay(frame, ov, 16, 16)
                else:
                    ov_y = H - self.pad_y - ov.shape[0] - 16
                    apply_overlay(frame, ov, 16, max(0, ov_y))

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
                try:
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    self.last_frame_jpeg = jpeg.tobytes()
                except Exception:
                    pass
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

        if self.is_vod:
            if self.completed:
                # Ensure ENDLIST is written (safety fallback)
                m3u8_path = self.out_dir / f"{self.slug}.m3u8"
                try:
                    if m3u8_path.exists():
                        content = m3u8_path.read_text()
                        if "#EXT-X-ENDLIST" not in content:
                            with open(m3u8_path, "a") as f:
                                f.write("#EXT-X-ENDLIST\n")
                except Exception:
                    pass
                # Delayed cleanup: give clients 5 min to finish downloading segments
                gen_at_cleanup = self._pipeline_gen
                def _delayed_vod_cleanup(xui_id, out_dir, gen):
                    time.sleep(300)
                    if self._pipeline_gen != gen:
                        print(f"[{tag}] VOD cleanup cancelled (pipeline restarted)", flush=True)
                        return
                    with vod_lock:
                        vod_active.pop(xui_id, None)
                    if out_dir.exists():
                        shutil.rmtree(out_dir, ignore_errors=True)
                    print(f"[{tag}] VOD cleanup complete", flush=True)
                threading.Thread(target=_delayed_vod_cleanup,
                                 args=(self.xui_id, self.out_dir, gen_at_cleanup), daemon=True).start()
                print(f"[{tag}] VOD completed, cleanup in 5 min", flush=True)
            else:
                # Stopped early — write ENDLIST so player can finish watching buffered content,
                # then delayed cleanup
                m3u8_path = self.out_dir / f"{self.slug}.m3u8"
                try:
                    if m3u8_path.exists():
                        content = m3u8_path.read_text()
                        if "#EXT-X-ENDLIST" not in content:
                            with open(m3u8_path, "a") as f:
                                f.write("#EXT-X-ENDLIST\n")
                except Exception:
                    pass
                gen_at_cleanup = self._pipeline_gen
                def _delayed_vod_cleanup(xui_id, out_dir, gen):
                    time.sleep(300)
                    if self._pipeline_gen != gen:
                        print(f"[{tag}] VOD cleanup cancelled (pipeline restarted)", flush=True)
                        return
                    with vod_lock:
                        vod_active.pop(xui_id, None)
                    if out_dir.exists():
                        shutil.rmtree(out_dir, ignore_errors=True)
                    print(f"[{tag}] VOD cleanup complete", flush=True)
                threading.Thread(target=_delayed_vod_cleanup,
                                 args=(self.xui_id, self.out_dir, gen_at_cleanup), daemon=True).start()
                print(f"[{tag}] VOD stopped, segments kept for 5 min", flush=True)
            with self.stats_lock:
                if not self.completed:
                    self.stats.update({"status": "listening", "fps": 0.0,
                                       "detections_count": 0, "frames_processed": 0,
                                       "uptime": 0.0, "last_error": "", "animal_counts": {}})
        else:
            with self.stats_lock:
                self.stats.update({"status": "listening", "fps": 0.0,
                                   "detections_count": 0, "frames_processed": 0,
                                   "uptime": 0.0, "last_error": "", "animal_counts": {}})
            print(f"[{tag}] pipeline stopped", flush=True)

    def run_pipeline_stdout(self):
        """Pipeline outputting MPEG-TS to encoder stdout. Runs in a background thread."""
        tag = self.slug

        with self.stats_lock:
            self.stats.update({"status": "probing", "fps": 0, "detections_count": 0,
                               "frames_processed": 0, "uptime": 0, "last_error": ""})

        # --- Probe source (retry once for seek scenarios) ---
        probe = probe_stream(self.stream_url)
        if probe is None and self.is_vod:
            time.sleep(1)
            probe = probe_stream(self.stream_url)
        if probe is None:
            with self.stats_lock:
                self.stats["last_error"] = "Failed to probe stream"
                self.stats["status"] = "listening"
            self.pipeline_ready.set()
            return
        has_audio = probe["has_audio"]
        src_w, src_h = probe.get("width", 0), probe.get("height", 0)
        print(f"[{tag}] probe: src={src_w}x{src_h} fps={probe['fps']:.1f} audio={has_audio} out={W}x{H}@{FPS}", flush=True)

        self.stop_event.clear()
        self.pipeline_procs = []

        # --- Create TCP server for decoder video (Python listens) ---
        v_in_srv = tcp_server(self.v_in_port)

        # --- Start encoder FFmpeg (MPEG-TS to stdout) ---
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
        enc_cmd += ["-f", "mpegts", "pipe:1"]

        print(f"[{tag}] encoder (stdout): {' '.join(enc_cmd)}", flush=True)
        enc_proc = subprocess.Popen(
            enc_cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.ABOVE_NORMAL_PRIORITY_CLASS,
        )
        self.pipeline_procs.append(enc_proc)
        self.encoder_proc = enc_proc
        drain_stderr(enc_proc)

        # --- Wait for encoder's audio TCP listener ---
        if has_audio:
            if wait_for_port(self.a_port):
                print(f"[{tag}] encoder audio TCP port {self.a_port} ready", flush=True)
            else:
                print(f"[{tag}] WARNING: audio port check timed out", flush=True)

        # --- Start decoder FFmpeg ---
        dec_cmd = [FFMPEG, "-y",
                   "-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        if not self.is_vod:
            dec_cmd += ["-c:v", "h264_cuvid"]
        if self.seek_offset > 0:
            dec_cmd += ["-ss", f"{self.seek_offset:.3f}"]

        # Build video filter chain — scale to output resolution
        self.pad_y = 0
        vf = f"fps={FPS:.2f},scale_cuda={W}:{H},hwdownload,format=nv12"

        dec_cmd += [
            "-i", self.stream_url,
            "-map", "0:v:0",
            "-vf", vf,
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
        drain_stderr(dec_proc)

        # --- Accept decoder video connection ---
        try:
            v_in_conn, _ = v_in_srv.accept()
            v_in_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            v_in_conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, FRAME_BYTES * 4)
            print(f"[{tag}] decoder video TCP connected", flush=True)
        except socket.timeout:
            with self.stats_lock:
                self.stats["last_error"] = "Decoder video TCP accept timed out"
                self.stats["status"] = "listening"
            v_in_srv.close()
            self.pipeline_ready.set()
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
            self.pipeline_ready.set()
            return
        print(f"[{tag}] encoder video TCP port {self.v_out_port} ready", flush=True)

        # --- Connect Python to encoder's video port ---
        v_out_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        v_out_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            v_out_conn.connect(('127.0.0.1', self.v_out_port))
            v_out_conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, FRAME_BYTES * 4)
            print(f"[{tag}] encoder video TCP connected", flush=True)
        except (ConnectionRefusedError, OSError) as e:
            with self.stats_lock:
                self.stats["last_error"] = f"Encoder video connect failed: {e}"
                self.stats["status"] = "listening"
            v_in_conn.close()
            self.pipeline_ready.set()
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

        # Signal that pipeline is ready — Flask can start reading encoder stdout
        self.pipeline_ready.set()
        print(f"[{tag}] stdout pipeline started", flush=True)

        # -------------------------------------------------------------------
        # MAIN LOOP: recv -> detect -> draw -> send
        # -------------------------------------------------------------------
        while not self.stop_event.is_set():
            if not recv_exact(v_in_conn, FRAME_BYTES, frame_buf):
                if self.is_vod:
                    self.completed = True
                    with self.stats_lock:
                        self.stats["status"] = "completed"
                        self.stats["last_error"] = ""
                    print(f"[{tag}] VOD playback completed", flush=True)
                else:
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
                for t in tracker.tracks:
                    if t["id"] >= next_id_before:
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
                    tiny = cv2.resize(roi, (max(1, rw // 40), max(1, rh // 40)),
                                      interpolation=cv2.INTER_AREA)
                    blurred = cv2.resize(tiny, (rw, rh), interpolation=cv2.INTER_LINEAR)
                    ksize = max(51, (min(rh, rw) // 3) | 1)
                    blurred = cv2.GaussianBlur(blurred, (ksize, ksize), 0)
                    avg = blurred.mean(axis=(0, 1)).astype(np.float32)
                    blurred = (blurred.astype(np.float32) * 0.3 + avg * 0.7).astype(np.uint8)
                    frame[y1c:y2c, x1c:x2c] = (roi * (1 - mask) + blurred * mask).astype(np.uint8)
                elif censor_mode == "fine_blur":
                    tiny = cv2.resize(roi, (max(1, rw // 12), max(1, rh // 12)),
                                      interpolation=cv2.INTER_AREA)
                    blurred = cv2.resize(tiny, (rw, rh), interpolation=cv2.INTER_LINEAR)
                    ksize = max(31, (min(rh, rw) // 5) | 1)
                    blurred = cv2.GaussianBlur(blurred, (ksize, ksize), 0)
                    avg = blurred.mean(axis=(0, 1)).astype(np.float32)
                    blurred = (blurred.astype(np.float32) * 0.5 + avg * 0.5).astype(np.uint8)
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
                if self.is_vod:
                    apply_overlay(frame, ov, 16, 16)
                else:
                    ov_y = H - self.pad_y - ov.shape[0] - 16
                    apply_overlay(frame, ov, 16, max(0, ov_y))

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
                try:
                    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    self.last_frame_jpeg = jpeg.tobytes()
                except Exception:
                    pass
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

        if self.is_vod:
            gen_at_cleanup = self._pipeline_gen
            def _delayed_vod_cleanup(xui_id, inst_ref, gen):
                time.sleep(300)
                if self._pipeline_gen != gen:
                    print(f"[{tag}] VOD stdout cleanup cancelled (pipeline restarted)", flush=True)
                    return
                with vod_lock:
                    if vod_active.get(xui_id) is inst_ref:
                        vod_active.pop(xui_id, None)
                print(f"[{tag}] VOD stdout cleanup complete", flush=True)
            threading.Thread(target=_delayed_vod_cleanup,
                             args=(self.xui_id, self, gen_at_cleanup), daemon=True).start()
            with self.stats_lock:
                if not self.completed:
                    self.stats.update({"status": "listening", "fps": 0.0,
                                       "detections_count": 0, "frames_processed": 0,
                                       "uptime": 0.0, "last_error": "", "animal_counts": {}})
        else:
            with self.stats_lock:
                self.stats.update({"status": "listening", "fps": 0.0,
                                   "detections_count": 0, "frames_processed": 0,
                                   "uptime": 0.0, "last_error": "", "animal_counts": {}})
        print(f"[{tag}] stdout pipeline stopped", flush=True)


# ---------------------------------------------------------------------------
# M3U playlist fetching
# ---------------------------------------------------------------------------

def fetch_m3u():
    """Download M3U playlist, extract Hallmark channels and VOD movies."""
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
            movies = {}
            pending_extinf = None  # raw #EXTINF line
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if pending_extinf is not None:
                    # This line is the URL for the previous #EXTINF
                    extinf = pending_extinf
                    pending_extinf = None

                    # --- Check for matching live channel ---
                    extinf_lower = extinf.lower()
                    with config_lock:
                        filters = list(config.get("channel_filters", ["hallmark"]))
                    if any(f in extinf_lower for f in filters) and len(channels) < MAX_STREAMS:
                        m = re.search(r'tvg-name="([^"]*)"', extinf, re.IGNORECASE)
                        name = m.group(1) if m else "Channel"
                        logo_m = re.search(r'tvg-logo="([^"]*)"', extinf, re.IGNORECASE)
                        logo = logo_m.group(1) if logo_m else ""
                        xid_m = re.search(r'xui-id="([^"]*)"', extinf, re.IGNORECASE)
                        xui_id_ch = xid_m.group(1) if xid_m else ""
                        tid_m = re.search(r'tvg-id="([^"]*)"', extinf, re.IGNORECASE)
                        tvg_id_ch = tid_m.group(1) if tid_m else ""
                        grp_m = re.search(r'group-title="([^"]*)"', extinf, re.IGNORECASE)
                        group_ch = grp_m.group(1) if grp_m else "United States"
                        slug = slugify(name)
                        if not any(c["slug"] == slug for c in channels):
                            channels.append({"name": name, "url": line, "slug": slug, "logo": logo,
                                             "xui_id": xui_id_ch, "tvg_id": tvg_id_ch, "group": group_ch})

                    # --- Check for VOD movie (URL ends with #.mkv / #.mp4 / #.avi) ---
                    xui_m = re.search(r'xui-id="([^"]*)"', extinf, re.IGNORECASE)
                    ext_m = re.search(r'#\.(mkv|mp4|avi)$', line, re.IGNORECASE)
                    if xui_m and ext_m:
                        xui_id = xui_m.group(1)
                        orig_ext = ext_m.group(1).lower()
                        name_m = re.search(r'tvg-name="([^"]*)"', extinf, re.IGNORECASE)
                        vod_name = name_m.group(1) if name_m else f"Movie {xui_id}"
                        # Skip TV show episodes (e.g. "S01E01", "S2E14")
                        if re.search(r'S\d+E\d+', vod_name, re.IGNORECASE):
                            continue
                        logo_m = re.search(r'tvg-logo="([^"]*)"', extinf, re.IGNORECASE)
                        group_m = re.search(r'group-title="([^"]*)"', extinf, re.IGNORECASE)
                        vod_logo = logo_m.group(1) if logo_m else ""
                        vod_group = group_m.group(1) if group_m else ""
                        # Strip #.ext fragment from URL so FFmpeg gets clean URL
                        clean_url = re.sub(r'#\.[a-zA-Z0-9]+$', '', line)
                        movies[xui_id] = {
                            "xui_id": xui_id,
                            "name": vod_name,
                            "url": clean_url,
                            "logo": vod_logo,
                            "group": vod_group,
                            "ext": orig_ext,
                        }

                elif line.startswith("#EXTINF"):
                    pending_extinf = line

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

        # Store VOD catalog
        with vod_lock:
            vod_catalog.clear()
            vod_catalog.update(movies)
        if movies:
            print(f"[m3u] found {len(movies)} VOD movies", flush=True)

        # Persist M3U state to settings file
        with config_lock:
            _save_config()

        # Create/update StreamInstance objects
        _sync_streams(channels)

    except Exception as e:
        with m3u_lock:
            m3u_state["fetching"] = False
            m3u_state["last_error"] = str(e)
        print(f"[m3u] fetch error: {e}", flush=True)


def _sync_streams(channels):
    """Create StreamInstance objects for discovered channels."""
    with config_lock:
        excluded = set(config.get("excluded_channels", []))
    with streams_lock:
        for i, ch in enumerate(channels):
            slug = ch["slug"]
            if slug in excluded:
                continue
            port_base = 19876 + (i * 10)
            if slug not in streams:
                inst = StreamInstance(slug, ch["name"], ch["url"], port_base)
                inst.logo = ch.get("logo", "")
                inst.tvg_id = ch.get("tvg_id", "")
                inst.xui_id = ch.get("xui_id", "")
                inst.group = ch.get("group", "United States")
                streams[slug] = inst
                print(f"[streams] created instance '{slug}' ports {port_base}-{port_base+2}", flush=True)
            else:
                # Update URL/metadata in case it changed
                streams[slug].stream_url = ch["url"]
                streams[slug].channel_name = ch["name"]
                streams[slug].logo = ch.get("logo", "")
                streams[slug].tvg_id = ch.get("tvg_id", "")
                streams[slug].xui_id = ch.get("xui_id", "")
                streams[slug].group = ch.get("group", "United States")


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
        # Live streams
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
        # VOD pipelines
        with vod_lock:
            vod_list = list(vod_active.items())
        for xui_id, vod_inst in vod_list:
            with vod_inst.stats_lock:
                st = vod_inst.stats["status"]
            if st not in ("running", "probing"):
                continue
            if vod_inst.last_client_request > 0 and (time.time() - vod_inst.last_client_request) > VOD_CLIENT_TIMEOUT:
                print(f"[watchdog] [vod-{xui_id}] no client requests for {VOD_CLIENT_TIMEOUT}s, stopping", flush=True)
                vod_inst.stop()


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
.topbar-conn{display:flex;flex-direction:column;gap:3px;background:#0a0a14;border:1px solid #0f3460;border-radius:10px;padding:8px 14px}
.topbar-conn-row{display:flex;align-items:center;gap:6px;white-space:nowrap}
.topbar-conn code{font-size:0.8em;color:#53d8fb}
.topbar-conn .btn-copy{padding:3px 8px;font-size:0.72em}
.topbar-label{color:#666;font-size:0.7em;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;min-width:42px}
.topbar-sep{color:#333;font-size:0.75em}
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
.filter-tag{display:inline-flex;align-items:center;gap:4px;background:#0f3460;color:#53d8fb;padding:4px 10px;border-radius:6px;font-size:0.82em;font-family:'SF Mono',Monaco,Consolas,monospace}
.filter-tag button{background:none;border:none;color:#e94560;cursor:pointer;font-size:1em;padding:0 2px;line-height:1}
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
.stream-preview{width:100%;border-bottom:1px solid rgba(15,52,96,0.4);background:#0a0a14;display:none;position:relative;overflow:hidden}
.stream-preview img{width:100%;display:block;aspect-ratio:16/9;object-fit:contain;background:#000}
.stream-preview .preview-overlay{position:absolute;top:8px;right:8px;background:rgba(0,0,0,0.7);color:#53d8fb;font-size:0.7em;padding:2px 8px;border-radius:4px;font-family:'SF Mono',Monaco,Consolas,monospace}
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
<div class="topbar-conn">
<div class="topbar-conn-row"><span class="topbar-label">M3U</span><code id="playlistUrl"></code><button class="btn-copy" onclick="copyText(document.getElementById('playlistUrl').textContent,this)">Copy</button></div>
<div class="topbar-conn-row"><span class="topbar-label">Xtream</span><code id="xcServer"></code> <span style="color:#666;font-size:0.72em">user</span> <code>x</code> <span style="color:#666;font-size:0.72em">pass</span> <code>x</code></div>
</div>
</div>
</div>
<div class="container">

<div class="section-label">Channels <span id="channelCount" style="color:#53d8fb;font-size:1em"></span></div>
<div class="stream-cards" id="streamCards">
<div class="card" style="text-align:center;color:#a0a0b0;padding:40px">
No channels discovered yet. Configure M3U source below and click Fetch Now.
</div>
</div>

<div class="card" style="margin-top:12px;padding:14px 20px">
<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px">
<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
<span style="font-size:0.75em;color:#a0a0b0;text-transform:uppercase;letter-spacing:1px;font-weight:600">Filters</span>
<div id="filterTags" style="display:flex;gap:6px;flex-wrap:wrap"></div>
</div>
<div style="display:flex;gap:6px;align-items:center">
<input type="text" id="filterInput" placeholder="add keyword..." style="padding:6px 10px;border-radius:8px;border:1px solid #0f3460;background:#0f0f1a;color:#e0e0e0;font-size:0.82em;width:150px" onkeydown="if(event.key==='Enter')addFilter()">
<button class="btn-start" style="padding:5px 12px;font-size:0.78em" onclick="addFilter()">Add</button>
</div>
<div id="excludedChannels" style="display:flex;gap:6px;flex-wrap:wrap;margin-top:8px"></div>
</div>
</div>

<div class="section-label" style="margin-top:24px" id="vodSectionLabel">Active VOD</div>
<div class="stream-cards" id="vodCards" style="display:none"></div>

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
<option value="fine_blur">Fine Blur</option>
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
    const countEl=document.getElementById('channelCount');
    if(countEl)countEl.textContent=slugs.length>0?'('+slugs.length+')':'';
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
<div class="stream-preview" id="preview-wrap-${slug}"><img id="preview-${slug}" alt="Preview"><div class="preview-overlay" id="preview-fps-${slug}"></div></div>
<div class="stream-card-header">
<div class="stream-logo-placeholder" id="logo-${slug}">${initial}</div>
<div class="stream-title">
<div class="channel-name" id="chname-${slug}"></div>
<div class="stream-status"><span class="status-dot" id="dot-${slug}"></span><span id="st-${slug}"></span></div>
</div>
<button class="btn-stop" id="stopbtn-${slug}" style="padding:5px 12px;font-size:0.75em;display:none" onclick="post('/api/stop/${slug}',{})">Stop</button>
<button class="btn-stop" style="padding:5px 10px;font-size:0.75em;background:#1a1a2e;border:1px solid #333;color:#e94560" onclick="removeChannel('${slug}')" title="Remove channel">&#215;</button>
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
        const pw=document.getElementById('preview-wrap-'+slug);const pi=document.getElementById('preview-'+slug);const pf=document.getElementById('preview-fps-'+slug);
        if(d.status==='running'&&d.frames_processed>0){pw.style.display='block';pi.src='/api/preview/'+slug+'?t='+Date.now();pf.textContent=d.fps.toFixed(1)+' fps'}else{pw.style.display='none'}
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

function renderVodCards(vodStats){
    const container=document.getElementById('vodCards');
    const label=document.getElementById('vodSectionLabel');
    const ids=Object.keys(vodStats);
    if(ids.length===0){container.style.display='none';label.style.display='none';return}
    container.style.display='flex';label.style.display='block';
    ids.forEach(xid=>{
        const d=vodStats[xid];
        let card=document.getElementById('vod-card-'+xid);
        if(!card){
            card=document.createElement('div');card.className='stream-card';card.id='vod-card-'+xid;
            card.innerHTML=`
<div class="stream-preview" id="vod-preview-wrap-${xid}"><img id="vod-preview-${xid}" alt="Preview"><div class="preview-overlay" id="vod-preview-fps-${xid}"></div></div>
<div class="stream-card-header">
<div class="stream-logo-placeholder" style="background:linear-gradient(135deg,#e94560,#1a1a2e);font-size:0.7em">VOD</div>
<div class="stream-title">
<div class="channel-name" id="vod-name-${xid}"></div>
<div class="stream-status"><span class="status-dot" id="vod-dot-${xid}"></span><span id="vod-st-${xid}"></span></div>
</div>
<button class="btn-stop" style="padding:5px 12px;font-size:0.75em" onclick="post('/api/vod/stop/${xid}',{})">Stop</button>
</div>
<div class="stream-card-body">
<div class="stats-row">
<div class="mini-stat"><div class="label">FPS</div><div class="value fps-val" id="vod-fps-${xid}">0.0</div></div>
<div class="mini-stat"><div class="label">Detections</div><div class="value det-val" id="vod-det-${xid}">0</div></div>
<div class="mini-stat"><div class="label">Frames</div><div class="value" id="vod-fr-${xid}">0</div></div>
<div class="mini-stat"><div class="label">Uptime</div><div class="value" id="vod-up-${xid}">0s</div></div>
</div>
</div>`;
            container.appendChild(card);
        }
        card.className='stream-card'+(d.status==='running'?' is-running':'');
        document.getElementById('vod-name-'+xid).textContent=d.channel_name||('Movie '+xid);
        document.getElementById('vod-dot-'+xid).className='status-dot status-'+(d.status==='completed'?'listening':d.status);
        document.getElementById('vod-st-'+xid).textContent=d.status;
        document.getElementById('vod-fps-'+xid).textContent=(d.fps||0).toFixed(1);
        document.getElementById('vod-det-'+xid).textContent=d.detections_count||0;
        document.getElementById('vod-fr-'+xid).textContent=d.frames_processed||0;
        document.getElementById('vod-up-'+xid).textContent=fmt(d.uptime||0);
        const vpw=document.getElementById('vod-preview-wrap-'+xid);const vpi=document.getElementById('vod-preview-'+xid);const vpf=document.getElementById('vod-preview-fps-'+xid);
        if(d.status==='running'&&(d.frames_processed||0)>0){vpw.style.display='block';vpi.src='/api/vod/preview/'+xid+'?t='+Date.now();vpf.textContent=(d.fps||0).toFixed(1)+' fps'}else{vpw.style.display='none'}
    });
    Array.from(container.children).forEach(el=>{
        if(!el.id||!el.id.startsWith('vod-card-')||!vodStats[el.id.substring(9)]){el.remove()}
    });
}

function syncSelect(id,val){const el=document.getElementById(id);if(el&&val!=null&&el.value!==String(val))el.value=val}
function syncRange(id,val){const el=document.getElementById(id);if(el&&val!=null&&String(el.value)!==String(val)){el.value=val;const v=document.getElementById(id+'Val');if(v)v.textContent=id==='confidence'?parseFloat(val).toFixed(2):val}}
let _resolution='-';
function poll(){fetch('/api/stats').then(r=>r.json()).then(d=>{const vodData=d._vod||{};delete d._vod;const discovered=d._discovered||[];delete d._discovered;if(d._config){syncSelect('overlay',d._config.overlay);syncSelect('censorMode',d._config.censor_mode);syncSelect('model',d._config.model);syncRange('confidence',d._config.confidence);syncRange('padding',d._config.padding);syncRange('persist',d._config.persist_frames);syncRange('smooth',d._config.smooth_window);if(d._config.resolution)_resolution=d._config.resolution;if(d._config.channel_filters)renderFilters(d._config.channel_filters);renderExcluded(discovered,d._config.excluded_channels||[]);updateOverlayPreview();delete d._config}renderStreamCards(d);renderVodCards(vodData)}).catch(()=>{})}
setInterval(poll,2000);poll();

function fmtDate(iso){if(!iso)return'Never';const d=new Date(iso);return d.toLocaleDateString('en-US',{month:'short',day:'numeric',year:'numeric'})+' '+d.toLocaleTimeString('en-US',{hour:'numeric',minute:'2-digit'})}
function updateM3u(d){document.getElementById('m3uLastFetched').textContent=fmtDate(d.last_fetched);document.getElementById('m3uError').textContent=d.last_error||'';document.getElementById('m3uFetchBtn').textContent=d.fetching?'Fetching\u2026':'Fetch Now';document.getElementById('m3uFetchBtn').disabled=d.fetching}
function pollM3u(){fetch('/api/m3u').then(r=>r.json()).then(updateM3u).catch(()=>{})}
function fetchM3u(){document.getElementById('m3uFetchBtn').textContent='Fetching\u2026';document.getElementById('m3uFetchBtn').disabled=true;post('/api/m3u/fetch',{}).then(()=>setTimeout(pollM3u,2000)).catch(()=>{document.getElementById('m3uFetchBtn').textContent='Fetch Now';document.getElementById('m3uFetchBtn').disabled=false})}
function saveM3u(){const u=document.getElementById('m3uUrl').value.trim();if(!u)return;post('/api/m3u',{m3u_url:u,fetch:true}).then(()=>{document.getElementById('m3uSaveBtn').textContent='Saved!';setTimeout(()=>{document.getElementById('m3uSaveBtn').textContent='Save';pollM3u()},2000)}).catch(()=>{})}
document.getElementById('playlistUrl').textContent='http://'+window.location.host+'/noanimals.m3u';
document.getElementById('xcServer').textContent='http://'+window.location.host;
fetch('/api/m3u').then(r=>r.json()).then(d=>{if(d.m3u_url)document.getElementById('m3uUrl').value=d.m3u_url;updateM3u(d)}).catch(()=>{});
setInterval(pollM3u,5000);
let _currentFilters=[],_filterDirty=false;
function renderFilters(filters){const key=JSON.stringify(filters);if(key===JSON.stringify(_currentFilters)&&!_filterDirty)return;_currentFilters=filters||[];_filterDirty=false;const c=document.getElementById('filterTags');if(!c)return;c.innerHTML='';filters.forEach(f=>{const t=document.createElement('span');t.className='filter-tag';const x=document.createElement('button');x.innerHTML='&#215;';x.onclick=function(e){e.stopPropagation();removeFilter(f)};t.textContent=f+' ';t.appendChild(x);c.appendChild(t)})}
function addFilter(){const inp=document.getElementById('filterInput');const raw=inp.value.trim();if(!raw)return;const newKws=raw.split(',').map(s=>s.trim().toLowerCase()).filter(s=>s&&!_currentFilters.includes(s));if(!newKws.length){inp.value='';return}const updated=_currentFilters.concat(newKws);inp.value='';_filterDirty=true;post('/api/settings',{channel_filters:updated}).then(()=>{renderFilters(updated);post('/api/m3u/fetch',{})}).catch(()=>{})}
function removeFilter(kw){const updated=_currentFilters.filter(f=>f!==kw);_filterDirty=true;post('/api/settings',{channel_filters:updated}).then(()=>{renderFilters(updated);post('/api/m3u/fetch',{})}).catch(()=>{})}
let _excludedChannels=[];
function renderExcluded(discovered,excluded){_excludedChannels=excluded||[];const c=document.getElementById('excludedChannels');if(!c)return;const items=discovered.filter(ch=>excluded.includes(ch.slug));if(!items.length){c.innerHTML='';return}c.innerHTML=items.map(ch=>'<span class="filter-tag" style="background:#1a1a2e;color:#a0a0b0;cursor:pointer" onclick="readdChannel(\''+ch.slug.replace(/'/g,"\\'")+'\')">+ '+ch.name+'</span>').join('')}
function removeChannel(slug){post('/api/channel/remove/'+slug,{}).catch(()=>{})}
function readdChannel(slug){post('/api/channel/add/'+slug,{}).catch(()=>{})}
</script>
</body></html>"""


@app.route("/")
def index():
    return Response(DASHBOARD_HTML, content_type="text/html")


@app.route("/noanimals.m3u")
def noanimals_m3u():
    """Serve an M3U playlist matching hivecast format — live channels + VOD movies."""
    base = request.host_url.rstrip("/")
    lines = ["#EXTM3U", '#EXT-X-SESSION-DATA:DATA-ID="com.xui.1_5_13"']
    # Live channels (no │ prefix — IPTV apps treat as live)
    with streams_lock:
        for slug, stream in streams.items():
            xui_attr = f' xui-id="{stream.xui_id}"' if getattr(stream, 'xui_id', '') else ''
            tvg_id_attr = f' tvg-id="{stream.tvg_id}"' if getattr(stream, 'tvg_id', '') else ''
            logo_attr = f' tvg-logo="{stream.logo}"' if getattr(stream, 'logo', '') else ''
            group = getattr(stream, 'group', 'United States')
            lines.append(f'#EXTINF:-1{xui_attr}{tvg_id_attr} tvg-name="{stream.channel_name}"{logo_attr} group-title="{group}",{stream.channel_name}')
            lines.append(f"{base}/stream/{slug}/{slug}.m3u8")
    # VOD movies (│ prefix on group-title — IPTV apps treat as VOD)
    with vod_lock:
        for xui_id, movie in vod_catalog.items():
            logo_attr = f' tvg-logo="{movie["logo"]}"' if movie.get("logo") else ''
            group_attr = f' group-title="{movie["group"]}"' if movie.get("group") else ''
            lines.append(f'#EXTINF:-1 xui-id="{xui_id}" tvg-name="{movie["name"]}"{logo_attr}{group_attr},{movie["name"]}')
            lines.append(f"{base}/vod/{xui_id}/movie.m3u8")
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
        just_started = False
        if st == "listening":
            print(f"[auto-start] [{slug}] client requested {filename}, starting pipeline", flush=True)
            stream.start()
            just_started = True
        # Wait for real playlist to appear (up to 60s)
        for _ in range(120):
            if fpath.exists() and fpath.stat().st_size > 100:
                break
            with stream.stats_lock:
                st = stream.stats["status"]
            # Don't break on "listening" right after we just started
            if st == "stopped" or st == "error":
                break
            if st == "listening" and not just_started:
                break
            time.sleep(0.5)
            stream.touch_client(request.remote_addr)
            just_started = False  # only skip first iteration

    fpath = stream.out_dir / filename
    if not fpath.exists():
        return "Not found", 404
    # For m3u8: rewrite relative segment paths to absolute URLs
    if filename.endswith(".m3u8"):
        base = request.host_url.rstrip("/")
        content = fpath.read_text()
        lines = []
        for line in content.splitlines():
            if line and not line.startswith("#"):
                lines.append(f"{base}/stream/{slug}/{line}")
            else:
                lines.append(line)
        return Response("\n".join(lines) + "\n", content_type="application/vnd.apple.mpegurl")
    return send_from_directory(str(stream.out_dir.resolve()), filename, mimetype="video/mp2t")


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


@app.route("/api/channel/remove/<slug>", methods=["POST"])
def api_channel_remove(slug):
    with streams_lock:
        stream = streams.pop(slug, None)
    if stream:
        stream.stop()
    with config_lock:
        excluded = config.get("excluded_channels", [])
        if slug not in excluded:
            excluded.append(slug)
            config["excluded_channels"] = excluded
        _save_config()
    return jsonify({"ok": True})


@app.route("/api/channel/add/<slug>", methods=["POST"])
def api_channel_add(slug):
    with config_lock:
        excluded = config.get("excluded_channels", [])
        config["excluded_channels"] = [s for s in excluded if s != slug]
        _save_config()
    # Re-sync streams from saved channels
    with m3u_lock:
        channels = list(m3u_state.get("channels", []))
    if channels:
        _sync_streams(channels)
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
        if "censor_mode" in data and data["censor_mode"] in ("black", "blur", "fine_blur", "pixelate", "color_match"):
            config["censor_mode"] = data["censor_mode"]
        if "channel_filters" in data:
            config["channel_filters"] = [f.strip().lower() for f in data["channel_filters"] if f.strip()]
        if "excluded_channels" in data:
            config["excluded_channels"] = [s.strip() for s in data["excluded_channels"] if s.strip()]
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
    # VOD active pipelines
    vod_stats = {}
    with vod_lock:
        vod_list = list(vod_active.items())
    for xui_id, vod_inst in vod_list:
        with vod_inst.stats_lock:
            vs = dict(vod_inst.stats)
        vs["channel_name"] = vod_inst.channel_name
        vs["xui_id"] = xui_id
        vod_stats[xui_id] = vs
    result["_vod"] = vod_stats
    with m3u_lock:
        result["_discovered"] = list(m3u_state.get("channels", []))
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
    with config_lock:
        _save_config()
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


# ---------------------------------------------------------------------------
# Preview frames
# ---------------------------------------------------------------------------

@app.route("/api/preview/<slug>")
def api_preview(slug):
    """Return latest processed frame as JPEG for a live stream."""
    with streams_lock:
        inst = streams.get(slug)
    if not inst or not inst.last_frame_jpeg:
        return Response(b"", status=204)
    return Response(inst.last_frame_jpeg, content_type="image/jpeg",
                    headers={"Cache-Control": "no-cache"})


@app.route("/api/vod/preview/<xui_id>")
def api_vod_preview(xui_id):
    """Return latest processed frame as JPEG for a VOD pipeline."""
    with vod_lock:
        inst = vod_active.get(xui_id)
    if not inst or not inst.last_frame_jpeg:
        return Response(b"", status=204)
    return Response(inst.last_frame_jpeg, content_type="image/jpeg",
                    headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------------------
# Xtream Codes API compatibility
# ---------------------------------------------------------------------------

def _xc_category_id(group_name):
    """Stable numeric category ID from group name."""
    return abs(hash(group_name)) % 999999 + 1


@app.route("/player_api.php", methods=["GET", "POST"])
@app.route("/panel_api.php", methods=["GET", "POST"])
def xc_player_api():
    """Xtream Codes API — main endpoint for IPTV apps."""
    # Support both GET params and POST form data
    def _param(key, default=""):
        return request.args.get(key, "") or request.form.get(key, "") or default
    action = _param("action")
    print(f"[xc-api] {request.method} {request.path} action={action!r} user={_param('username')!r} from={request.remote_addr}", flush=True)
    base = request.host_url.rstrip("/")
    ts = int(time.time())

    # Auth / server info (no action)
    if not action:
        return jsonify({
            "user_info": {
                "auth": 1,
                "username": _param("username", "noanimals"),
                "password": _param("password", "noanimals"),
                "message": "",
                "status": "Active",
                "exp_date": str(ts + 365 * 86400),
                "is_trial": "0",
                "active_cons": "0",
                "created_at": str(ts - 86400),
                "max_connections": "3",
                "allowed_output_formats": ["m3u8", "ts"],
            },
            "server_info": {
                "url": request.host.split(":")[0],
                "port": str(request.host.split(":")[-1]) if ":" in request.host else "8080",
                "https_port": "",
                "server_protocol": "http",
                "rtmp_port": "",
                "timezone": "America/Chicago",
                "timestamp_now": ts,
                "time_now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        })

    # Live categories
    if action == "get_live_categories":
        cats = {}
        with streams_lock:
            for stream in streams.values():
                g = getattr(stream, "group", "United States")
                if g not in cats:
                    cats[g] = {"category_id": str(_xc_category_id(g)),
                               "category_name": g, "parent_id": 0}
        return jsonify(list(cats.values()))

    # Live streams
    if action == "get_live_streams":
        cat_filter = _param("category_id")
        result = []
        with streams_lock:
            for i, (slug, stream) in enumerate(streams.items()):
                g = getattr(stream, "group", "United States")
                cat_id = str(_xc_category_id(g))
                if cat_filter and cat_filter != cat_id:
                    continue
                result.append({
                    "num": i + 1,
                    "name": stream.channel_name,
                    "stream_type": "live",
                    "stream_id": int(stream.xui_id) if getattr(stream, "xui_id", "") else i + 1,
                    "stream_icon": getattr(stream, "logo", ""),
                    "epg_channel_id": getattr(stream, "tvg_id", ""),
                    "added": str(ts),
                    "category_id": cat_id,
                    "custom_sid": "",
                    "tv_archive": 0,
                    "direct_source": "",
                    "tv_archive_duration": 0,
                })
        return jsonify(result)

    # VOD categories
    if action == "get_vod_categories":
        cats = {}
        with vod_lock:
            for movie in vod_catalog.values():
                g = movie.get("group", "")
                if not g:
                    continue
                # Strip │ prefix for display name
                display = re.sub(r'^[│|]\s*', '', g)
                if display not in cats:
                    cats[display] = {"category_id": str(_xc_category_id(g)),
                                     "category_name": display, "parent_id": 0}
        return jsonify(sorted(cats.values(), key=lambda c: c["category_name"]))

    # VOD streams
    if action == "get_vod_streams":
        cat_filter = _param("category_id")
        result = []
        with vod_lock:
            for i, (xui_id, movie) in enumerate(vod_catalog.items()):
                g = movie.get("group", "")
                cat_id = str(_xc_category_id(g))
                if cat_filter and cat_filter != cat_id:
                    continue
                result.append({
                    "num": i + 1,
                    "name": movie["name"],
                    "stream_type": "movie",
                    "stream_id": int(xui_id),
                    "stream_icon": movie.get("logo", ""),
                    "rating": "",
                    "rating_5based": 0,
                    "added": str(ts),
                    "category_id": cat_id,
                    "container_extension": movie.get("ext", "mkv"),
                    "custom_sid": "",
                    "direct_source": "",
                })
        return jsonify(result)

    # VOD info (single movie details)
    if action == "get_vod_info":
        vod_id = _param("vod_id")
        with vod_lock:
            movie = vod_catalog.get(vod_id)
        if not movie:
            return jsonify({}), 404
        return jsonify({
            "info": {"name": movie["name"], "cover": movie.get("logo", ""),
                     "plot": "", "genre": re.sub(r'^[│|]\s*', '', movie.get("group", "")),
                     "duration": ""},
            "movie_data": {"stream_id": int(vod_id), "container_extension": movie.get("ext", "mkv")},
        })

    return jsonify({}), 404


@app.route("/get.php")
def xc_get_m3u():
    """Xtream Codes M3U playlist endpoint."""
    return noanimals_m3u()


@app.route("/xmltv.php")
def xc_xmltv():
    """Xtream Codes EPG endpoint — return empty XMLTV."""
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n<tv generator-info-name="NoAnimals"></tv>'
    return Response(xml, content_type="application/xml")


@app.route("/live/<user>/<password>/<stream_id>")
@app.route("/live/<user>/<password>/<stream_id>.<ext>")
def xc_live_stream(user, password, stream_id, ext="m3u8"):
    """Xtream Codes live stream URL — proxy to our HLS endpoint."""
    # Find matching slug first, then release lock BEFORE calling stream_file
    # (stream_file also acquires streams_lock — Lock is not re-entrant)
    matched_slug = None
    with streams_lock:
        for slug, stream in streams.items():
            if getattr(stream, "xui_id", "") == stream_id:
                matched_slug = slug
                break
    if matched_slug:
        return stream_file(matched_slug, f"{matched_slug}.m3u8")
    return "Unknown stream", 404


@app.route("/live/<user>/<password>/<path:extra>")
def xc_live_extra(user, password, extra):
    """Serve HLS segments requested relative to live stream URL."""
    # Find matching stream and resolve path under lock, then serve outside lock
    matched_stream = None
    matched_dir = None
    with streams_lock:
        for slug, stream in streams.items():
            fpath = stream.out_dir / extra
            if fpath.exists():
                matched_stream = stream
                matched_dir = str(stream.out_dir.resolve())
                break
    if matched_stream:
        matched_stream.touch_client(request.remote_addr)
        ct = "application/vnd.apple.mpegurl" if extra.endswith(".m3u8") else "video/mp2t"
        return send_from_directory(matched_dir, extra, mimetype=ct)
    return "Not found", 404


def _proxy_vod_direct(movie):
    """Uncensored VOD proxy fallback — direct passthrough from upstream."""
    source_url = movie["url"]
    client_range = request.headers.get("Range")
    headers = {"User-Agent": "Mozilla/5.0"}
    if client_range:
        headers["Range"] = client_range

    print(f"[vod-proxy] fallback -> '{movie['name']}' "
          f"{'Range: ' + client_range if client_range else '(full)'}", flush=True)

    try:
        req = urllib.request.Request(source_url, headers=headers)
        try:
            upstream = urllib.request.urlopen(req, timeout=30)
            status_code = upstream.status
        except urllib.error.HTTPError as e:
            if e.code == 206:
                upstream = e
                status_code = 206
            else:
                print(f"[vod-proxy] upstream error {e.code} for '{movie['name']}'", flush=True)
                return Response(f"Upstream error: {e.code}", status=502)
    except Exception as e:
        print(f"[vod-proxy] connection failed for '{movie['name']}': {e}", flush=True)
        return Response("Upstream connection failed", status=502)

    resp_headers = {"Accept-Ranges": "bytes"}
    cl = upstream.headers.get("Content-Length")
    if cl:
        resp_headers["Content-Length"] = cl
    if status_code == 206:
        cr = upstream.headers.get("Content-Range")
        if cr:
            resp_headers["Content-Range"] = cr
    ct = upstream.headers.get("Content-Type", "video/x-matroska")
    resp_headers["Content-Type"] = ct

    CHUNK_SIZE = 256 * 1024

    def stream_upstream():
        try:
            while True:
                chunk = upstream.read(CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
        finally:
            upstream.close()

    return Response(stream_upstream(), status=status_code, headers=resp_headers)


@app.route("/movie/<user>/<password>/<stream_id>")
@app.route("/movie/<user>/<password>/<stream_id>.<ext>")
def xc_vod_stream(user, password, stream_id, ext="mkv"):
    """Xtream Codes VOD stream — redirect to HLS censoring pipeline for seekable playback."""
    with vod_lock:
        movie = vod_catalog.get(stream_id)
    if not movie:
        return "Unknown movie", 404

    print(f"[vod] {request.remote_addr} -> '{movie['name']}' (xui_id={stream_id}), redirecting to HLS pipeline", flush=True)
    return redirect(f"/vod/{stream_id}/movie.m3u8")


@app.route("/movie/<user>/<password>/<stream_id>/<path:extra>")
def xc_vod_extra(user, password, stream_id, extra):
    """Serve HLS segments requested relative to movie stream URL."""
    return vod_file(stream_id, extra)


@app.route("/<user>/<password>/<stream_id>")
@app.route("/<user>/<password>/<stream_id>.<ext>")
def xc_shorthand_stream(user, password, stream_id, ext=None):
    """Xtream Codes shorthand URL — some apps omit /live/ prefix."""
    # Try live stream first
    matched_slug = None
    with streams_lock:
        for slug, stream in streams.items():
            if getattr(stream, "xui_id", "") == stream_id:
                matched_slug = slug
                break
    if matched_slug:
        return stream_file(matched_slug, f"{matched_slug}.m3u8")
    return "Unknown stream", 404


# ---------------------------------------------------------------------------
# VOD routes
# ---------------------------------------------------------------------------

_vod_port_counter = 0
_vod_port_lock = threading.Lock()


def _next_vod_port():
    """Allocate the next VOD port base (each pipeline uses 3 ports)."""
    global _vod_port_counter
    with _vod_port_lock:
        base = VOD_PORT_BASE + (_vod_port_counter * 10)
        _vod_port_counter = (_vod_port_counter + 1) % 100
        return base


def _serve_synthetic_m3u8(inst, req):
    """Generate a full-length synthetic HLS playlist so IPTV apps show complete seek bar."""
    base = req.host_url.rstrip("/")
    HLS_TIME = 4.0
    total_segs = max(1, int(inst.total_duration / HLS_TIME) + 1)
    disc_segs = set(inst._discontinuities)

    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        "#EXT-X-TARGETDURATION:5",
        "#EXT-X-MEDIA-SEQUENCE:0",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        "#EXT-X-START:TIME-OFFSET=0,PRECISE=YES",
    ]
    for i in range(total_segs):
        remaining = inst.total_duration - i * HLS_TIME
        dur = min(HLS_TIME, remaining)
        if dur <= 0:
            break
        if i in disc_segs:
            lines.append("#EXT-X-DISCONTINUITY")
        lines.append(f"#EXTINF:{dur:.6f},")
        lines.append(f"{base}/vod/{inst.xui_id}/{inst.slug}_{i:05d}.ts")
    lines.append("#EXT-X-ENDLIST")
    return Response("\n".join(lines) + "\n", content_type="application/vnd.apple.mpegurl")


@app.route("/vod/<xui_id>/<path:filename>")
def vod_file(xui_id, filename):
    """Serve HLS playlist/segments for a VOD movie, auto-starting pipeline on first request.

    Supports forward seeking: if a segment that hasn't been processed yet is
    requested, the pipeline restarts with FFmpeg -ss to seek to that position.
    """
    # Look up movie in catalog
    with vod_lock:
        movie = vod_catalog.get(xui_id)
        inst = vod_active.get(xui_id)

    if not movie:
        return "Unknown movie", 404

    # Auto-start pipeline if not running
    if inst is None and (filename.endswith(".m3u8") or filename == "movie.m3u8"):
        with vod_lock:
            # Double-check under lock
            inst = vod_active.get(xui_id)
            if inst is None:
                active_count = sum(1 for v in vod_active.values() if not v.completed)
                if active_count >= MAX_VOD_PIPELINES:
                    for old_id, old_inst in list(vod_active.items()):
                        if not old_inst.completed:
                            print(f"[vod] stopping '{old_inst.channel_name}' to make room for new movie", flush=True)
                            old_inst.stop()
                            vod_active.pop(old_id, None)
                            break
                for old_id in [k for k, v in vod_active.items() if v.completed]:
                    vod_active.pop(old_id, None)
                port_base = _next_vod_port()
                slug = f"vod-{xui_id}"
                inst = StreamInstance(slug, movie["name"], movie["url"], port_base)
                inst.is_vod = True
                inst.xui_id = xui_id
                inst.logo = movie.get("logo", "")
                vod_active[xui_id] = inst
        inst.touch_client(request.remote_addr)
        print(f"[vod] starting pipeline for '{movie['name']}' (xui_id={xui_id})", flush=True)
        inst.start()

    if inst is None:
        return "Not found", 404

    inst.touch_client(request.remote_addr)

    # Map movie.m3u8 to the real playlist filename
    real_filename = filename
    if filename == "movie.m3u8":
        real_filename = f"{inst.slug}.m3u8"

    # --- Restart pipeline if it failed or was stopped and m3u8 is requested ---
    if real_filename.endswith(".m3u8"):
        with inst.stats_lock:
            st = inst.stats["status"]
        if st in ("listening", "stopped", "error"):
            # Only restart if no seek is in progress (seek temporarily sets "listening")
            if inst._seek_lock.acquire(blocking=False):
                try:
                    # Re-check status under seek lock to avoid race
                    with inst.stats_lock:
                        st = inst.stats["status"]
                    if st in ("listening", "stopped", "error"):
                        print(f"[vod] restarting pipeline for '{inst.channel_name}' (status was '{st}')", flush=True)
                        inst.completed = False
                        inst.stop_event.clear()
                        with inst.stats_lock:
                            inst.stats["status"] = "listening"
                            inst.stats["last_error"] = ""
                        inst.start()
                finally:
                    inst._seek_lock.release()

    # --- M3U8 playlist ---
    if real_filename.endswith(".m3u8"):
        # Wait for probe to finish so we know total_duration
        for _ in range(120):  # 60s
            if inst.total_duration > 0:
                break
            with inst.stats_lock:
                st = inst.stats["status"]
            if st in ("stopped", "error", "listening"):
                break
            time.sleep(0.5)
            inst.touch_client(request.remote_addr)

        # Serve synthetic full-length playlist if duration known
        if inst.total_duration > 0:
            return _serve_synthetic_m3u8(inst, request)

        # Fallback: serve encoder's real m3u8 (no seek support)
        fpath = inst.out_dir / real_filename
        for _ in range(60):
            if fpath.exists() and fpath.stat().st_size > 100:
                break
            with inst.stats_lock:
                st = inst.stats["status"]
            if st in ("stopped", "error", "completed"):
                break
            time.sleep(0.5)
            inst.touch_client(request.remote_addr)
        if not fpath.exists() or fpath.stat().st_size <= 100:
            return Response("Service Unavailable", status=503,
                            headers={"Retry-After": "5"})
        content = fpath.read_text()
        lines = []
        for line in content.splitlines():
            if line.strip() == "#EXTM3U":
                lines.append(line)
                lines.append("#EXT-X-START:TIME-OFFSET=0,PRECISE=YES")
                continue
            if line and not line.startswith("#"):
                base = request.host_url.rstrip("/")
                lines.append(f"{base}/vod/{inst.xui_id}/{line}")
            else:
                lines.append(line)
        return Response("\n".join(lines) + "\n", content_type="application/vnd.apple.mpegurl")

    # --- TS segment ---
    fpath = inst.out_dir / real_filename
    if not fpath.exists():
        # Parse segment number to detect seek
        seg_match = re.search(r'_(\d+)\.ts$', real_filename)
        if seg_match and inst.total_duration > 0:
            HLS_TIME = 4
            seg_num = int(seg_match.group(1))
            target_time = seg_num * HLS_TIME

            # Check current processing position
            with inst.stats_lock:
                st = inst.stats["status"]
                frames = inst.stats["frames_processed"]
            current_time = frames / FPS + inst.seek_offset

            # Decide: wait (segment is close) or seek-restart (far ahead/behind)
            need_seek = False
            if st == "probing":
                pass  # pipeline is restarting (likely from another seek), just wait
            elif st not in ("running",):
                need_seek = True
            elif target_time < inst.seek_offset:
                # Backward seek — before current pipeline start point
                need_seek = True
            elif target_time - current_time > 60:
                # Forward seek — more than 60s ahead of processing
                need_seek = True

            if need_seek:
                print(f"[vod] seeking to {target_time:.0f}s (seg {seg_num}) for '{inst.channel_name}'", flush=True)
                inst.seek_restart(target_time)

    # Streaming response: sends 200 immediately, data flows when segment appears.
    # This keeps the IPTV app's HTTP connection alive during the seek delay.
    client_ip = request.remote_addr  # capture before generator (no request context in generator)

    def _segment_stream():
        deadline = time.time() + 60
        while time.time() < deadline:
            if fpath.exists() and fpath.stat().st_size > 0:
                time.sleep(0.3)  # let FFmpeg finish writing the segment
                inst.touch_client(client_ip)
                try:
                    with open(fpath, "rb") as f:
                        yield f.read()
                except OSError:
                    pass
                return
            with inst.stats_lock:
                st = inst.stats["status"]
            if st in ("stopped", "error", "completed"):
                return
            time.sleep(0.2)
            inst.touch_client(client_ip)

    if fpath.exists() and fpath.stat().st_size > 0:
        return send_from_directory(str(inst.out_dir.resolve()), real_filename, mimetype="video/mp2t")
    return Response(_segment_stream(), content_type="video/mp2t")


@app.route("/api/vod/catalog")
def api_vod_catalog():
    """Paginated movie catalog. Query params: q (search), group, page, per_page."""
    q = request.args.get("q", "").lower().strip()
    group = request.args.get("group", "").strip()
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(200, max(1, int(request.args.get("per_page", 50))))

    with vod_lock:
        items = list(vod_catalog.values())

    if q:
        items = [m for m in items if q in m["name"].lower()]
    if group:
        items = [m for m in items if m.get("group", "").lower() == group.lower()]

    total = len(items)
    start = (page - 1) * per_page
    page_items = items[start:start + per_page]

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "movies": page_items,
    })


@app.route("/api/vod/groups")
def api_vod_groups():
    """List distinct genre groups from VOD catalog."""
    with vod_lock:
        groups = sorted(set(m["group"] for m in vod_catalog.values() if m.get("group")))
    return jsonify(groups)


@app.route("/api/vod/active")
def api_vod_active():
    """Currently running VOD pipelines."""
    with vod_lock:
        vod_list = list(vod_active.items())
    result = {}
    for xui_id, inst in vod_list:
        with inst.stats_lock:
            s = dict(inst.stats)
        s["channel_name"] = inst.channel_name
        s["xui_id"] = xui_id
        result[xui_id] = s
    return jsonify(result)


@app.route("/api/vod/stop/<xui_id>", methods=["POST"])
def api_vod_stop(xui_id):
    """Stop a running VOD pipeline."""
    with vod_lock:
        inst = vod_active.get(xui_id)
    if not inst:
        return jsonify({"error": "No active VOD pipeline for this movie"}), 404
    inst.stop()
    with vod_lock:
        vod_active.pop(xui_id, None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    # Clean output directory
    out_root = Path("./stream_output")
    if out_root.exists():
        shutil.rmtree(out_root, ignore_errors=True)

    print("Animal Censor Stream Server v3 — Multi-Stream")

    # Restore streams from saved channels, then refresh from M3U source
    with m3u_lock:
        saved_channels = list(m3u_state["channels"])
    if saved_channels:
        print(f"[startup] restoring {len(saved_channels)} channel(s) from saved settings", flush=True)
        _sync_streams(saved_channels)

    # Fetch M3U synchronously to refresh channel URLs (no-op if no URL configured)
    fetch_m3u()

    with streams_lock:
        if streams:
            print(f"Discovered {len(streams)} channel(s):")
            for slug, inst in streams.items():
                print(f"  [{slug}] {inst.channel_name} -> http://localhost:8080/stream/{slug}/{slug}.m3u8")
        else:
            print("No channels discovered yet. Configure M3U source in dashboard.")
    with vod_lock:
        if vod_catalog:
            print(f"VOD catalog: {len(vod_catalog)} movies")

    print(f"Dashboard: http://localhost:8080")
    threading.Thread(target=m3u_refresh_loop, daemon=True).start()
    threading.Thread(target=client_watchdog, daemon=True).start()
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
