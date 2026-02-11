"""
NoAnimals – System tray launcher for stream_animals_v3.py
Double-click this (or the .bat) to start the server with a tray icon.
"""

import subprocess
import sys
import os
import webbrowser
import threading
import time

from PIL import Image, ImageDraw, ImageFont
import pystray

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_SCRIPT = os.path.join(SCRIPT_DIR, "stream_animals_v3.py")
DASHBOARD_URL = "http://localhost:8080"
PYTHON = sys.executable

server_proc = None
icon = None


def create_icon_image(color="#22c55e"):
    """Draw a simple 64x64 icon: colored circle with 'NA' text."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # filled circle
    draw.ellipse([4, 4, 60, 60], fill=color, outline="#ffffff", width=2)
    # text
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), "NA", font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((64 - tw) / 2, (64 - th) / 2 - 2), "NA", fill="#ffffff", font=font)
    return img


def start_server():
    global server_proc
    if server_proc and server_proc.poll() is None:
        return  # already running
    env = os.environ.copy()
    # Ensure FFmpeg is findable even if not on PATH
    if "FFMPEG_PATH" not in env:
        ffmpeg_winget = os.path.expandvars(
            r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
        )
        if os.path.isdir(ffmpeg_winget):
            env["PATH"] = ffmpeg_winget + ";" + env.get("PATH", "")
    server_proc = subprocess.Popen(
        [PYTHON, SERVER_SCRIPT],
        cwd=SCRIPT_DIR,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW | subprocess.ABOVE_NORMAL_PRIORITY_CLASS,
    )
    update_icon_state()


def stop_server():
    global server_proc
    if server_proc and server_proc.poll() is None:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
    server_proc = None
    update_icon_state()


def update_icon_state():
    if icon is None:
        return
    running = server_proc is not None and server_proc.poll() is None
    icon.icon = create_icon_image("#22c55e" if running else "#ef4444")
    icon.title = "NoAnimals – Running" if running else "NoAnimals – Stopped"


def on_open_dashboard(icon_ref, item):
    webbrowser.open(DASHBOARD_URL)


def on_start(icon_ref, item):
    start_server()


def on_stop(icon_ref, item):
    stop_server()


def on_quit(icon_ref, item):
    stop_server()
    icon_ref.stop()


def health_monitor():
    """Watch the server process; update icon if it exits unexpectedly."""
    while True:
        time.sleep(3)
        if icon is None:
            break
        update_icon_state()


def main():
    global icon

    menu = pystray.Menu(
        pystray.MenuItem("Open Dashboard", on_open_dashboard, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Start Server", on_start),
        pystray.MenuItem("Stop Server", on_stop),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )

    icon = pystray.Icon(
        name="NoAnimals",
        icon=create_icon_image("#ef4444"),
        title="NoAnimals – Starting...",
        menu=menu,
    )

    # Start server, then health monitor
    start_server()
    threading.Thread(target=health_monitor, daemon=True).start()

    # Blocks until icon.stop() is called
    icon.run()


if __name__ == "__main__":
    main()
