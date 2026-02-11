"""
Censor animals in a video file using YOLOv8 object detection.
Draws black boxes over any detected animals and re-encodes with audio.

Usage:
    python censor_animals.py input_video.mkv
    python censor_animals.py input_video.mkv --output censored.mkv
    python censor_animals.py input_video.mkv --confidence 0.2 --padding 30
"""

import argparse
import os
import shutil
import subprocess
import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# COCO class IDs that are animals
ANIMAL_CLASS_IDS = {
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    20,  # elephant
    21,  # bear
    22,  # zebra
    23,  # giraffe
}

FFMPEG_PATH = (
    shutil.which("ffmpeg")
    or os.environ.get("FFMPEG_PATH")
)
if not FFMPEG_PATH:
    raise RuntimeError("ffmpeg not found on PATH. Install FFmpeg or set the FFMPEG_PATH environment variable.")

# Smoothing parameters
PERSIST_FRAMES = 12    # Keep a box visible for this many frames after last detection
SMOOTH_WINDOW = 5      # Average box coordinates over this many frames


def iou(box_a, box_b):
    """Compute intersection-over-union of two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


class BoxTracker:
    """Tracks detection boxes across frames for temporal smoothing."""

    def __init__(self, persist_frames=PERSIST_FRAMES, smooth_window=SMOOTH_WINDOW):
        self.persist_frames = persist_frames
        self.smooth_window = smooth_window
        self.tracks = []  # list of active tracks
        self.next_id = 0

    def update(self, detections):
        """Update tracks with new frame detections. Returns smoothed boxes to draw.

        detections: list of [x1, y1, x2, y2] boxes (already padded)
        """
        # Try to match each detection to an existing track via IoU
        used_tracks = set()
        used_dets = set()

        # Score all pairs
        pairs = []
        for ti, track in enumerate(self.tracks):
            avg_box = self._avg_box(track["history"])
            for di, det in enumerate(detections):
                score = iou(avg_box, det)
                if score > 0.15:  # loose threshold to keep matching
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        for score, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            # Update existing track
            self.tracks[ti]["history"].append(detections[di])
            if len(self.tracks[ti]["history"]) > self.smooth_window:
                self.tracks[ti]["history"] = self.tracks[ti]["history"][-self.smooth_window:]
            self.tracks[ti]["frames_since_seen"] = 0
            used_tracks.add(ti)
            used_dets.add(di)

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in used_dets:
                self.tracks.append({
                    "id": self.next_id,
                    "history": [det],
                    "frames_since_seen": 0,
                })
                self.next_id += 1

        # Age out unmatched tracks
        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track["frames_since_seen"] += 1

        # Remove expired tracks
        self.tracks = [t for t in self.tracks if t["frames_since_seen"] <= self.persist_frames]

        # Return smoothed boxes for all active tracks
        result = []
        for track in self.tracks:
            result.append(self._avg_box(track["history"]))
        return result

    def _avg_box(self, history):
        """Average box coordinates over history for smooth edges."""
        arr = np.array(history)
        return arr.mean(axis=0).astype(int).tolist()


def get_video_info(input_path):
    """Get video metadata using ffprobe."""
    ffprobe = FFMPEG_PATH.replace("ffmpeg.exe", "ffprobe.exe")
    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(input_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)


def format_time(seconds):
    """Format seconds into h:mm:ss."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Censor animals in video files")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("--output", "-o", help="Output video file path (default: input_censored.ext)")
    parser.add_argument("--confidence", "-c", type=float, default=0.2,
                        help="Detection confidence threshold (default: 0.2)")
    parser.add_argument("--padding", "-p", type=int, default=30,
                        help="Extra pixels around detection box (default: 30)")
    parser.add_argument("--model", "-m", default="yolov8m.pt",
                        help="YOLO model to use (default: yolov8m.pt)")
    parser.add_argument("--persist", type=int, default=PERSIST_FRAMES,
                        help=f"Frames to keep box after detection lost (default: {PERSIST_FRAMES})")
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW,
                        help=f"Frames to average box coordinates over (default: {SMOOTH_WINDOW})")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Build output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + "_censored")

    # Temp file for video-only output (we'll mux audio in at the end)
    temp_video = output_path.with_stem(output_path.stem + "_tempvideo")

    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Model:      {args.model}")
    print(f"Confidence: {args.confidence}")
    print(f"Padding:    {args.padding}px")
    print(f"Persist:    {args.persist} frames")
    print(f"Smooth:     {args.smooth} frames")
    print()

    # Get video info
    print("Probing video...")
    info = get_video_info(input_path)
    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    width = int(video_stream["width"])
    height = int(video_stream["height"])

    fps_parts = video_stream["r_frame_rate"].split("/")
    fps = int(fps_parts[0]) / int(fps_parts[1])

    total_frames = int(video_stream.get("nb_frames", 0))
    if total_frames == 0:
        duration = float(info["format"].get("duration", 0))
        total_frames = int(duration * fps)

    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    has_subs = any(s["codec_type"] == "subtitle" for s in info["streams"])

    total_duration = total_frames / fps if fps > 0 else 0

    print(f"Resolution: {width}x{height}")
    print(f"FPS:        {fps:.2f}")
    print(f"Frames:     ~{total_frames}")
    print(f"Duration:   ~{format_time(total_duration)}")
    print(f"Audio:      {'yes' if has_audio else 'no'}")
    print()

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(args.model)
    model.to("cuda")
    print("Model loaded on GPU")
    print()

    # Use OpenCV to read frames and write video-only temp file
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print("Error: Could not open input video with OpenCV")
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))

    if not writer.isOpened():
        print("Error: Could not open video writer")
        sys.exit(1)

    tracker = BoxTracker(persist_frames=args.persist, smooth_window=args.smooth)
    frame_num = 0
    detections_total = 0
    start_time = time.time()

    print("Processing frames...")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model.predict(frame, conf=args.confidence, verbose=False, classes=list(ANIMAL_CLASS_IDS))

        # Collect raw detections with padding
        raw_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                x1 = max(0, x1 - args.padding)
                y1 = max(0, y1 - args.padding)
                x2 = min(width, x2 + args.padding)
                y2 = min(height, y2 + args.padding)
                raw_boxes.append([x1, y1, x2, y2])
                detections_total += 1

        # Get smoothed boxes from tracker
        smoothed_boxes = tracker.update(raw_boxes)

        # Draw black boxes over all tracked regions
        for box in smoothed_boxes:
            bx1 = max(0, box[0])
            by1 = max(0, box[1])
            bx2 = min(width, box[2])
            by2 = min(height, box[3])
            frame[by1:by2, bx1:bx2] = 0

        writer.write(frame)
        frame_num += 1

        if frame_num % 50 == 0 or frame_num == 1:
            elapsed = time.time() - start_time
            pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
            fps_actual = frame_num / elapsed if elapsed > 0 else 0
            remaining_frames = total_frames - frame_num
            eta = remaining_frames / fps_actual if fps_actual > 0 else 0
            movie_pos = format_time(frame_num / fps) if fps > 0 else "?"

            print(f"  {pct:5.1f}%  |  Frame {frame_num}/{total_frames}  |  "
                  f"Movie time: {movie_pos}/{format_time(total_duration)}  |  "
                  f"{fps_actual:.1f} fps  |  ETA: {format_time(eta)}  |  "
                  f"{detections_total} detections")

    cap.release()
    writer.release()

    elapsed_total = time.time() - start_time
    print(f"\nFrame processing done in {format_time(elapsed_total)}.")
    print(f"Muxing audio/subtitles with FFmpeg (NVENC re-encode)...")

    # Now use FFmpeg to combine the processed video with original audio/subs
    mux_cmd = [
        FFMPEG_PATH, "-y",
        "-i", str(temp_video),
        "-i", str(input_path),
        "-map", "0:v",
        "-c:v", "h264_nvenc",
        "-preset", "p4",
        "-cq", "20",
    ]

    if has_audio:
        mux_cmd += ["-map", "1:a", "-c:a", "copy"]
    if has_subs:
        mux_cmd += ["-map", "1:s", "-c:s", "copy"]

    mux_cmd.append(str(output_path))

    result = subprocess.run(mux_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg mux error:\n{result.stderr}")
        sys.exit(1)

    # Clean up temp file
    temp_video.unlink(missing_ok=True)

    output_size = output_path.stat().st_size / (1024 * 1024)
    input_size = input_path.stat().st_size / (1024 * 1024)

    print()
    print("Done!")
    print(f"  Frames processed:  {frame_num}")
    print(f"  Animal detections: {detections_total}")
    print(f"  Processing time:   {format_time(elapsed_total)}")
    print(f"  Avg speed:         {frame_num / elapsed_total:.1f} fps")
    print(f"  Input size:  {input_size:.1f} MB")
    print(f"  Output size: {output_size:.1f} MB")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
