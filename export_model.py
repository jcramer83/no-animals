"""
Export a YOLO .pt model to OpenVINO IR format for CPU/iGPU inference.

Usage:
    python export_model.py                          # exports yolov8x.pt
    python export_model.py --model yolov8n.pt       # exports specific model
    python export_model.py --model yolov8s.pt --half  # export with FP16
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to OpenVINO IR format")
    parser.add_argument("--model", "-m", default="yolov8x.pt",
                        help="YOLO model to export (default: yolov8x.pt)")
    parser.add_argument("--half", action="store_true",
                        help="Export with FP16 precision (smaller, faster on iGPU)")
    parser.add_argument("--imgsz", type=int, default=480,
                        help="Input image size (default: 480)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"Exporting to OpenVINO IR (imgsz={args.imgsz}, half={args.half})...")
    model.export(format="openvino", imgsz=args.imgsz, half=args.half)

    # The exported model directory name follows ultralytics convention:
    # yolov8x.pt -> yolov8x_openvino_model/
    model_stem = args.model.replace(".pt", "")
    out_dir = f"{model_stem}_openvino_model"
    print(f"Export complete: {out_dir}/")
    print(f"Use this model name in the dashboard: {out_dir}")


if __name__ == "__main__":
    main()
