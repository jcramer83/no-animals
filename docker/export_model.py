"""
One-time YOLO model export to OpenVINO IR format.

Usage:
    python export_model.py                          # Export yolov8x.pt (default)
    python export_model.py --model yolov8m.pt       # Export a specific model
    python export_model.py --int8                    # Export with INT8 quantization (~2x faster)
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to OpenVINO IR format")
    parser.add_argument("--model", "-m", default="yolov8x.pt",
                        help="YOLO model to export (default: yolov8x.pt)")
    parser.add_argument("--int8", action="store_true",
                        help="Enable INT8 quantization (~2x faster, slight accuracy cost)")
    parser.add_argument("--imgsz", type=int, default=480,
                        help="Inference image size (default: 480)")
    args = parser.parse_args()

    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)

    print(f"Exporting to OpenVINO IR format (imgsz={args.imgsz}, int8={args.int8})...")
    model.export(format="openvino", imgsz=args.imgsz, int8=args.int8)

    output_dir = args.model.replace(".pt", "_openvino_model")
    print(f"Export complete! Model saved to: {output_dir}/")
    print(f"Use this model name in the dashboard: {output_dir}")


if __name__ == "__main__":
    main()
