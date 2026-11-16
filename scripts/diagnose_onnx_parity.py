"""Print one-image PyTorch/ONNX GPU/ONNX CPU outputs for parity diagnosis."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    models = (
        ("pytorch", Path("runs/experiment_mixed5_yolo26n_960/weights/best.pt"), 0),
        ("onnx_gpu", Path("runs/experiment_mixed5_yolo26n_960/weights/best.onnx"), 0),
        (
            "onnx_cpu",
            Path("runs/experiment_mixed5_yolo26n_960/weights/best.onnx"),
            "cpu",
        ),
    )
    for name, model_path, device in models:
        result = YOLO(str(model_path)).predict(
            str(args.image),
            imgsz=960,
            conf=0.11,
            iou=0.70,
            device=device,
            verbose=False,
        )[0]
        rows = [
            {
                "class_id": int(class_id),
                "confidence": round(float(confidence), 8),
                "bbox": [round(float(value), 3) for value in bbox],
            }
            for class_id, confidence, bbox in zip(
                result.boxes.cls,
                result.boxes.conf,
                result.boxes.xyxy,
            )
        ]
        print(name, rows)


if __name__ == "__main__":
    main()
