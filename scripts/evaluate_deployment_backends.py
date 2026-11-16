"""Evaluate deterministic detector parity, validation metrics, and detector speed."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from visionops.deployment.parity import match_detections
from visionops.video.backend import DetectorBackend
from visionops.video.config import load_frozen_detector_config
from visionops.video.policy import CandidateBox, apply_class_thresholds

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {0: "helmet", 1: "vest", 2: "Person", 3: "no_helmet", 4: "no_vest"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _label_aware_sample(image_dir: Path, count: int) -> list[Path]:
    images = sorted(image_dir.glob("*.*"))
    label_dir = Path(str(image_dir).replace("images", "labels"))
    selected: list[Path] = []
    class_counts: Counter[int] = Counter()
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        classes = {
            int(line.split()[0])
            for line in label.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if classes and any(class_counts[class_id] < 3 for class_id in classes):
            selected.append(image)
            class_counts.update(classes)
        if len(selected) >= count:
            break
    for image in images:
        if len(selected) >= count:
            break
        if image not in selected:
            selected.append(image)
    return selected


def _video_frames(path: Path) -> list[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sorted({0, frame_count // 4, frame_count // 2, frame_count - 1})
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not sample frame {index} from {path}")
        frames.append((index, frame))
    capture.release()
    return frames


def build_parity_set() -> tuple[list[np.ndarray], list[dict[str, object]]]:
    arrays = []
    manifest = []
    datasets = (
        ("cppe_validation", ROOT / "data/generated/cppe-five-class/images/val"),
        (
            "rf100_filtered_validation",
            ROOT / "data/generated/rf100-valid-filtered/images/valid",
        ),
    )
    for source_name, directory in datasets:
        for path in _label_aware_sample(directory, 15):
            image = cv2.imread(str(path))
            if image is None:
                raise RuntimeError(f"Could not read parity image: {path}")
            arrays.append(image)
            manifest.append(
                {
                    "source": source_name,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": _sha256(path),
                    "width": image.shape[1],
                    "height": image.shape[0],
                }
            )
    for video in sorted((ROOT / "data/demo_videos").glob("*.mp4")):
        video_hash = _sha256(video)
        for frame_index, frame in _video_frames(video):
            arrays.append(frame)
            manifest.append(
                {
                    "source": "internal_real_video_frame",
                    "path": str(video.relative_to(ROOT)),
                    "video_sha256": video_hash,
                    "frame_index": frame_index,
                    "pixel_sha256": hashlib.sha256(frame.tobytes()).hexdigest().upper(),
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                }
            )
    return arrays, manifest


def _boxes(result: Any) -> list[CandidateBox]:
    if result.boxes is None:
        return []
    return [
        CandidateBox(
            class_id=int(result.boxes.cls[index].item()),
            confidence=float(result.boxes.conf[index].item()),
            bbox_xyxy=tuple(
                float(value) for value in result.boxes.xyxy[index].tolist()
            ),
        )
        for index in range(len(result.boxes))
    ]


def _predict(
    backend: DetectorBackend,
    image: np.ndarray,
    config: Any,
) -> tuple[list[CandidateBox], dict[str, float]]:
    result = backend.predict(
        image,
        imgsz=config.imgsz,
        confidence=config.inference_confidence_floor,
        iou=config.nms_iou,
        device=0,
    )
    candidates = apply_class_thresholds(
        _boxes(result), config.classes, config.confidence_thresholds
    )
    return candidates, {key: float(value) for key, value in result.speed.items()}


def parity(
    reference_backend: DetectorBackend,
    candidate_backend: DetectorBackend,
    images: list[np.ndarray],
    config: Any,
) -> dict[str, object]:
    reference_total = candidate_total = matched_total = 0
    ious: list[float] = []
    deltas: list[float] = []
    per_class = defaultdict(lambda: Counter(reference=0, candidate=0, matched=0))
    sample_results = []
    for index, image in enumerate(images):
        reference, _ = _predict(reference_backend, image, config)
        candidate, _ = _predict(candidate_backend, image, config)
        matches = match_detections(reference, candidate, minimum_iou=0.90)
        reference_total += len(reference)
        candidate_total += len(candidate)
        matched_total += len(matches)
        ious.extend(match.iou for match in matches)
        deltas.extend(match.confidence_delta for match in matches)
        for box in reference:
            per_class[box.class_id]["reference"] += 1
        for box in candidate:
            per_class[box.class_id]["candidate"] += 1
        for match in matches:
            per_class[match.class_id]["matched"] += 1
        sample_results.append(
            {
                "sample_index": index,
                "reference": len(reference),
                "candidate": len(candidate),
                "matched": len(matches),
            }
        )
    ious.sort()
    deltas.sort()

    def percentile(values: list[float], fraction: float) -> float | None:
        return values[round((len(values) - 1) * fraction)] if values else None

    summary = {
        "matching_iou": 0.90,
        "reference_detection_count": reference_total,
        "candidate_detection_count": candidate_total,
        "matched_detections": matched_total,
        "unmatched_reference_detections": reference_total - matched_total,
        "unmatched_candidate_detections": candidate_total - matched_total,
        "match_rate": matched_total / max(reference_total, candidate_total, 1),
        "mean_matched_iou": statistics.fmean(ious) if ious else None,
        "p5_matched_iou": percentile(ious, 0.05),
        "mean_absolute_confidence_delta": statistics.fmean(deltas) if deltas else None,
        "p95_absolute_confidence_delta": percentile(deltas, 0.95),
        "per_class": {
            CLASSES[class_id]: dict(per_class[class_id])
            for class_id in sorted(per_class)
        },
        "samples": sample_results,
    }
    summary["accepted"] = bool(
        summary["match_rate"] >= 0.995
        and summary["mean_matched_iou"] >= 0.995
        and summary["p95_absolute_confidence_delta"] <= 0.005
    )
    return summary


def detector_benchmark(
    backend: DetectorBackend,
    images: list[np.ndarray],
    config: Any,
    rounds: int = 3,
) -> dict[str, object]:
    for image in images[:5]:
        _predict(backend, image, config)
    wall_ms = []
    components: dict[str, list[float]] = defaultdict(list)
    if backend.name == "pytorch":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(rounds):
        for image in images:
            torch.cuda.synchronize()
            started = time.perf_counter()
            _, speed = _predict(backend, image, config)
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - started) * 1000)
            for stage, value in speed.items():
                components[stage].append(value)

    def summary(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "mean_ms": statistics.fmean(values),
            "p50_ms": statistics.median(values),
            "p95_ms": ordered[round((len(ordered) - 1) * 0.95)],
        }

    return {
        "warmup_samples": min(5, len(images)),
        "measured_rounds": rounds,
        "measured_samples": len(wall_ms),
        "wall_latency": summary(wall_ms),
        "throughput_fps": 1000 / statistics.fmean(wall_ms),
        "ultralytics_components": {
            stage: summary(values) for stage, values in sorted(components.items())
        },
        "peak_torch_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
            if backend.name == "pytorch"
            else None
        ),
        "memory_note": (
            "PyTorch CUDA allocator peak"
            if backend.name == "pytorch"
            else "ONNX Runtime GPU allocation is not exposed reliably by this API"
        ),
    }


def validation_metrics(model_path: Path, data_path: Path) -> dict[str, float]:
    result = YOLO(str(model_path)).val(
        data=str(data_path),
        imgsz=960,
        batch=1,
        workers=0,
        device=0,
        iou=0.7,
        plots=False,
        save_json=False,
        verbose=False,
        project=str(ROOT / "results/deployment_optimization/validation"),
        name=f"{model_path.suffix.removeprefix('.')}_{data_path.parent.name}",
        exist_ok=True,
    )
    return {
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/deployment_optimization/detector_evaluation.json",
    )
    args = parser.parse_args()
    config = load_frozen_detector_config(
        ROOT / "configs/production_detector.yaml", ROOT
    )
    images, manifest = build_parity_set()
    reference = DetectorBackend(
        "pytorch", config, ROOT / "configs/deployment_backends.yaml", ROOT
    )
    candidate = DetectorBackend(
        "onnx", config, ROOT / "configs/deployment_backends.yaml", ROOT
    )
    results = {
        "protocol": {
            "reference": "pytorch",
            "candidate": "onnx",
            "imgsz": 960,
            "batch": 1,
            "static_shape": True,
            "class_specific_thresholds": config.confidence_thresholds,
            "nms_iou": config.nms_iou,
            "acceptance": {
                "match_rate_minimum": 0.995,
                "mean_iou_minimum": 0.995,
                "p95_confidence_delta_maximum": 0.005,
            },
            "rf100_test_used": False,
        },
        "parity_set": manifest,
        "detection_parity": parity(reference, candidate, images, config),
        "detector_benchmark": {
            "pytorch": detector_benchmark(reference, images, config),
            "onnx": detector_benchmark(candidate, images, config),
        },
        "validation_metrics": {},
    }
    data_paths = {
        "cppe_validation": ROOT / "data/generated/cppe-five-class/data.yaml",
        "rf100_filtered_validation": (
            ROOT / "data/generated/rf100-valid-filtered/data.yaml"
        ),
    }
    for domain, data_path in data_paths.items():
        pytorch_metrics = validation_metrics(config.weights_path, data_path)
        onnx_metrics = validation_metrics(candidate.artifact_path, data_path)
        results["validation_metrics"][domain] = {
            "pytorch": pytorch_metrics,
            "onnx": onnx_metrics,
            "onnx_minus_pytorch": {
                key: onnx_metrics[key] - pytorch_metrics[key] for key in pytorch_metrics
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    docs_manifest = ROOT / "docs/experiments/deployment_optimization/parity_set.json"
    docs_manifest.parent.mkdir(parents=True, exist_ok=True)
    docs_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
