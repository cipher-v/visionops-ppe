"""Measure five-second input/output drift windows on the three internal videos."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from visionops.drift.features import image_features, output_features
from visionops.drift.monitor import (
    DriftReference,
    DriftWindowRecord,
    aggregate_input_features,
    fixed_time_windows,
    score_drift,
)

ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "data" / "demo_videos"
DETECTION_ROOT = ROOT / "results" / "deployment_optimization" / "video" / "pytorch"
OUTPUT_ROOT = ROOT / "results" / "robustness_drift" / "video"
REFERENCE_PATH = ROOT / "configs" / "drift_reference.json"
CLIPS = ("compliant_multiworker", "helmet_transition", "walking_multiworker")
WINDOW_SECONDS = 5.0


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
    }


def _load_detections(path: Path, frame_count: int) -> list[list[dict[str, object]]]:
    frames: list[list[dict[str, object]]] = [[] for _ in range(frame_count)]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            index = int(record["frame_index"])
            if not 0 <= index < frame_count:
                raise ValueError(f"Detection frame index outside video: {index}")
            frames[index].append(record)
    return frames


def _read_video(path: Path) -> tuple[list[dict[str, object]], list[float], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Video has invalid FPS: {path}")
    records: list[dict[str, object]] = []
    latencies: list[float] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        started = time.perf_counter()
        records.append(image_features(frame))
        latencies.append((time.perf_counter() - started) * 1000)
    capture.release()
    if not records:
        raise RuntimeError(f"Video contains no readable frames: {path}")
    return records, latencies, fps


def monitor_clip(name: str, reference: DriftReference) -> dict[str, object]:
    video_path = VIDEO_DIR / f"{name}.mp4"
    detection_path = DETECTION_ROOT / name / "detections.jsonl"
    input_records, feature_latencies, fps = _read_video(video_path)
    detection_frames = _load_detections(detection_path, len(input_records))
    timestamps = [index / fps for index in range(len(input_records))]

    records: list[DriftWindowRecord] = []
    scoring_latencies: list[float] = []
    for start, end in fixed_time_windows(timestamps, WINDOW_SECONDS):
        started = time.perf_counter()
        input_metrics = aggregate_input_features(input_records[start:end])
        output_metrics = output_features(detection_frames[start:end])
        drift_scores = score_drift(input_records[start:end], output_metrics, reference)
        scoring_latencies.append((time.perf_counter() - started) * 1000)
        records.append(
            DriftWindowRecord(
                source_id=str(video_path),
                window_start_seconds=timestamps[start],
                window_end_seconds=min(end / fps, len(input_records) / fps),
                input_metrics=input_metrics,
                output_metrics=output_metrics,
                drift_scores=drift_scores,
                overall_level=str(drift_scores["overall_level"]),
            )
        )

    output_dir = OUTPUT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "drift_windows.jsonl"
    output_path.write_text(
        "".join(json.dumps(record.to_dict()) + "\n" for record in records),
        encoding="utf-8",
    )
    levels = {level: 0 for level in ("normal", "warning", "high")}
    for record in records:
        levels[record.overall_level] += 1
    aggregate_output = output_features(detection_frames)
    return {
        "clip": name,
        "video": str(video_path),
        "existing_detection_source": str(detection_path),
        "frame_count": len(input_records),
        "fps": fps,
        "duration_seconds": len(input_records) / fps,
        "window_seconds": WINDOW_SECONDS,
        "window_count": len(records),
        "window_levels": levels,
        "window_scores": [
            {
                "start_seconds": record.window_start_seconds,
                "end_seconds": record.window_end_seconds,
                "overall_score": record.drift_scores["overall_score"],
                "input_score": record.drift_scores["input_score"],
                "output_class_frequency_psi": record.drift_scores[
                    "output_class_frequency_psi"
                ],
                "level": record.overall_level,
            }
            for record in records
        ],
        "aggregate_input_metrics": aggregate_input_features(input_records),
        "aggregate_output_metrics": aggregate_output,
        "feature_overhead_per_frame": _percentiles(feature_latencies),
        "scoring_overhead_per_window": _percentiles(scoring_latencies),
        "output": str(output_path),
    }


def main() -> None:
    reference = DriftReference.load(REFERENCE_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": {
            "reference": str(REFERENCE_PATH),
            "window_seconds": WINDOW_SECONDS,
            "detector_outputs_reused": True,
            "accuracy_claimed": False,
        },
        "clips": [monitor_clip(name, reference) for name in CLIPS],
    }
    summary_path = OUTPUT_ROOT / "real_video_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
