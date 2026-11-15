"""Streaming inference pipeline for the frozen production detector."""

from __future__ import annotations

import json
import platform
import statistics
import time
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from visionops.tracking.metrics import TrackingDiagnostics
from visionops.tracking.schema import PersonTrack
from visionops.tracking.tracker import PersonByteTracker, TrackerSettings
from visionops.video.config import FrozenDetectorConfig, verify_weights
from visionops.video.policy import CandidateBox, apply_class_thresholds
from visionops.video.schema import VideoDetection, normalized_box
from visionops.video.source import AnnotatedVideoWriter, FramePacket, VideoReader


@dataclass(frozen=True)
class PipelineOptions:
    """Runtime controls that do not alter the frozen detector policy."""

    source: str
    output_dir: Path
    frame_stride: int = 1
    render: bool = False
    fallback_fps: float = 30.0
    device: str | int | None = None
    max_frames: int | None = None
    enable_tracking: bool = False
    tracker_config: Path | None = None

    def __post_init__(self) -> None:
        if self.frame_stride < 1:
            raise ValueError("Frame stride must be at least 1")
        if self.fallback_fps <= 0:
            raise ValueError("Fallback FPS must be positive")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("Maximum processed frames must be positive")
        if self.enable_tracking and self.tracker_config is None:
            raise ValueError("Tracking requires a tracker configuration path")


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) * 1000,
        "p50_ms": statistics.median(values) * 1000,
        "p95_ms": percentile(0.95) * 1000,
    }


def _component_summary(values_ms: list[float]) -> dict[str, float | int | None]:
    return _summary([value / 1000 for value in values_ms])


def _candidates(result: Any) -> list[CandidateBox]:
    boxes = result.boxes
    if boxes is None:
        return []
    return [
        CandidateBox(
            class_id=int(boxes.cls[index].item()),
            confidence=float(boxes.conf[index].item()),
            bbox_xyxy=tuple(float(value) for value in boxes.xyxy[index].tolist()),
        )
        for index in range(len(boxes))
    ]


def _structured_detections(
    packet: FramePacket,
    candidates: list[CandidateBox],
    config: FrozenDetectorConfig,
    source_id: str,
) -> list[VideoDetection]:
    height, width = packet.image.shape[:2]
    detections = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.bbox_xyxy
        clipped_box = (
            min(max(x1, 0.0), float(width)),
            min(max(y1, 0.0), float(height)),
            min(max(x2, 0.0), float(width)),
            min(max(y2, 0.0), float(height)),
        )
        detections.append(
            VideoDetection(
                frame_index=packet.frame_index,
                timestamp_seconds=packet.timestamp_seconds,
                class_id=candidate.class_id,
                class_name=config.classes[candidate.class_id],
                confidence=candidate.confidence,
                bbox_xyxy=clipped_box,
                bbox_normalized=normalized_box(clipped_box, width, height),
                source_id=source_id,
            )
        )
    return detections


def render_detections(
    frame: np.ndarray, detections: list[VideoDetection]
) -> np.ndarray:
    """Draw only detector class, confidence, and bounding box."""
    rendered = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection.bbox_xyxy)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_y = max(16, y1 - 5)
        cv2.putText(
            rendered,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    return rendered


def render_tracked_detections(
    frame: np.ndarray,
    detections: list[VideoDetection],
    tracks: tuple[PersonTrack, ...],
) -> np.ndarray:
    """Draw PPE as detections and Person as local track observations."""
    rendered = render_detections(
        frame, [detection for detection in detections if detection.class_id != 2]
    )
    for track in tracks:
        x1, y1, x2, y2 = (round(value) for value in track.bbox_xyxy)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (255, 170, 0), 2)
        label = f"Person #{track.track_id} {track.confidence:.2f}"
        cv2.putText(
            rendered,
            label,
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 170, 0),
            1,
            cv2.LINE_AA,
        )
    return rendered


def build_tracking_metadata(
    tracker: PersonByteTracker | None,
    diagnostics: TrackingDiagnostics | None,
    reported_fps: float | None,
    tracking_latencies: list[float],
) -> dict[str, object]:
    """Build explicit enabled/disabled tracking metadata for a completed run."""
    if tracker is None or diagnostics is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "tracker": tracker.metadata(),
        "diagnostics": diagnostics.summary(reported_fps, tracker.live_track_ids),
        "timing": _summary(tracking_latencies),
    }


def run_video_pipeline(
    config: FrozenDetectorConfig, options: PipelineOptions
) -> dict[str, object]:
    """Run bounded frame-by-frame inference and return persisted run metadata."""
    wall_started = time.perf_counter()
    verified_hash = verify_weights(config.weights_path, config.weights_sha256)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    detections_path = options.output_dir / "detections.jsonl"
    metadata_path = options.output_dir / "run_metadata.json"
    annotated_path = options.output_dir / "annotated.mp4"
    tracks_path = options.output_dir / "tracks.jsonl"

    model_load_started = time.perf_counter()
    model = YOLO(str(config.weights_path))
    model_load_seconds = time.perf_counter() - model_load_started
    selected_device: str | int = (
        options.device
        if options.device is not None
        else (0 if torch.cuda.is_available() else "cpu")
    )

    detector_latencies: list[float] = []
    end_to_end_latencies: list[float] = []
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    tracking_latencies: list[float] = []
    frames_processed = 0
    frames_with_detections = 0
    detections_written = 0
    frames_skipped_by_stride = 0
    warmup_seconds = 0.0
    stopped_by_max_frames = False
    tracker = None
    tracking_diagnostics = None
    if options.enable_tracking:
        if options.tracker_config is None:
            raise ValueError("Tracking configuration was not provided")
        tracker = PersonByteTracker(TrackerSettings.from_yaml(options.tracker_config))
        tracking_diagnostics = TrackingDiagnostics()

    with ExitStack() as stack:
        reader = stack.enter_context(VideoReader(options.source))
        if reader.metadata is None:
            raise RuntimeError("Opened reader did not expose source metadata")
        source_metadata = reader.metadata
        output_fps = source_metadata.reported_fps or options.fallback_fps
        writer = None
        effective_render = options.render or options.enable_tracking
        if effective_render:
            writer = stack.enter_context(
                AnnotatedVideoWriter(
                    annotated_path,
                    source_metadata.original_width,
                    source_metadata.original_height,
                    output_fps,
                )
            )
        detections_file = stack.enter_context(
            detections_path.open("w", encoding="utf-8", newline="\n")
        )
        tracks_file = (
            stack.enter_context(tracks_path.open("w", encoding="utf-8", newline="\n"))
            if options.enable_tracking
            else None
        )

        warmup_frame = np.zeros(
            (
                source_metadata.original_height,
                source_metadata.original_width,
                3,
            ),
            dtype=np.uint8,
        )
        warmup_started = time.perf_counter()
        model.predict(
            source=warmup_frame,
            imgsz=config.imgsz,
            conf=config.inference_confidence_floor,
            iou=config.nms_iou,
            device=selected_device,
            verbose=False,
        )
        warmup_seconds = time.perf_counter() - warmup_started
        processing_started = time.perf_counter()

        while True:
            packet = reader.read()
            if packet is None:
                break
            if packet.frame_index % options.frame_stride != 0:
                frames_skipped_by_stride += 1
                if writer is not None:
                    writer.write(packet.image)
                continue

            frame_started = time.perf_counter()
            detector_started = time.perf_counter()
            results = model.predict(
                source=packet.image,
                imgsz=config.imgsz,
                conf=config.inference_confidence_floor,
                iou=config.nms_iou,
                device=selected_device,
                verbose=False,
            )
            detector_latencies.append(time.perf_counter() - detector_started)
            result = results[0]
            candidates = apply_class_thresholds(
                _candidates(result), config.classes, config.confidence_thresholds
            )
            detections = _structured_detections(
                packet, candidates, config, source_metadata.source_id
            )
            for detection in detections:
                detections_file.write(json.dumps(detection.to_dict()) + "\n")
            frame_tracks: tuple[PersonTrack, ...] = ()
            if tracker is not None and tracking_diagnostics is not None:
                tracking_started = time.perf_counter()
                tracking_result = tracker.update(
                    packet.frame_index,
                    packet.timestamp_seconds,
                    detections,
                    packet.image.shape[:2],
                    source_metadata.source_id,
                )
                tracking_latencies.append(time.perf_counter() - tracking_started)
                tracking_diagnostics.update(tracking_result)
                frame_tracks = tracking_result.tracks
                if tracks_file is None:
                    raise RuntimeError("Tracking output file was not initialized")
                for track in frame_tracks:
                    tracks_file.write(json.dumps(track.to_dict()) + "\n")
            if writer is not None:
                rendered = (
                    render_tracked_detections(packet.image, detections, frame_tracks)
                    if tracker is not None
                    else render_detections(packet.image, detections)
                )
                writer.write(rendered)

            speed = result.speed or {}
            if "preprocess" in speed:
                preprocess_ms.append(float(speed["preprocess"]))
            if "inference" in speed:
                inference_ms.append(float(speed["inference"]))
            if "postprocess" in speed:
                postprocess_ms.append(float(speed["postprocess"]))
            frames_processed += 1
            detections_written += len(detections)
            if detections:
                frames_with_detections += 1
            end_to_end_latencies.append(time.perf_counter() - frame_started)
            if (
                options.max_frames is not None
                and frames_processed >= options.max_frames
            ):
                stopped_by_max_frames = True
                break

        processing_seconds = time.perf_counter() - processing_started
        frames_read = reader.frames_read
        read_failures = reader.read_failures
        reached_end_of_file = reader.reached_end_of_file

    detector_seconds = sum(detector_latencies)
    model_inference_seconds = sum(inference_ms) / 1000
    metadata: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "status": "completed",
        "source": source_metadata.to_dict(),
        "model": {
            "architecture": config.architecture,
            "manifest": str(config.manifest_path),
            "weights": str(config.weights_path),
            "sha256": verified_hash,
            "imgsz": config.imgsz,
            "classes": config.classes,
            "confidence_thresholds": config.confidence_thresholds,
            "ultralytics_confidence_floor": config.inference_confidence_floor,
            "nms_iou": config.nms_iou,
        },
        "processing": {
            "frame_stride": options.frame_stride,
            "frames_read": frames_read,
            "frames_processed": frames_processed,
            "frames_skipped_by_stride": frames_skipped_by_stride,
            "frames_with_detections": frames_with_detections,
            "detections_written": detections_written,
            "read_failures": read_failures,
            "dropped_frames": read_failures,
            "reached_end_of_file": reached_end_of_file,
            "stopped_by_max_frames": stopped_by_max_frames,
            "output_fps": output_fps,
            "fallback_fps_used": source_metadata.reported_fps is None,
            "annotated_video_written": effective_render,
        },
        "timing": {
            "model_load_seconds": model_load_seconds,
            "warmup_frames": 1,
            "warmup_seconds": warmup_seconds,
            "processing_wall_clock_seconds": processing_seconds,
            "total_wall_clock_seconds": time.perf_counter() - wall_started,
            "effective_pipeline_fps": (
                frames_processed / processing_seconds if processing_seconds else 0.0
            ),
            "detector_call_fps": (
                frames_processed / detector_seconds if detector_seconds else 0.0
            ),
            "model_inference_fps": (
                frames_processed / model_inference_seconds
                if model_inference_seconds
                else None
            ),
            "detector_latency": _summary(detector_latencies),
            "end_to_end_frame_latency": _summary(end_to_end_latencies),
            "ultralytics_preprocess": _component_summary(preprocess_ms),
            "ultralytics_inference": _component_summary(inference_ms),
            "ultralytics_postprocess": _component_summary(postprocess_ms),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "tracking": build_tracking_metadata(
            tracker,
            tracking_diagnostics,
            source_metadata.reported_fps,
            tracking_latencies,
        ),
        "outputs": {
            "detections_jsonl": str(detections_path),
            "run_metadata_json": str(metadata_path),
            "tracks_jsonl": str(tracks_path) if options.enable_tracking else None,
            "annotated_video": str(annotated_path) if effective_render else None,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
