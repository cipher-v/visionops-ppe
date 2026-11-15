"""Streaming inference pipeline for the frozen production detector."""

from __future__ import annotations

import json
import platform
import statistics
import time
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from visionops.association.associator import PPEAssociator
from visionops.association.config import AssociationConfig
from visionops.association.metrics import AssociationDiagnostics
from visionops.association.schema import PersonPPEFrame
from visionops.compliance.config import TemporalComplianceConfig
from visionops.compliance.engine import TemporalComplianceEngine
from visionops.compliance.schema import TemporalComplianceRecord
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
    enable_ppe_association: bool = False
    ppe_association_config: Path | None = None
    enable_temporal_compliance: bool = False
    temporal_compliance_config: Path | None = None

    def __post_init__(self) -> None:
        if self.frame_stride < 1:
            raise ValueError("Frame stride must be at least 1")
        if self.fallback_fps <= 0:
            raise ValueError("Fallback FPS must be positive")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("Maximum processed frames must be positive")
        if self.enable_tracking and self.tracker_config is None:
            raise ValueError("Tracking requires a tracker configuration path")
        if self.enable_ppe_association and not self.enable_tracking:
            raise ValueError("PPE association requires tracking to be enabled")
        if self.enable_ppe_association and self.ppe_association_config is None:
            raise ValueError("PPE association requires an association configuration")
        if self.enable_temporal_compliance and not self.enable_tracking:
            raise ValueError("Temporal compliance requires tracking to be enabled")
        if self.enable_temporal_compliance and not self.enable_ppe_association:
            raise ValueError(
                "Temporal compliance requires PPE association to be enabled"
            )
        if self.enable_temporal_compliance and self.temporal_compliance_config is None:
            raise ValueError("Temporal compliance requires a temporal configuration")


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


def render_ppe_associations(
    frame: np.ndarray,
    detections: list[VideoDetection],
    tracks: tuple[PersonTrack, ...],
    records: tuple[PersonPPEFrame, ...],
) -> np.ndarray:
    """Add frame-level H/V evidence states without compliance conclusions."""
    rendered = render_tracked_detections(frame, detections, tracks)
    record_by_track = {record.track_id: record for record in records}
    for track in tracks:
        record = record_by_track.get(track.track_id)
        if record is None:
            continue
        x1, y1, _, _ = (round(value) for value in track.bbox_xyxy)
        cv2.putText(
            rendered,
            f"H: {record.helmet.state}",
            (x1, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 170, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            f"V: {record.vest.state}",
            (x1, y1 + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 170, 0),
            1,
            cv2.LINE_AA,
        )
    return rendered


def _temporal_label(state: object) -> str:
    if state.temporal_state == "compliant":
        return "OK"
    if state.temporal_state == "suspected_violation":
        return f"SUSPECT {state.continuous_violation_duration:.1f}s"
    if state.temporal_state == "confirmed_violation":
        duration = (
            state.incident_duration_seconds
            if state.incident_duration_seconds is not None
            else state.continuous_violation_duration
        )
        return f"VIOLATION {duration:.1f}s"
    return state.temporal_state.upper()


def render_temporal_compliance(
    frame: np.ndarray,
    detections: list[VideoDetection],
    tracks: tuple[PersonTrack, ...],
    records: tuple[TemporalComplianceRecord, ...],
) -> np.ndarray:
    """Render explicit temporal evidence without zones or alert semantics."""
    rendered = render_tracked_detections(frame, detections, tracks)
    record_by_track = {record.track_id: record for record in records}
    for track in tracks:
        record = record_by_track.get(track.track_id)
        if record is None:
            continue
        x1, y1, _, _ = (round(value) for value in track.bbox_xyxy)
        cv2.putText(
            rendered,
            f"H: {_temporal_label(record.helmet)}",
            (x1, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 170, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            f"V: {_temporal_label(record.vest)}",
            (x1, y1 + 36),
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


def build_association_metadata(
    associator: PPEAssociator | None,
    diagnostics: AssociationDiagnostics | None,
    association_latencies: list[float],
) -> dict[str, object]:
    """Build explicit enabled/disabled association metadata."""
    if associator is None or diagnostics is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "associator": associator.metadata(),
        "diagnostics": diagnostics.summary(),
        "timing": _summary(association_latencies),
    }


def build_temporal_metadata(
    engine: TemporalComplianceEngine | None,
    temporal_latencies: list[float],
) -> dict[str, object]:
    """Build temporal-engine metadata with an explicit real-video validation flag."""
    if engine is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "engine": engine.metadata(),
        "diagnostics": engine.summary(),
        "timing": _summary(temporal_latencies),
        "real_video_temporal_validation": False,
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
    associations_path = options.output_dir / "ppe_associations.jsonl"
    compliance_path = options.output_dir / "compliance_states.jsonl"
    incidents_path = options.output_dir / "incidents.jsonl"

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
    association_latencies: list[float] = []
    temporal_latencies: list[float] = []
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
    associator = None
    association_diagnostics = None
    if options.enable_ppe_association:
        if options.ppe_association_config is None:
            raise ValueError("PPE association configuration was not provided")
        associator = PPEAssociator(
            AssociationConfig.from_yaml(options.ppe_association_config)
        )
        association_diagnostics = AssociationDiagnostics()
    temporal_engine = None
    if options.enable_temporal_compliance:
        if options.temporal_compliance_config is None:
            raise ValueError("Temporal compliance configuration was not provided")
        temporal_engine = TemporalComplianceEngine(
            TemporalComplianceConfig.from_yaml(options.temporal_compliance_config),
            run_id=f"run-{uuid.uuid4().hex}",
        )

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
        associations_file = (
            stack.enter_context(
                associations_path.open("w", encoding="utf-8", newline="\n")
            )
            if options.enable_ppe_association
            else None
        )
        compliance_file = (
            stack.enter_context(
                compliance_path.open("w", encoding="utf-8", newline="\n")
            )
            if options.enable_temporal_compliance
            else None
        )
        incidents_file = (
            stack.enter_context(
                incidents_path.open("w", encoding="utf-8", newline="\n")
            )
            if options.enable_temporal_compliance
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
        last_processed_timestamp = 0.0

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
            last_processed_timestamp = packet.timestamp_seconds
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
            tracking_result = None
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
            association_records: tuple[PersonPPEFrame, ...] = ()
            if associator is not None and association_diagnostics is not None:
                association_started = time.perf_counter()
                association_result = associator.associate(frame_tracks, detections)
                association_latencies.append(time.perf_counter() - association_started)
                association_diagnostics.update(association_result)
                association_records = association_result.records
                if associations_file is None:
                    raise RuntimeError(
                        "PPE association output file was not initialized"
                    )
                for record in association_records:
                    associations_file.write(json.dumps(record.to_dict()) + "\n")
            temporal_records: tuple[TemporalComplianceRecord, ...] = ()
            if temporal_engine is not None:
                temporal_started = time.perf_counter()
                if compliance_file is None or incidents_file is None:
                    raise RuntimeError("Temporal compliance files were not initialized")
                active_ids = {record.track_id for record in association_records}
                expired_ids = (
                    set(tracking_result.events.expired_ids)
                    if tracking_result is not None
                    else set()
                )
                for track_id in sorted(expired_ids):
                    for event in temporal_engine.expire_track(
                        track_id, packet.timestamp_seconds
                    ):
                        incidents_file.write(json.dumps(event.to_dict()) + "\n")
                live_ids = set(tracker.live_track_ids) if tracker is not None else set()
                missing_ids = (
                    set(temporal_engine.known_track_ids) - active_ids - expired_ids
                ) & live_ids
                for track_id in sorted(missing_ids):
                    for event in temporal_engine.mark_missing(
                        track_id, packet.timestamp_seconds
                    ):
                        incidents_file.write(json.dumps(event.to_dict()) + "\n")
                produced_records = []
                for association_record in association_records:
                    update = temporal_engine.update(association_record)
                    produced_records.append(update.record)
                    compliance_file.write(json.dumps(update.record.to_dict()) + "\n")
                    for event in update.events:
                        incidents_file.write(json.dumps(event.to_dict()) + "\n")
                temporal_records = tuple(produced_records)
                temporal_latencies.append(time.perf_counter() - temporal_started)
            if writer is not None:
                rendered = (
                    render_temporal_compliance(
                        packet.image,
                        detections,
                        frame_tracks,
                        temporal_records,
                    )
                    if temporal_engine is not None
                    else render_ppe_associations(
                        packet.image,
                        detections,
                        frame_tracks,
                        association_records,
                    )
                    if associator is not None
                    else render_tracked_detections(
                        packet.image, detections, frame_tracks
                    )
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

        if temporal_engine is not None:
            if incidents_file is None:
                raise RuntimeError("Temporal incident file was not initialized")
            for event in temporal_engine.finalize(last_processed_timestamp):
                incidents_file.write(json.dumps(event.to_dict()) + "\n")

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
        "ppe_association": build_association_metadata(
            associator,
            association_diagnostics,
            association_latencies,
        ),
        "temporal_compliance": build_temporal_metadata(
            temporal_engine,
            temporal_latencies,
        ),
        "outputs": {
            "detections_jsonl": str(detections_path),
            "run_metadata_json": str(metadata_path),
            "tracks_jsonl": str(tracks_path) if options.enable_tracking else None,
            "ppe_associations_jsonl": (
                str(associations_path) if options.enable_ppe_association else None
            ),
            "compliance_states_jsonl": (
                str(compliance_path) if options.enable_temporal_compliance else None
            ),
            "incidents_jsonl": (
                str(incidents_path) if options.enable_temporal_compliance else None
            ),
            "annotated_video": str(annotated_path) if effective_render else None,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
