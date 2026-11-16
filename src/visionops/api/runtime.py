"""One-time model lifecycle and bounded local video job execution."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from visionops.api.errors import ServiceError
from visionops.api.media import (
    FFmpegTools,
    finalize_browser_media,
    resolve_ffmpeg_tools,
)
from visionops.api.metrics import ServiceMetrics
from visionops.association.config import AssociationConfig
from visionops.compliance.config import TemporalComplianceConfig
from visionops.drift.monitor import DriftReference
from visionops.tracking.tracker import TrackerSettings
from visionops.video.backend import DetectorBackend
from visionops.video.config import load_frozen_detector_config, verify_weights
from visionops.video.pipeline import PipelineOptions, run_video_pipeline
from visionops.video.policy import apply_class_thresholds
from visionops.zones.config import ZoneConfig

LOGGER = logging.getLogger("visionops.api")


@dataclass(frozen=True)
class ServiceSettings:
    root: Path
    runtime_dir: Path
    max_image_bytes: int = 10 * 1024 * 1024
    max_video_bytes: int = 250 * 1024 * 1024
    image_concurrency: int = 1
    video_workers: int = 1

    @classmethod
    def from_environment(cls, root: Path) -> ServiceSettings:
        return cls(
            root=root,
            runtime_dir=Path(
                os.getenv("VISIONOPS_RUNTIME_DIR", root / "results/service_runtime")
            ),
            max_image_bytes=int(
                os.getenv("VISIONOPS_MAX_IMAGE_BYTES", 10 * 1024 * 1024)
            ),
            max_video_bytes=int(
                os.getenv("VISIONOPS_MAX_VIDEO_BYTES", 250 * 1024 * 1024)
            ),
            image_concurrency=int(os.getenv("VISIONOPS_IMAGE_CONCURRENCY", "1")),
            video_workers=int(os.getenv("VISIONOPS_VIDEO_WORKERS", "1")),
        )


@dataclass
class VideoJob:
    job_id: str
    input_path: Path
    output_dir: Path
    zone_path: Path | None = None
    presentation_source_path: Path | None = None
    presentation_original_path: Path | None = None
    presentation_annotated_path: Path | None = None
    status: str = "queued"
    progress: float = 0.0
    error: str | None = None
    result: dict[str, Any] | None = None
    incidents: list[dict[str, Any]] = field(default_factory=list)


class ProductionRuntime:
    def __init__(self, settings: ServiceSettings, metrics: ServiceMetrics) -> None:
        if settings.image_concurrency < 1 or settings.video_workers != 1:
            raise ValueError(
                "Image concurrency must be positive and video workers must equal one"
            )
        self.settings = settings
        self.metrics = metrics
        self.device: str | int = 0 if torch.cuda.is_available() else "cpu"
        root = settings.root
        self.config = load_frozen_detector_config(
            root / "configs/production_detector.yaml", root
        )
        verify_weights(self.config.weights_path, self.config.weights_sha256)
        TrackerSettings.from_yaml(root / "configs/production_tracker.yaml")
        AssociationConfig.from_yaml(root / "configs/ppe_association.yaml")
        TemporalComplianceConfig.from_yaml(root / "configs/compliance_temporal.yaml")
        drift_path = root / "configs/drift_reference.json"
        self.drift_reference = (
            DriftReference.load(drift_path) if drift_path.is_file() else None
        )
        self.detector = DetectorBackend("pytorch", self.config)
        self.ffmpeg_tools: FFmpegTools = resolve_ffmpeg_tools()
        self.gpu_lock = threading.BoundedSemaphore(settings.image_concurrency)
        self.jobs_lock = threading.Lock()
        self.jobs: dict[str, VideoJob] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="visionops-video"
        )
        settings.runtime_dir.mkdir(parents=True, exist_ok=True)
        warmup = np.zeros((640, 640, 3), dtype=np.uint8)
        with self.gpu_lock:
            self.detector.predict(
                warmup,
                imgsz=self.config.imgsz,
                confidence=self.config.inference_confidence_floor,
                iou=self.config.nms_iou,
                device=self.device,
            )

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)

    def infer_image(self, image: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        with self.gpu_lock:
            result = self.detector.predict(
                image,
                imgsz=self.config.imgsz,
                confidence=self.config.inference_confidence_floor,
                iou=self.config.nms_iou,
                device=self.device,
            )
        detector_seconds = time.perf_counter() - started
        self.metrics.detector_latency.observe(detector_seconds)
        self.metrics.images.inc()
        boxes = result.boxes
        detections = []
        if boxes is not None:
            from visionops.video.pipeline import _candidates

            candidates = apply_class_thresholds(
                _candidates(result),
                self.config.classes,
                self.config.confidence_thresholds,
            )
            height, width = image.shape[:2]
            for item in candidates:
                box = [float(value) for value in item.bbox_xyxy]
                detections.append(
                    {
                        "class": self.config.classes[item.class_id],
                        "confidence": item.confidence,
                        "bbox_xyxy": box,
                        "bbox_normalized": [
                            box[0] / width,
                            box[1] / height,
                            box[2] / width,
                            box[3] / height,
                        ],
                    }
                )
        height, width = image.shape[:2]
        return {
            "width": width,
            "height": height,
            "processing_latency_ms": detector_seconds * 1000,
            "detections": detections,
        }

    def validate_zone(self, content: bytes, job_dir: Path) -> Path:
        path = job_dir / f"zones-{uuid.uuid4().hex}.yaml"
        path.write_bytes(content)
        try:
            ZoneConfig.from_yaml(path)
        except Exception as error:
            path.unlink(missing_ok=True)
            raise ServiceError(
                422, "malformed_zone_config", "Zone configuration is invalid"
            ) from error
        return path

    def submit_video(
        self,
        input_path: Path,
        zone_path: Path | None,
        presentation_path: Path | None = None,
    ) -> VideoJob:
        job_id = uuid.uuid4().hex
        output_dir = self.settings.runtime_dir / "jobs" / job_id
        output_dir.mkdir(parents=True)
        destination = output_dir / f"input{input_path.suffix.lower()}"
        input_path.replace(destination)
        if zone_path is not None:
            target_zone = output_dir / "zones.yaml"
            zone_path.replace(target_zone)
            zone_path = target_zone
        presentation_source = None
        if presentation_path is not None:
            presentation_dir = output_dir / "presentation"
            presentation_dir.mkdir(parents=True, exist_ok=True)
            presentation_source = (
                presentation_dir / f"source{presentation_path.suffix.lower()}"
            )
            presentation_path.replace(presentation_source)
        job = VideoJob(
            job_id,
            destination,
            output_dir,
            zone_path,
            presentation_source_path=presentation_source,
        )
        with self.jobs_lock:
            self.jobs[job_id] = job
        self.metrics.jobs_submitted.inc()
        self.executor.submit(self._run_job, job_id)
        return job

    def _run_job(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.status, job.progress = "running", 0.05
        self.metrics.active_jobs.inc()
        started = time.perf_counter()
        try:
            with self.gpu_lock:
                metadata = run_video_pipeline(
                    self.config,
                    PipelineOptions(
                        source=str(job.input_path),
                        output_dir=job.output_dir / "artifacts",
                        render=True,
                        overlay_mode="demo",
                        device=self.device,
                        enable_tracking=True,
                        tracker_config=self.settings.root
                        / "configs/production_tracker.yaml",
                        enable_ppe_association=True,
                        ppe_association_config=self.settings.root
                        / "configs/ppe_association.yaml",
                        enable_temporal_compliance=True,
                        temporal_compliance_config=self.settings.root
                        / "configs/compliance_temporal.yaml",
                        zones_config=job.zone_path,
                    ),
                    detector=self.detector,
                    warmup=False,
                )
            if job.presentation_source_path is None:
                raise RuntimeError("Original presentation source is unavailable")
            presentation_dir = job.output_dir / "presentation"
            job.presentation_original_path = presentation_dir / "original.mp4"
            job.presentation_annotated_path = presentation_dir / "annotated.mp4"
            finalize_browser_media(
                job.presentation_source_path,
                job.presentation_original_path,
                self.ffmpeg_tools,
            )
            finalize_browser_media(
                job.output_dir / "artifacts/annotated.mp4",
                job.presentation_annotated_path,
                self.ffmpeg_tools,
            )
            incidents_path = job.output_dir / "artifacts/incidents.jsonl"
            job.incidents = (
                [
                    json.loads(line)
                    for line in incidents_path.read_text(encoding="utf-8").splitlines()
                ]
                if incidents_path.is_file()
                else []
            )
            p = metadata["processing"]
            job.result = {
                "frames_processed": p["frames_processed"],
                "duration_seconds": metadata["source"]["duration_seconds"],
                "effective_fps": metadata["timing"]["effective_pipeline_fps"],
                "detector_counts": {
                    "detections": p["detections_written"],
                    "frames_with_detections": p["frames_with_detections"],
                },
                "unique_person_tracks": metadata["tracking"]
                .get("diagnostics", {})
                .get("unique_tracks_created", 0),
                "association_counts": metadata["ppe_association"].get(
                    "diagnostics", {}
                ),
                "temporal_state_counts": metadata["temporal_compliance"]
                .get("diagnostics", {})
                .get("temporal_state_observations", {}),
                "incident_count": len(job.incidents),
                "zone_statistics": metadata["zones"].get("diagnostics")
                if metadata["zones"].get("enabled")
                else None,
                "drift": {
                    "diagnostic_only": True,
                    "accuracy_failure_predictor": False,
                    "available": False,
                },
                "warnings": [
                    "Local single-node execution; artifacts have local retention."
                ],
                "artifacts": sorted(
                    path.name for path in (job.output_dir / "artifacts").iterdir()
                ),
            }
            job.status, job.progress = "completed", 1.0
            self.metrics.jobs_completed.labels(status="completed").inc()
            self.metrics.incidents.inc(len(job.incidents))
        except Exception:
            LOGGER.exception(
                json.dumps(
                    {"event": "video_job", "status": "failed", "job_type": "video"}
                )
            )
            job.status, job.error = (
                "failed",
                "Video processing or presentation finalization failed",
            )
            self.metrics.jobs_completed.labels(status="failed").inc()
        finally:
            self.metrics.active_jobs.dec()
            self.metrics.pipeline_latency.observe(time.perf_counter() - started)
            job.input_path.unlink(missing_ok=True)

    def get_job(self, job_id: str) -> VideoJob:
        try:
            return self.jobs[job_id]
        except KeyError as error:
            raise ServiceError(
                404, "job_not_found", "Video job does not exist"
            ) from error
