"""VisionOps PPE FastAPI application factory."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from visionops.api.errors import ServiceError
from visionops.api.health import readiness
from visionops.api.metrics import ServiceMetrics
from visionops.api.runtime import ProductionRuntime, ServiceSettings
from visionops.api.schemas import (
    ImageInferenceResponse,
    JobStatusResponse,
    VideoResultResponse,
)

ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger("visionops.api")


def _log(event: str, **values: object) -> None:
    LOGGER.info(json.dumps({"event": event, **values}, default=str))


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    content = bytearray()
    while chunk := await upload.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > limit:
            raise ServiceError(413, "upload_too_large", f"Upload exceeds {limit} bytes")
    if not content:
        raise ServiceError(400, "empty_upload", "Upload is empty")
    return bytes(content)


def create_app(
    runtime_factory: Callable[
        [ServiceSettings, ServiceMetrics], Any
    ] = ProductionRuntime,
    settings: ServiceSettings | None = None,
) -> FastAPI:
    service_metrics = ServiceMetrics()
    effective_settings = settings or ServiceSettings.from_environment(ROOT)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.runtime = runtime_factory(effective_settings, service_metrics)
        _log("runtime_startup", status="ready", backend="pytorch")
        try:
            yield
        finally:
            application.state.runtime.close()

    application = FastAPI(title="VisionOps PPE", version="1.0", lifespan=lifespan)
    application.state.metrics = service_metrics

    @application.exception_handler(ServiceError)
    async def service_error_handler(
        _request: Request, error: ServiceError
    ) -> JSONResponse:
        service_metrics.failures.inc()
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "message": error.message}},
        )

    @application.exception_handler(Exception)
    async def internal_error_handler(
        _request: Request, error: Exception
    ) -> JSONResponse:
        LOGGER.exception("Unhandled API error", exc_info=error)
        service_metrics.failures.inc()
        return JSONResponse(
            status_code=500,
            content={
                "error": {"code": "internal_error", "message": "Internal service error"}
            },
        )

    @application.middleware("http")
    async def observe_request(
        request: Request, call_next: Callable[..., Any]
    ) -> Response:
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        route_name = getattr(route, "path", request.url.path)
        service_metrics.requests.labels(
            method=request.method, route=route_name, status=str(response.status_code)
        ).inc()
        _log(
            "http_request",
            request_type=route_name,
            status=response.status_code,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )
        return response

    @application.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/health/ready")
    def ready(request: Request) -> dict[str, object]:
        return readiness(request.app.state.runtime)

    @application.get("/v1/config")
    def safe_config(request: Request) -> dict[str, object]:
        runtime = request.app.state.runtime
        config = runtime.config
        return {
            "detector_backend": "pytorch",
            "model_input_size": config.imgsz,
            "classes": config.classes,
            "confidence_thresholds": config.confidence_thresholds,
            "nms_iou": config.nms_iou,
            "tracker": "Ultralytics BoT-SORT",
            "association_enabled": True,
            "temporal_policy_enabled": True,
            "zone_support": True,
            "drift": {
                "available": runtime.drift_reference is not None,
                "diagnostic_only": True,
                "accuracy_failure_predictor": False,
            },
        }

    @application.post("/v1/infer/image", response_model=ImageInferenceResponse)
    async def infer_image(
        request: Request, file: UploadFile = File(...)
    ) -> dict[str, object]:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            raise ServiceError(
                415, "unsupported_image_type", "Unsupported image extension"
            )
        content = await _read_limited(file, effective_settings.max_image_bytes)
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ServiceError(422, "invalid_image", "Image could not be decoded")
        return request.app.state.runtime.infer_image(image)

    @application.post(
        "/v1/jobs/video", response_model=JobStatusResponse, status_code=202
    )
    async def submit_video(
        request: Request,
        file: UploadFile = File(...),
        zone_config: UploadFile | None = File(default=None),
    ) -> dict[str, object]:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
            raise ServiceError(
                415, "unsupported_video_type", "Unsupported video extension"
            )
        content = await _read_limited(file, effective_settings.max_video_bytes)
        incoming = effective_settings.runtime_dir / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        input_path = incoming / f"{token}{suffix}"
        input_path.write_bytes(content)
        capture = cv2.VideoCapture(str(input_path))
        valid = capture.isOpened() and capture.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        capture.release()
        if not valid:
            input_path.unlink(missing_ok=True)
            raise ServiceError(422, "invalid_video", "Video could not be decoded")
        zone_path = None
        if zone_config is not None:
            zone_content = await _read_limited(zone_config, 1024 * 1024)
            zone_path = request.app.state.runtime.validate_zone(zone_content, incoming)
        job = request.app.state.runtime.submit_video(input_path, zone_path)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "error": None,
        }

    @application.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
    def job_status(job_id: str, request: Request) -> dict[str, object]:
        job = request.app.state.runtime.get_job(job_id)
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "error": job.error,
        }

    @application.get("/v1/jobs/{job_id}/result", response_model=VideoResultResponse)
    def job_result(job_id: str, request: Request) -> dict[str, object]:
        job = request.app.state.runtime.get_job(job_id)
        if job.status != "completed" or job.result is None:
            raise ServiceError(409, "job_not_complete", "Video job result is not ready")
        return {"job_id": job.job_id, "summary": job.result}

    @application.get("/v1/jobs/{job_id}/incidents")
    def job_incidents(job_id: str, request: Request) -> dict[str, object]:
        job = request.app.state.runtime.get_job(job_id)
        if job.status != "completed":
            raise ServiceError(409, "job_not_complete", "Video job result is not ready")
        return {
            "job_id": job.job_id,
            "track_id_semantics": "anonymous run-local identifier, not worker identity",
            "incidents": job.incidents,
        }

    @application.get("/metrics")
    def metrics() -> Response:
        return Response(
            generate_latest(service_metrics.registry), media_type=CONTENT_TYPE_LATEST
        )

    return application


app = create_app()
