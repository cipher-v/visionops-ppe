"""API response contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class DetectionResponse(BaseModel):
    class_name: str = Field(alias="class")
    confidence: float
    bbox_xyxy: list[float]
    bbox_normalized: list[float]


class ImageInferenceResponse(BaseModel):
    width: int
    height: int
    processing_latency_ms: float
    detections: list[DetectionResponse]


class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: float
    error: str | None = None


class VideoResultResponse(BaseModel):
    job_id: str
    summary: dict[str, Any]
