"""Bounded OpenCV video ingestion for files, devices, and stream URIs."""

from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

STREAM_SCHEMES = {"rtsp", "rtsps", "http", "https"}


class VideoSourceError(RuntimeError):
    """Base error for malformed or unavailable video sources."""


class VideoSourceOpenError(VideoSourceError):
    """Raised when OpenCV cannot open a parsed source."""


@dataclass(frozen=True)
class SourceSpec:
    """Parsed source passed to OpenCV with a safe identifier for metadata."""

    kind: str
    capture_value: str | int
    source_id: str


@dataclass(frozen=True)
class SourceMetadata:
    """Properties reported when a video source opens."""

    source_id: str
    source_type: str
    original_width: int
    original_height: int
    reported_fps: float | None
    total_frame_count: int | None
    duration_seconds: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FramePacket:
    """One frame read sequentially from a source."""

    frame_index: int
    timestamp_seconds: float
    image: np.ndarray


def _safe_stream_identifier(value: str) -> str:
    parts = urlsplit(value)
    host = parts.hostname or ""
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme.lower(), host, parts.path, parts.query, ""))


def parse_source(value: str) -> SourceSpec:
    """Classify a file, non-negative device index, or supported stream URI."""
    raw = value.strip()
    if not raw:
        raise ValueError("Video source cannot be empty")
    if re.fullmatch(r"\d+", raw):
        index = int(raw)
        return SourceSpec("device", index, f"device:{index}")

    parts = urlsplit(raw)
    if parts.scheme.lower() in STREAM_SCHEMES:
        if not parts.hostname:
            raise ValueError(f"Stream URI has no hostname: {raw}")
        return SourceSpec("stream", raw, _safe_stream_identifier(raw))
    if "://" in raw:
        raise ValueError(f"Unsupported video URI scheme: {parts.scheme or '<missing>'}")

    path = Path(raw).expanduser().resolve()
    return SourceSpec("file", str(path), str(path))


def frame_timestamp_seconds(
    frame_index: int, reported_fps: float | None, elapsed_seconds: float = 0.0
) -> float:
    """Calculate a stable frame timestamp, falling back to elapsed wall time."""
    if frame_index < 0:
        raise ValueError("Frame index cannot be negative")
    if reported_fps is not None and reported_fps > 0 and math.isfinite(reported_fps):
        return frame_index / reported_fps
    return max(0.0, elapsed_seconds)


class VideoReader:
    """Context-managed sequential reader that never buffers the complete video."""

    def __init__(self, source: str) -> None:
        self.spec = parse_source(source)
        self.capture: cv2.VideoCapture | None = None
        self.metadata: SourceMetadata | None = None
        self.frames_read = 0
        self.read_failures = 0
        self.reached_end_of_file = False
        self._started_at = 0.0

    def open(self) -> SourceMetadata:
        self.capture = cv2.VideoCapture(self.spec.capture_value)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise VideoSourceOpenError(
                f"Unable to open {self.spec.kind} video source: {self.spec.source_id}"
            )
        width = int(round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps_value = float(self.capture.get(cv2.CAP_PROP_FPS))
        fps = fps_value if fps_value > 0 and math.isfinite(fps_value) else None
        count_value = float(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        count = (
            int(round(count_value))
            if self.spec.kind == "file"
            and count_value > 0
            and math.isfinite(count_value)
            else None
        )
        duration = count / fps if count is not None and fps is not None else None
        if width <= 0 or height <= 0:
            self.close()
            raise VideoSourceOpenError(
                f"Video source reported invalid dimensions: {width}x{height}"
            )
        self.metadata = SourceMetadata(
            source_id=self.spec.source_id,
            source_type=self.spec.kind,
            original_width=width,
            original_height=height,
            reported_fps=fps,
            total_frame_count=count,
            duration_seconds=duration,
        )
        self._started_at = time.perf_counter()
        return self.metadata

    def read(self) -> FramePacket | None:
        if self.capture is None or self.metadata is None:
            raise VideoSourceError("Video source must be opened before reading")
        ok, frame = self.capture.read()
        if not ok or frame is None:
            if self.spec.kind == "file":
                expected = self.metadata.total_frame_count
                if expected is not None and self.frames_read < expected:
                    self.read_failures += 1
                else:
                    self.reached_end_of_file = True
            else:
                self.read_failures += 1
            return None
        index = self.frames_read
        self.frames_read += 1
        timestamp = frame_timestamp_seconds(
            index,
            self.metadata.reported_fps,
            time.perf_counter() - self._started_at,
        )
        return FramePacket(index, timestamp, frame)

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> VideoReader:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AnnotatedVideoWriter:
    """Context-managed annotated-video writer preserving source dimensions."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("Output video dimensions and FPS must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        codec = "mp4v" if path.suffix.lower() == ".mp4" else "MJPG"
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise VideoSourceOpenError(f"Unable to open output video writer: {path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()

    def __enter__(self) -> AnnotatedVideoWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
