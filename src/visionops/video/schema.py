"""Structured video detection schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class VideoDetection:
    """One detector result in original-frame and normalized coordinates."""

    frame_index: int
    timestamp_seconds: float
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_normalized: tuple[float, float, float, float]
    source_id: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping with array-form coordinates."""
        values = asdict(self)
        values["bbox_xyxy"] = list(self.bbox_xyxy)
        values["bbox_normalized"] = list(self.bbox_normalized)
        return values


def normalized_box(
    bbox_xyxy: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Clip an original-frame box and normalize it by frame dimensions."""
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    x1, y1, x2, y2 = bbox_xyxy
    clipped = (
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    )
    return (
        clipped[0] / width,
        clipped[1] / height,
        clipped[2] / width,
        clipped[3] / height,
    )
