"""Repository-owned class-specific confidence filtering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateBox:
    """Minimal model output needed before structured serialization."""

    class_id: int
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


def passes_class_threshold(
    class_id: int,
    confidence: float,
    classes: dict[int, str],
    thresholds: dict[str, float],
) -> bool:
    """Return whether a candidate meets its frozen class threshold."""
    if class_id not in classes:
        raise ValueError(f"Unknown detector class ID: {class_id}")
    class_name = classes[class_id]
    if class_name not in thresholds:
        raise ValueError(f"No confidence threshold configured for {class_name}")
    return confidence >= thresholds[class_name]


def apply_class_thresholds(
    candidates: list[CandidateBox],
    classes: dict[int, str],
    thresholds: dict[str, float],
) -> list[CandidateBox]:
    """Filter model candidates without replacing the frozen per-class policy."""
    return [
        candidate
        for candidate in candidates
        if passes_class_threshold(
            candidate.class_id, candidate.confidence, classes, thresholds
        )
    ]
