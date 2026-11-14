"""Pure box matching and size-binning logic for detection diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

TINY_MAX_AREA = 0.001
SMALL_MAX_AREA = 0.01
SIZE_BINS = ("tiny", "small", "large")


@dataclass(frozen=True)
class Detection:
    """One normalized detection or ground-truth box."""

    class_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0

    @property
    def relative_area(self) -> float:
        """Return normalized box area after clipping negative dimensions to zero."""
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(frozen=True)
class MatchResult:
    """One-to-one class-aware matching result."""

    pairs: tuple[tuple[int, int, float], ...]
    matched_ground_truth_indices: frozenset[int]
    matched_prediction_indices: frozenset[int]


def area_bin(relative_area: float) -> str:
    """Assign a normalized box area to the fixed Checkpoint 3 bins."""
    if relative_area < 0:
        raise ValueError("Relative area cannot be negative")
    if relative_area < TINY_MAX_AREA:
        return "tiny"
    if relative_area < SMALL_MAX_AREA:
        return "small"
    return "large"


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Return intersection over union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


def match_detections(
    ground_truth: list[Detection],
    predictions: list[Detection],
    iou_threshold: float,
) -> MatchResult:
    """Greedily match predictions to same-class ground truth one-to-one.

    Predictions are processed by descending confidence. Each prediction is matched
    to the highest-IoU still-unmatched ground-truth box of the same class.
    """
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be between zero and one")

    unmatched_ground_truth = set(range(len(ground_truth)))
    matched_predictions: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    prediction_order = sorted(
        range(len(predictions)),
        key=lambda index: (-predictions[index].confidence, index),
    )
    for prediction_index in prediction_order:
        prediction = predictions[prediction_index]
        candidates = [
            (box_iou(prediction.xyxy, ground_truth[index].xyxy), index)
            for index in unmatched_ground_truth
            if ground_truth[index].class_id == prediction.class_id
        ]
        if not candidates:
            continue
        best_iou, ground_truth_index = max(
            candidates, key=lambda candidate: (candidate[0], -candidate[1])
        )
        if best_iou < iou_threshold:
            continue
        unmatched_ground_truth.remove(ground_truth_index)
        matched_predictions.add(prediction_index)
        pairs.append((ground_truth_index, prediction_index, best_iou))

    return MatchResult(
        pairs=tuple(pairs),
        matched_ground_truth_indices=frozenset(
            ground_truth_index for ground_truth_index, _, _ in pairs
        ),
        matched_prediction_indices=frozenset(matched_predictions),
    )
