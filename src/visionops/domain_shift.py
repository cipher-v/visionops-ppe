"""Pure, deterministic error decomposition for domain-shift diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

from visionops.detection_analysis import Detection, box_iou

LOCALIZATION_IOU_FLOOR = 0.10
MATCH_IOU = 0.50


@dataclass(frozen=True)
class GroundTruthOutcome:
    """Mutually exclusive diagnostic outcome for one ground-truth object."""

    ground_truth_index: int
    category: str
    prediction_index: int | None = None
    iou: float = 0.0


@dataclass(frozen=True)
class ErrorDecomposition:
    """Ground-truth outcomes and predictions left as false positives."""

    outcomes: tuple[GroundTruthOutcome, ...]
    false_positive_indices: frozenset[int]


def _assign_candidates(
    candidates: list[tuple[float, float, int, int]],
    category: str,
    assigned_ground_truth: set[int],
    assigned_predictions: set[int],
    outcomes: dict[int, GroundTruthOutcome],
) -> None:
    """Greedily assign highest-IoU candidates one-to-one with stable tie breaks."""
    for iou, _confidence, ground_truth_index, prediction_index in sorted(
        candidates,
        key=lambda item: (-item[0], -item[1], item[2], item[3]),
    ):
        if ground_truth_index in assigned_ground_truth:
            continue
        if prediction_index in assigned_predictions:
            continue
        assigned_ground_truth.add(ground_truth_index)
        assigned_predictions.add(prediction_index)
        outcomes[ground_truth_index] = GroundTruthOutcome(
            ground_truth_index,
            category,
            prediction_index,
            iou,
        )


def decompose_errors(
    ground_truth: list[Detection],
    predictions: list[Detection],
    match_iou: float = MATCH_IOU,
    localization_iou_floor: float = LOCALIZATION_IOU_FLOOR,
) -> ErrorDecomposition:
    """Classify each GT as correct, confusion, localization failure, or miss.

    Precedence is correct detection, wrong-class confusion at the match threshold,
    same-class localization error at the lower overlap floor, then complete miss.
    A prediction can be assigned to at most one ground-truth object.
    """
    if not 0 <= localization_iou_floor < match_iou <= 1:
        raise ValueError("Require 0 <= localization floor < match IoU <= 1")

    assigned_ground_truth: set[int] = set()
    assigned_predictions: set[int] = set()
    outcomes: dict[int, GroundTruthOutcome] = {}
    correct: list[tuple[float, float, int, int]] = []
    confusion: list[tuple[float, float, int, int]] = []
    localization: list[tuple[float, float, int, int]] = []

    for ground_truth_index, target in enumerate(ground_truth):
        for prediction_index, prediction in enumerate(predictions):
            iou = box_iou(target.xyxy, prediction.xyxy)
            candidate = (
                iou,
                prediction.confidence,
                ground_truth_index,
                prediction_index,
            )
            if target.class_id == prediction.class_id and iou >= match_iou:
                correct.append(candidate)
            elif target.class_id != prediction.class_id and iou >= match_iou:
                confusion.append(candidate)
            elif (
                target.class_id == prediction.class_id
                and localization_iou_floor <= iou < match_iou
            ):
                localization.append(candidate)

    _assign_candidates(
        correct,
        "correct",
        assigned_ground_truth,
        assigned_predictions,
        outcomes,
    )
    _assign_candidates(
        confusion,
        "classification_confusion",
        assigned_ground_truth,
        assigned_predictions,
        outcomes,
    )
    _assign_candidates(
        localization,
        "localization_failure",
        assigned_ground_truth,
        assigned_predictions,
        outcomes,
    )
    for ground_truth_index in range(len(ground_truth)):
        outcomes.setdefault(
            ground_truth_index,
            GroundTruthOutcome(ground_truth_index, "complete_miss"),
        )
    return ErrorDecomposition(
        outcomes=tuple(outcomes[index] for index in range(len(ground_truth))),
        false_positive_indices=frozenset(
            set(range(len(predictions))) - assigned_predictions
        ),
    )
