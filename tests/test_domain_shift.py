import pytest

from visionops.detection_analysis import Detection
from visionops.domain_shift import decompose_errors


def detection(
    class_id: int,
    xyxy: tuple[float, float, float, float],
    confidence: float = 1.0,
) -> Detection:
    return Detection(class_id, xyxy, confidence)


def test_error_categories_are_mutually_exclusive() -> None:
    ground_truth = [
        detection(0, (0.0, 0.0, 0.4, 0.4)),
        detection(2, (0.5, 0.0, 0.9, 0.4)),
        detection(5, (0.0, 0.5, 0.4, 0.9)),
        detection(6, (0.5, 0.5, 0.9, 0.9)),
    ]
    predictions = [
        detection(0, (0.0, 0.0, 0.4, 0.4), 0.9),
        detection(11, (0.5, 0.0, 0.9, 0.4), 0.8),
        detection(5, (0.0, 0.5, 0.15, 0.9), 0.7),
    ]
    result = decompose_errors(ground_truth, predictions)
    assert [outcome.category for outcome in result.outcomes] == [
        "correct",
        "classification_confusion",
        "localization_failure",
        "complete_miss",
    ]
    assert result.false_positive_indices == frozenset()


def test_prediction_cannot_match_multiple_ground_truth_objects() -> None:
    ground_truth = [
        detection(0, (0.0, 0.0, 0.5, 0.5)),
        detection(0, (0.1, 0.1, 0.6, 0.6)),
    ]
    predictions = [detection(0, (0.0, 0.0, 0.5, 0.5), 0.9)]
    result = decompose_errors(ground_truth, predictions)
    assert sum(outcome.category == "correct" for outcome in result.outcomes) == 1
    assert sum(outcome.category == "complete_miss" for outcome in result.outcomes) == 1


def test_unassigned_prediction_is_false_positive() -> None:
    result = decompose_errors([], [detection(0, (0.0, 0.0, 0.2, 0.2), 0.4)])
    assert result.false_positive_indices == frozenset({0})


def test_invalid_overlap_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError):
        decompose_errors([], [], match_iou=0.5, localization_iou_floor=0.5)
