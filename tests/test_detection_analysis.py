import pytest

from visionops.detection_analysis import (
    Detection,
    area_bin,
    box_iou,
    match_detections,
)


@pytest.mark.parametrize(
    ("area", "expected"),
    [
        (0.0, "tiny"),
        (0.000999, "tiny"),
        (0.001, "small"),
        (0.009999, "small"),
        (0.01, "large"),
    ],
)
def test_area_bin_boundaries(area: float, expected: str) -> None:
    assert area_bin(area) == expected


def test_box_iou_handles_identical_and_disjoint_boxes() -> None:
    box = (0.1, 0.1, 0.4, 0.4)

    assert box_iou(box, box) == pytest.approx(1.0)
    assert box_iou(box, (0.6, 0.6, 0.8, 0.8)) == 0.0


def test_matching_is_class_aware_and_one_to_one() -> None:
    ground_truth = [
        Detection(0, (0.0, 0.0, 0.2, 0.2)),
        Detection(0, (0.05, 0.05, 0.25, 0.25)),
        Detection(1, (0.0, 0.0, 0.2, 0.2)),
    ]
    predictions = [
        Detection(0, (0.0, 0.0, 0.2, 0.2), 0.9),
        Detection(0, (0.0, 0.0, 0.2, 0.2), 0.8),
        Detection(0, (0.0, 0.0, 0.2, 0.2), 0.7),
    ]

    result = match_detections(ground_truth, predictions, iou_threshold=0.3)

    assert result.matched_ground_truth_indices == frozenset({0, 1})
    assert result.matched_prediction_indices == frozenset({0, 1})
    assert len(result.pairs) == 2


def test_higher_confidence_prediction_claims_match_first() -> None:
    ground_truth = [Detection(0, (0.0, 0.0, 0.2, 0.2))]
    predictions = [
        Detection(0, (0.0, 0.0, 0.2, 0.2), 0.5),
        Detection(0, (0.0, 0.0, 0.2, 0.2), 0.9),
    ]

    result = match_detections(ground_truth, predictions, iou_threshold=0.5)

    assert result.matched_prediction_indices == frozenset({1})
