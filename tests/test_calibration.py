from pathlib import Path

import pytest
import yaml

from visionops.calibration import (
    OperatingCounts,
    apply_threshold_policy,
    class_policy_is_material,
    domain_balanced_score,
    select_threshold,
    threshold_grid,
    validate_production_config,
)
from visionops.detection_analysis import Detection


def test_threshold_grid_is_exact_and_inclusive() -> None:
    values = threshold_grid()
    assert len(values) == 86
    assert values[0] == 0.05
    assert values[-1] == 0.90
    assert values[1] == 0.06


def test_f1_f2_and_domain_balancing() -> None:
    counts = OperatingCounts(tp=8, fp=2, fn=4)
    assert counts.precision == pytest.approx(0.8)
    assert counts.recall == pytest.approx(2 / 3)
    assert counts.f_beta(1) == pytest.approx(8 / 11)
    assert counts.f_beta(2) == pytest.approx(20 / 29)
    assert domain_balanced_score(counts, OperatingCounts(4, 1, 1)) == pytest.approx(
        (8 / 11 + 0.8) / 2
    )


def test_threshold_selection_tie_prefers_precision_then_higher_threshold() -> None:
    first = OperatingCounts(8, 2, 2)
    candidates = [(0.20, first, first), (0.30, first, first)]
    assert select_threshold(candidates)["threshold"] == 0.30


def test_class_specific_policy_application_and_gate() -> None:
    predictions = [
        Detection(0, (0.1, 0.1, 0.2, 0.2), 0.4),
        Detection(3, (0.2, 0.2, 0.3, 0.3), 0.2),
    ]
    retained = apply_threshold_policy(predictions, 0.3, {3: 0.15})
    assert [item.class_id for item in retained] == [0, 3]
    decision = class_policy_is_material(
        (OperatingCounts(80, 20, 20), OperatingCounts(40, 10, 10)),
        (OperatingCounts(85, 21, 15), OperatingCounts(44, 11, 6)),
        {3: 0.60, 4: 0.70},
        {3: 0.66, 4: 0.71},
    )
    assert decision["adopt_class_specific"] is True


def test_final_config_contract_and_test_guard() -> None:
    values = {
        "version": 1,
        "model": {"weights": "runs/final/best.pt", "sha256": "abc"},
        "input": {"imgsz": 960},
        "classes": {
            0: "helmet",
            1: "vest",
            2: "Person",
            3: "no_helmet",
            4: "no_vest",
        },
        "nms_iou": 0.70,
        "confidence_policy": {"type": "global", "global_threshold": 0.3},
        "preprocessing": {"color_space": "RGB"},
    }
    validate_production_config(values)
    invalid = {**values, "nms_iou": 0.5}
    with pytest.raises(ValueError, match="NMS"):
        validate_production_config(invalid)


def test_generated_production_config_parses() -> None:
    config_path = Path(__file__).parents[1] / "configs/production_detector.yaml"
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_production_config(values)
    assert values["calibration"]["rf100_test_used"] is False
