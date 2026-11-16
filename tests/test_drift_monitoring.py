import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from visionops.drift.features import image_features, output_features
from visionops.drift.monitor import (
    DriftReference,
    DriftWindowRecord,
    aggregate_input_features,
    classify_drift,
    fixed_time_windows,
    population_stability_index,
    score_drift,
)
from visionops.drift.perturbations import apply_perturbation


def _gradient() -> np.ndarray:
    values = np.tile(np.arange(64, dtype=np.uint8), (64, 1)) * 4
    return cv2.merge((values, values, values))


def _reference(features: dict[str, object]) -> DriftReference:
    profile = aggregate_input_features([features, features])
    return DriftReference(
        version=1,
        sample_count=2,
        input_profile=profile,
        output_profile={"class_frequency_distribution": [1, 1, 1, 0.1, 0.1]},
        alert_thresholds={"warning": 0.1, "high": 0.25},
    )


def test_perturbations_are_deterministic_and_do_not_mutate() -> None:
    image = _gradient()
    original = image.copy()
    for name in (
        "brightness_decrease",
        "brightness_increase",
        "contrast_reduction",
        "gaussian_blur",
        "motion_blur",
        "gaussian_noise",
        "jpeg_compression",
        "saturation_reduction",
    ):
        first = apply_perturbation(image, name, "moderate", seed=7)
        second = apply_perturbation(image, name, "moderate", seed=7)
        assert np.array_equal(first, second)
        assert first.shape == image.shape and first.dtype == np.uint8
    assert np.array_equal(image, original)


def test_image_metrics_and_histograms() -> None:
    features = image_features(_gradient())
    assert features["width"] == 64 and features["height"] == 64
    assert features["mean_brightness"] == pytest.approx(126.0)
    assert features["contrast"] > 0
    assert features["sharpness_laplacian_variance"] > 0
    assert sum(features["grayscale_histogram"]) == pytest.approx(1.0)
    assert sum(features["hue_histogram"]) == pytest.approx(1.0)


def test_psi_and_drift_levels() -> None:
    assert population_stability_index([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0)
    assert population_stability_index([0.9, 0.1], [0.1, 0.9]) > 1
    thresholds = {"warning": 0.1, "high": 0.25}
    assert classify_drift(0.05, thresholds) == "normal"
    assert classify_drift(0.1, thresholds) == "warning"
    assert classify_drift(0.25, thresholds) == "high"


def test_output_distribution_metrics() -> None:
    metrics = output_features(
        [
            [
                {"class_name": "Person", "confidence": 0.9},
                {"class_name": "helmet", "confidence": 0.8},
            ],
            [{"class_name": "no_helmet", "confidence": 0.7}],
        ]
    )
    assert metrics["detections_per_frame"] == 1.5
    assert metrics["per_class"]["Person"]["frequency_per_frame"] == 0.5
    assert metrics["zero_person_frame_fraction"] == 0.5
    assert metrics["negative_ppe_frame_fraction"] == 0.5


def test_scoring_serialization_and_reference_loading(tmp_path: Path) -> None:
    features = image_features(_gradient())
    reference = _reference(features)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference.to_dict()), encoding="utf-8")
    loaded = DriftReference.load(path)
    outputs = output_features([[{"class_name": "Person", "confidence": 0.9}]])
    score = score_drift([features], outputs, loaded)
    record = DriftWindowRecord(
        "source",
        0.0,
        5.0,
        features,
        outputs,
        score,
        score["overall_level"],
    )
    assert record.to_dict()["source_id"] == "source"
    assert score["input_score"] == pytest.approx(0)


def test_fixed_time_windowing_uses_timestamps() -> None:
    timestamps = [0.0, 1.0, 4.9, 5.0, 9.9, 10.0]
    assert fixed_time_windows(timestamps, 5.0) == ((0, 3), (3, 5), (5, 6))


@pytest.mark.parametrize(
    "call",
    [
        lambda: apply_perturbation(np.zeros((2, 2)), "gaussian_blur", "mild"),
        lambda: population_stability_index([1], [1, 2]),
        lambda: output_features([]),
        lambda: image_features(np.zeros((2, 2, 3), dtype=np.float32)),
        lambda: fixed_time_windows([1.0, 0.0], 5.0),
    ],
)
def test_malformed_inputs_reject(call) -> None:
    with pytest.raises(ValueError):
        call()


def test_malformed_reference_rejects(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        DriftReference.load(path)
