"""Reference profiles, PSI scoring, and drift alert records."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCALAR_FEATURES = (
    "mean_brightness",
    "brightness_std",
    "contrast",
    "saturation_mean",
    "sharpness_laplacian_variance",
    "aspect_ratio",
)
PROFILE_SCALAR_FEATURES = SCALAR_FEATURES + ("width", "height")
HISTOGRAM_FEATURES = (
    "grayscale_histogram",
    "hue_histogram",
    "saturation_histogram",
    "value_histogram",
)


def population_stability_index(
    reference: list[float], observed: list[float], smoothing: float = 1e-6
) -> float:
    """Return PSI for equal-length normalized distributions."""
    if smoothing <= 0 or len(reference) != len(observed) or not reference:
        raise ValueError("PSI distributions and smoothing are invalid")
    expected = np.asarray(reference, dtype=float)
    actual = np.asarray(observed, dtype=float)
    if np.any(expected < 0) or np.any(actual < 0):
        raise ValueError("PSI distributions cannot contain negative values")
    expected = (expected + smoothing) / (expected.sum() + smoothing * len(expected))
    actual = (actual + smoothing) / (actual.sum() + smoothing * len(actual))
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def aggregate_input_features(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        raise ValueError("Reference aggregation requires input records")
    aggregated: dict[str, object] = {}
    for feature in PROFILE_SCALAR_FEATURES:
        values = np.asarray([float(record[feature]) for record in records])
        aggregated[feature] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    for feature in HISTOGRAM_FEATURES:
        values = np.asarray([record[feature] for record in records], dtype=float)
        histogram = values.mean(axis=0)
        aggregated[feature] = (histogram / histogram.sum()).tolist()
    return aggregated


@dataclass(frozen=True)
class DriftReference:
    version: int
    sample_count: int
    input_profile: dict[str, object]
    output_profile: dict[str, object]
    alert_thresholds: dict[str, float]

    @classmethod
    def load(cls, path: Path) -> DriftReference:
        if not path.is_file():
            raise FileNotFoundError(f"Drift reference does not exist: {path}")
        values = json.loads(path.read_text(encoding="utf-8"))
        try:
            reference = cls(
                version=int(values["version"]),
                sample_count=int(values["sample_count"]),
                input_profile=dict(values["input_profile"]),
                output_profile=dict(values["output_profile"]),
                alert_thresholds={
                    key: float(value)
                    for key, value in values["alert_thresholds"].items()
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed drift reference: {error}") from error
        if reference.version != 1 or reference.sample_count < 1:
            raise ValueError("Unsupported or empty drift reference")
        if reference.alert_thresholds.keys() != {"warning", "high"}:
            raise ValueError("Drift reference alert thresholds are malformed")
        return reference

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_drift(score: float, thresholds: dict[str, float]) -> str:
    if not math.isfinite(score) or score < 0:
        raise ValueError("Drift score must be finite and non-negative")
    warning, high = thresholds["warning"], thresholds["high"]
    if not 0 <= warning < high:
        raise ValueError("Drift thresholds must be ordered")
    return "high" if score >= high else "warning" if score >= warning else "normal"


def score_drift(
    input_records: list[dict[str, object]],
    output_metrics: dict[str, object],
    reference: DriftReference,
) -> dict[str, object]:
    observed = aggregate_input_features(input_records)
    histogram_psi = {
        feature: population_stability_index(
            reference.input_profile[feature], observed[feature]
        )
        for feature in HISTOGRAM_FEATURES
    }
    scalar_z = {}
    for feature in SCALAR_FEATURES:
        expected = reference.input_profile[feature]
        denominator = max(float(expected["std"]), 1e-6)
        scalar_z[feature] = (
            abs(float(observed[feature]["mean"]) - float(expected["mean"]))
            / denominator
        )
    reference_frequency = reference.output_profile["class_frequency_distribution"]
    observed_frequency = [
        output_metrics["per_class"][name]["frequency_per_frame"]
        for name in ("helmet", "vest", "Person", "no_helmet", "no_vest")
    ]
    output_psi = population_stability_index(reference_frequency, observed_frequency)
    input_score = max(
        max(histogram_psi.values()), min(max(scalar_z.values()) / 10, 1.0)
    )
    overall = max(input_score, output_psi)
    return {
        "input_histogram_psi": histogram_psi,
        "input_scalar_absolute_z": scalar_z,
        "input_score": input_score,
        "output_class_frequency_psi": output_psi,
        "overall_score": overall,
        "overall_level": classify_drift(overall, reference.alert_thresholds),
    }


@dataclass(frozen=True)
class DriftWindowRecord:
    source_id: str
    window_start_seconds: float
    window_end_seconds: float
    input_metrics: dict[str, object]
    output_metrics: dict[str, object]
    drift_scores: dict[str, object]
    overall_level: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fixed_time_windows(
    timestamps: list[float], window_seconds: float
) -> tuple[tuple[int, int], ...]:
    if window_seconds <= 0 or any(
        timestamp < 0 or not math.isfinite(timestamp) for timestamp in timestamps
    ):
        raise ValueError("Window timestamps and duration must be valid")
    if timestamps != sorted(timestamps):
        raise ValueError("Window timestamps must be ordered")
    if not timestamps:
        return ()
    windows = []
    start = 0
    while start < len(timestamps):
        boundary = timestamps[start] + window_seconds
        end = start
        while end < len(timestamps) and timestamps[end] < boundary:
            end += 1
        windows.append((start, end))
        start = end
    return tuple(windows)
