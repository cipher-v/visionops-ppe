"""Pure operating-threshold calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from visionops.detection_analysis import Detection, match_detections


@dataclass(frozen=True)
class OperatingCounts:
    """Detection counts at one operating point."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    def f_beta(self, beta: float = 1.0) -> float:
        """Return F-beta from accumulated counts."""
        if beta <= 0:
            raise ValueError("beta must be positive")
        precision = self.precision
        recall = self.recall
        denominator = beta * beta * precision + recall
        if denominator == 0:
            return 0.0
        return (1 + beta * beta) * precision * recall / denominator

    def __add__(self, other: OperatingCounts) -> OperatingCounts:
        return OperatingCounts(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
        )


def threshold_grid(
    start: str = "0.05", stop: str = "0.90", step: str = "0.01"
) -> tuple[float, ...]:
    """Build an inclusive decimal threshold grid without float drift."""
    first, last, increment = map(Decimal, (start, stop, step))
    if increment <= 0 or first < 0 or last > 1 or first > last:
        raise ValueError("Invalid confidence grid")
    values = []
    current = first
    while current <= last:
        values.append(float(current))
        current += increment
    return tuple(values)


def apply_threshold_policy(
    predictions: list[Detection],
    global_threshold: float,
    class_thresholds: dict[int, float] | None = None,
) -> list[Detection]:
    """Filter predictions with a global or class-specific threshold policy."""
    if not 0 <= global_threshold <= 1:
        raise ValueError("Global threshold must be normalized")
    class_thresholds = class_thresholds or {}
    if any(not 0 <= value <= 1 for value in class_thresholds.values()):
        raise ValueError("Class thresholds must be normalized")
    return [
        prediction
        for prediction in predictions
        if prediction.confidence
        >= class_thresholds.get(prediction.class_id, global_threshold)
    ]


def operating_counts(
    records: list[tuple[list[Detection], list[Detection]]],
    class_count: int,
    global_threshold: float,
    class_thresholds: dict[int, float] | None = None,
    match_iou: float = 0.50,
) -> tuple[OperatingCounts, dict[int, OperatingCounts]]:
    """Accumulate micro and per-class counts with class-aware one-to-one matching."""
    per_class = {class_id: OperatingCounts() for class_id in range(class_count)}
    for ground_truth, raw_predictions in records:
        predictions = apply_threshold_policy(
            raw_predictions, global_threshold, class_thresholds
        )
        matches = match_detections(ground_truth, predictions, match_iou)
        matched_ground_truth = matches.matched_ground_truth_indices
        matched_predictions = matches.matched_prediction_indices
        for class_id in range(class_count):
            tp = sum(
                ground_truth[index].class_id == class_id
                for index in matched_ground_truth
            )
            fp = sum(
                prediction.class_id == class_id
                for index, prediction in enumerate(predictions)
                if index not in matched_predictions
            )
            fn = sum(
                target.class_id == class_id
                for index, target in enumerate(ground_truth)
                if index not in matched_ground_truth
            )
            per_class[class_id] = per_class[class_id] + OperatingCounts(tp, fp, fn)
    overall = sum(per_class.values(), start=OperatingCounts())
    return overall, per_class


def domain_balanced_score(
    first: OperatingCounts, second: OperatingCounts, beta: float = 1.0
) -> float:
    """Average domain-level F-beta so neither domain dominates by object count."""
    return (first.f_beta(beta) + second.f_beta(beta)) / 2


def select_threshold(
    candidates: list[tuple[float, OperatingCounts, OperatingCounts]],
    beta: float = 1.0,
) -> dict[str, float]:
    """Select maximum balanced F-beta, breaking ties by precision then threshold."""
    if not candidates:
        raise ValueError("At least one threshold candidate is required")
    scored = []
    for threshold, first, second in candidates:
        score = domain_balanced_score(first, second, beta)
        balanced_precision = (first.precision + second.precision) / 2
        scored.append((score, balanced_precision, threshold))
    score, precision, threshold = max(scored)
    return {
        "threshold": threshold,
        "domain_balanced_fbeta": score,
        "domain_balanced_precision": precision,
        "beta": beta,
    }


def class_policy_is_material(
    global_domains: tuple[OperatingCounts, OperatingCounts],
    class_domains: tuple[OperatingCounts, OperatingCounts],
    global_violation_recall: dict[int, float],
    class_violation_recall: dict[int, float],
) -> dict[str, bool | float]:
    """Apply the predeclared complexity gate for class-specific thresholds."""
    global_score = domain_balanced_score(*global_domains)
    class_score = domain_balanced_score(*class_domains)
    score_gain = class_score - global_score
    global_fp = sum(item.fp for item in global_domains)
    class_fp = sum(item.fp for item in class_domains)
    fp_growth = (class_fp - global_fp) / global_fp if global_fp else 0.0
    violation_gain = max(
        class_violation_recall[class_id] - global_violation_recall[class_id]
        for class_id in global_violation_recall
    )
    domain_regression = max(
        global_domains[index].f_beta() - class_domains[index].f_beta()
        for index in range(2)
    )
    material_gain = score_gain >= 0.01 or violation_gain >= 0.05
    acceptable_cost = fp_growth <= 0.20 and domain_regression <= 0.02
    return {
        "global_balanced_f1": global_score,
        "class_balanced_f1": class_score,
        "balanced_f1_gain": score_gain,
        "false_positive_growth": fp_growth,
        "maximum_violation_recall_gain": violation_gain,
        "maximum_domain_f1_regression": domain_regression,
        "material_gain": material_gain,
        "acceptable_cost": acceptable_cost,
        "adopt_class_specific": material_gain and acceptable_cost,
    }


def validate_production_config(values: dict[str, object]) -> None:
    """Validate the repository-owned production detector contract."""
    required = {
        "version",
        "model",
        "input",
        "classes",
        "nms_iou",
        "confidence_policy",
        "preprocessing",
    }
    missing = required - values.keys()
    if missing:
        raise ValueError(f"Missing production config fields: {sorted(missing)}")
    if values["classes"] != {
        0: "helmet",
        1: "vest",
        2: "Person",
        3: "no_helmet",
        4: "no_vest",
    }:
        raise ValueError("Production classes differ from the frozen vocabulary")
    if values["input"].get("imgsz") != 960:
        raise ValueError("Production input resolution must be 960")
    if values["nms_iou"] != 0.70:
        raise ValueError("Production NMS IoU must be 0.70")
    policy = values["confidence_policy"]
    if policy.get("type") not in {"global", "class_specific"}:
        raise ValueError("Unknown confidence policy type")
