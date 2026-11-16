"""Run frozen-detector robustness and practical drift evaluation on validation data."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class

from visionops.calibration import operating_counts
from visionops.detection_analysis import Detection, box_iou
from visionops.drift.features import CLASS_NAMES, image_features, output_features
from visionops.drift.monitor import (
    DriftReference,
    aggregate_input_features,
    score_drift,
)
from visionops.drift.perturbations import (
    PERTURBATION_PARAMETERS,
    SEVERITIES,
    apply_perturbation,
)
from visionops.video.config import load_frozen_detector_config

ROOT = Path(__file__).resolve().parents[1]
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
MATERIAL_MAP_RELATIVE_DROP = 0.10
MATERIAL_VIOLATION_RECALL_POINT_DROP = 0.10
DATASETS = {
    "cppe_validation": ROOT / "data/generated/cppe-five-class",
    "rf100_filtered_validation": ROOT / "data/generated/rf100-valid-filtered",
}


def _dataset_records(root: Path) -> list[tuple[Path, Path]]:
    split = "val" if (root / "images/val").is_dir() else "valid"
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    records = []
    for image in sorted(image_dir.glob("*.*")):
        label = label_dir / f"{image.stem}.txt"
        if label.is_file():
            records.append((image, label))
    if not records:
        raise RuntimeError(f"No validation records found under {root}")
    return records


def _ground_truth(path: Path) -> list[Detection]:
    targets = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, width, height = map(float, line.split())
        targets.append(
            Detection(
                int(class_id),
                (
                    x - width / 2,
                    y - height / 2,
                    x + width / 2,
                    y + height / 2,
                ),
            )
        )
    return targets


def _predictions(result: Any, width: int, height: int) -> list[Detection]:
    if result.boxes is None:
        return []
    return [
        Detection(
            int(result.boxes.cls[index].item()),
            tuple(
                float(value)
                for value in (
                    result.boxes.xyxy[index]
                    / result.boxes.xyxy[index].new_tensor(
                        [width, height, width, height]
                    )
                ).tolist()
            ),
            float(result.boxes.conf[index].item()),
        )
        for index in range(len(result.boxes))
    ]


def _true_positive_matrix(
    predictions: list[Detection], targets: list[Detection]
) -> np.ndarray:
    correct = np.zeros((len(predictions), len(IOU_THRESHOLDS)), dtype=bool)
    if not predictions or not targets:
        return correct
    iou = np.asarray(
        [
            [
                box_iou(target.xyxy, prediction.xyxy)
                if target.class_id == prediction.class_id
                else 0.0
                for prediction in predictions
            ]
            for target in targets
        ]
    )
    matched = np.zeros((len(targets), len(IOU_THRESHOLDS)), dtype=bool)
    for prediction_index in np.flatnonzero((iou >= IOU_THRESHOLDS.min()).any(axis=0)):
        available = np.where(matched, 0, iou[:, prediction_index, None])
        target_indices = available.argmax(axis=0)
        threshold_indices = np.arange(len(IOU_THRESHOLDS))
        correct[prediction_index] = (
            available[target_indices, threshold_indices] >= IOU_THRESHOLDS
        )
        matched[target_indices, threshold_indices] |= correct[prediction_index]
    return correct


def _ranking_metrics(
    true_positive: list[np.ndarray],
    confidence: list[float],
    predicted_class: list[int],
    target_class: list[int],
) -> dict[str, object]:
    tp = np.concatenate(true_positive, axis=0) if true_positive else np.zeros((0, 10))
    result = ap_per_class(
        tp,
        np.asarray(confidence),
        np.asarray(predicted_class),
        np.asarray(target_class),
        plot=False,
        names={index: name for index, name in enumerate(CLASS_NAMES)},
    )
    _, _, precision, recall, _, average_precision, classes, *_ = result
    per_class = {}
    for position, class_id in enumerate(classes):
        per_class[CLASS_NAMES[class_id]] = {
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "ap50": float(average_precision[position, 0]),
            "map50_95": float(average_precision[position].mean()),
        }
    return {
        "overall": {
            "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)),
            "map50": float(np.mean(average_precision[:, 0])),
            "map50_95": float(np.mean(average_precision)),
        },
        "per_class": per_class,
    }


def evaluate_condition(
    model: YOLO,
    records: list[tuple[Path, Path]],
    config: Any,
    perturbation: str | None,
    severity: str | None,
) -> tuple[dict[str, object], list[dict[str, object]], list[list[dict[str, object]]]]:
    calibration_records = []
    true_positive = []
    confidence = []
    predicted_class = []
    target_class = []
    inputs = []
    output_frames = []
    feature_latencies = []
    images = []
    targets_by_image = []
    for index, (image_path, label_path) in enumerate(records):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        if perturbation is not None and severity is not None:
            image = apply_perturbation(image, perturbation, severity, seed=42 + index)
        feature_started = time.perf_counter()
        inputs.append(image_features(image))
        feature_latencies.append(time.perf_counter() - feature_started)
        images.append(image)
        targets_by_image.append(_ground_truth(label_path))
    results_by_index: dict[int, Any] = {}
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, image in enumerate(images):
        groups[image.shape[:2]].append(index)
    for indices in groups.values():
        for offset in range(0, len(indices), 8):
            batch_indices = indices[offset : offset + 8]
            source = [images[index] for index in batch_indices]
            batch_results = model.predict(
                source=source if len(source) > 1 else source[0],
                imgsz=config.imgsz,
                conf=0.001,
                iou=config.nms_iou,
                device=0,
                batch=len(source),
                verbose=False,
            )
            for index, result in zip(batch_indices, batch_results):
                results_by_index[index] = result
    for index, image in enumerate(images):
        result = results_by_index[index]
        height, width = image.shape[:2]
        targets = targets_by_image[index]
        predictions = _predictions(result, width, height)
        calibration_records.append((targets, predictions))
        true_positive.append(_true_positive_matrix(predictions, targets))
        confidence.extend(prediction.confidence for prediction in predictions)
        predicted_class.extend(prediction.class_id for prediction in predictions)
        target_class.extend(target.class_id for target in targets)
        thresholded = [
            prediction
            for prediction in predictions
            if prediction.confidence
            >= config.confidence_thresholds[config.classes[prediction.class_id]]
        ]
        output_frames.append(
            [
                {
                    "class_name": config.classes[prediction.class_id],
                    "confidence": prediction.confidence,
                }
                for prediction in thresholded
            ]
        )
    overall_counts, per_class_counts = operating_counts(
        calibration_records,
        class_count=5,
        global_threshold=config.global_confidence_threshold,
        class_thresholds={
            class_id: config.confidence_thresholds[class_name]
            for class_id, class_name in config.classes.items()
        },
        match_iou=0.50,
    )
    ranking = _ranking_metrics(true_positive, confidence, predicted_class, target_class)
    ranking["operational"] = {
        "overall": {
            "tp": overall_counts.tp,
            "fp": overall_counts.fp,
            "fn": overall_counts.fn,
            "precision": overall_counts.precision,
            "recall": overall_counts.recall,
            "f1": overall_counts.f_beta(),
        },
        "per_class": {
            CLASS_NAMES[class_id]: {
                "tp": counts.tp,
                "fp": counts.fp,
                "fn": counts.fn,
                "precision": counts.precision,
                "recall": counts.recall,
                "f1": counts.f_beta(),
            }
            for class_id, counts in per_class_counts.items()
        },
    }
    ordered = sorted(value * 1000 for value in feature_latencies)
    ranking["monitoring_feature_latency_ms"] = {
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p95": ordered[round((len(ordered) - 1) * 0.95)],
    }
    return ranking, inputs, output_frames


def _output_reference(outputs: dict[str, object]) -> dict[str, object]:
    frequencies = [
        outputs["per_class"][name]["frequency_per_frame"] for name in CLASS_NAMES
    ]
    total = sum(frequencies)
    return {
        "class_frequency_distribution": [
            value / total if total else 0.0 for value in frequencies
        ],
        "clean_output_metrics": outputs,
    }


def _relative_drop(clean: float, observed: float) -> float:
    return (clean - observed) / clean if clean else 0.0


def _rank(values: list[float]) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _correlation(first: list[float], second: list[float]) -> dict[str, float]:
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return {"pearson": 0.0, "spearman": 0.0}
    return {
        "pearson": float(np.corrcoef(first, second)[0, 1]),
        "spearman": float(np.corrcoef(_rank(first), _rank(second))[0, 1]),
    }


def _plots(metrics: dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = metrics["conditions"]
    for filename, key, ylabel in (
        ("map50_95_vs_severity.png", "map50_95", "mAP50-95"),
        ("violation_recall_vs_severity.png", "violation_recall", "Violation recall"),
        ("drift_score_vs_severity.png", "drift_score", "Drift score"),
    ):
        figure, axis = plt.subplots(figsize=(10, 6))
        for perturbation in PERTURBATION_PARAMETERS:
            values = []
            for severity in SEVERITIES:
                rows = [
                    row
                    for row in conditions.values()
                    if row["perturbation"] == perturbation
                    and row["severity"] == severity
                ]
                values.append(float(np.mean([row[key] for row in rows])))
            axis.plot(SEVERITIES, values, marker="o", label=perturbation)
        axis.set_xlabel("Severity")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)
    rows = list(conditions.values())
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        [row["drift_score"] for row in rows],
        [row["map50_95_relative_drop"] for row in rows],
        alpha=0.7,
    )
    axis.set_xlabel("Overall drift score")
    axis.set_ylabel("Relative mAP50-95 drop")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "drift_vs_degradation.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 5))
    clean = metrics["clean_domain_drift"]
    axis.bar(clean.keys(), [value["overall_score"] for value in clean.values()])
    axis.axhline(0.1, linestyle="--", label="warning")
    axis.axhline(0.25, linestyle=":", label="high")
    axis.set_ylabel("Drift score vs combined clean reference")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "clean_domain_drift.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-file",
        type=Path,
        default=ROOT / "results/robustness_drift/evaluation.json",
    )
    args = parser.parse_args()
    args.work_file.parent.mkdir(parents=True, exist_ok=True)
    existing_conditions = {}
    if args.work_file.is_file():
        existing = json.loads(args.work_file.read_text(encoding="utf-8"))
        existing_conditions = existing.get("conditions", {})
    config = load_frozen_detector_config(
        ROOT / "configs/production_detector.yaml", ROOT
    )
    model = YOLO(str(config.weights_path))
    raw: dict[str, object] = {
        "protocol": {
            "seed": 42,
            "datasets": list(DATASETS),
            "rf100_test_used": False,
            "perturbations": PERTURBATION_PARAMETERS,
            "map_material_relative_drop": MATERIAL_MAP_RELATIVE_DROP,
            "violation_recall_material_point_drop": (
                MATERIAL_VIOLATION_RECALL_POINT_DROP
            ),
            "production_thresholds": config.confidence_thresholds,
        },
        "clean": {},
        "conditions": existing_conditions,
    }
    clean_inputs: dict[str, list[dict[str, object]]] = {}
    clean_outputs: dict[str, list[list[dict[str, object]]]] = {}
    for dataset_name, dataset_root in DATASETS.items():
        records = _dataset_records(dataset_root)
        metrics, inputs, outputs = evaluate_condition(
            model, records, config, None, None
        )
        raw["clean"][dataset_name] = metrics
        clean_inputs[dataset_name] = inputs
        clean_outputs[dataset_name] = outputs
        args.work_file.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    combined_inputs = [
        record for records in clean_inputs.values() for record in records
    ]
    combined_outputs = [frame for frames in clean_outputs.values() for frame in frames]
    combined_output_metrics = output_features(combined_outputs)
    reference = DriftReference(
        version=1,
        sample_count=len(combined_inputs),
        input_profile=aggregate_input_features(combined_inputs),
        output_profile=_output_reference(combined_output_metrics),
        alert_thresholds={"warning": 0.10, "high": 0.25},
    )
    reference_path = ROOT / "configs/drift_reference.json"
    reference_path.write_text(
        json.dumps(reference.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    raw["clean_domain_drift"] = {}
    for dataset_name in DATASETS:
        outputs = output_features(clean_outputs[dataset_name])
        raw["clean_domain_drift"][dataset_name] = score_drift(
            clean_inputs[dataset_name], outputs, reference
        )
    for dataset_name, dataset_root in DATASETS.items():
        records = _dataset_records(dataset_root)
        clean = raw["clean"][dataset_name]
        for perturbation in PERTURBATION_PARAMETERS:
            for severity in SEVERITIES:
                key = f"{dataset_name}:{perturbation}:{severity}"
                if key in raw["conditions"]:
                    continue
                metrics, inputs, outputs = evaluate_condition(
                    model, records, config, perturbation, severity
                )
                drift = score_drift(inputs, output_features(outputs), reference)
                clean_map = clean["overall"]["map50_95"]
                observed_map = metrics["overall"]["map50_95"]
                clean_violation = np.mean(
                    [
                        clean["operational"]["per_class"][name]["recall"]
                        for name in ("no_helmet", "no_vest")
                    ]
                )
                observed_violation = np.mean(
                    [
                        metrics["operational"]["per_class"][name]["recall"]
                        for name in ("no_helmet", "no_vest")
                    ]
                )
                raw["conditions"][key] = {
                    "dataset": dataset_name,
                    "perturbation": perturbation,
                    "severity": severity,
                    "parameter": PERTURBATION_PARAMETERS[perturbation][severity],
                    "metrics": metrics,
                    "drift": drift,
                    "map50_95": observed_map,
                    "map50_95_relative_drop": _relative_drop(clean_map, observed_map),
                    "violation_recall": observed_violation,
                    "violation_recall_point_drop": (
                        clean_violation - observed_violation
                    ),
                    "drift_score": drift["overall_score"],
                }
                args.work_file.write_text(
                    json.dumps(raw, indent=2) + "\n", encoding="utf-8"
                )
    rows = list(raw["conditions"].values())
    drift_scores = [row["drift_score"] for row in rows]
    raw["correlation"] = {
        "map50_95_relative_drop": _correlation(
            drift_scores, [row["map50_95_relative_drop"] for row in rows]
        ),
        "person_recall_drop": _correlation(
            drift_scores,
            [
                raw["clean"][row["dataset"]]["operational"]["per_class"]["Person"][
                    "recall"
                ]
                - row["metrics"]["operational"]["per_class"]["Person"]["recall"]
                for row in rows
            ],
        ),
        "no_helmet_recall_drop": _correlation(
            drift_scores,
            [
                raw["clean"][row["dataset"]]["operational"]["per_class"]["no_helmet"][
                    "recall"
                ]
                - row["metrics"]["operational"]["per_class"]["no_helmet"]["recall"]
                for row in rows
            ],
        ),
        "no_vest_recall_drop": _correlation(
            drift_scores,
            [
                raw["clean"][row["dataset"]]["operational"]["per_class"]["no_vest"][
                    "recall"
                ]
                - row["metrics"]["operational"]["per_class"]["no_vest"]["recall"]
                for row in rows
            ],
        ),
    }
    docs_dir = ROOT / "docs/experiments/robustness_drift"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "metrics.json").write_text(
        json.dumps(raw, indent=2) + "\n", encoding="utf-8"
    )
    _plots(raw, docs_dir)
    print(
        json.dumps(
            {
                "clean": raw["clean"],
                "clean_domain_drift": raw["clean_domain_drift"],
                "correlation": raw["correlation"],
                "condition_count": len(raw["conditions"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
