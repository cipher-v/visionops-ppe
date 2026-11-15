"""Select and freeze the final detector operating policy without training."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from ultralytics import YOLO

from analyze_data_diversity import predictions, read_labels
from visionops.calibration import (
    OperatingCounts,
    class_policy_is_material,
    domain_balanced_score,
    operating_counts,
    select_threshold,
    threshold_grid,
    validate_production_config,
)
from visionops.data_diversity import FIVE_CLASS_NAMES, assert_rf100_test_excluded
from visionops.detection_analysis import Detection
from visionops.exposure_control import image_files

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs/experiment_mixed5_yolo26n_960/weights/best.pt"
TRAINING_METADATA = ROOT / "runs/experiment_mixed5_yolo26n_960/training_metadata.json"
DATA_MANIFEST = ROOT / "data/generated/data-diversity-manifest.json"
CPPE_IMAGES = ROOT / "data/generated/cppe-five-class/images/val"
CPPE_LABELS = ROOT / "data/generated/cppe-five-class/labels/val"
RF100_IMAGES = ROOT / "data/generated/rf100-valid-filtered/images/valid"
RF100_LABELS = ROOT / "data/generated/rf100-valid-filtered/labels/valid"
EXPOSURE_METRICS = (
    ROOT / "docs/experiments/data_diversity_exposure_control/metrics.json"
)
DOCS = ROOT / "docs/experiments/final_model_selection"
ASSETS = ROOT / "docs/assets/calibration"
RUNS = ROOT / "runs/final_model_calibration"
FINAL_MODEL_CONFIG = ROOT / "configs/final_model.yaml"
PRODUCTION_CONFIG = ROOT / "configs/production_detector.yaml"
CONFIDENCE_FLOOR = 0.05
NMS_IOU = 0.70
MATCH_IOU = 0.50
VIOLATION_IDS = (3, 4)
DOMAINS = ("cppe_validation", "rf100_filtered_validation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def collect_records(
    model: YOLO, images: Path, labels: Path
) -> list[tuple[list[Detection], list[Detection]]]:
    results = model.predict(
        source=str(images),
        imgsz=960,
        conf=CONFIDENCE_FLOOR,
        iou=NMS_IOU,
        batch=4,
        device=0 if torch.cuda.is_available() else "cpu",
        stream=True,
        verbose=False,
    )
    records = []
    for result in results:
        path = Path(result.path)
        records.append(
            (
                read_labels(labels / f"{path.stem}.txt"),
                predictions(result),
            )
        )
    if len(records) != len(image_files(images)):
        raise RuntimeError("Prediction count differs from calibration image count")
    return records


def counts_dict(counts: OperatingCounts) -> dict[str, float | int]:
    return {
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "precision": counts.precision,
        "recall": counts.recall,
        "f1": counts.f_beta(1),
        "f2": counts.f_beta(2),
    }


def evaluate_policy(
    records: dict[str, list[tuple[list[Detection], list[Detection]]]],
    global_threshold: float,
    class_thresholds: dict[int, float] | None = None,
) -> tuple[dict[str, object], dict[str, OperatingCounts]]:
    values = {}
    overall_counts = {}
    for domain in DOMAINS:
        overall, per_class = operating_counts(
            records[domain],
            len(FIVE_CLASS_NAMES),
            global_threshold,
            class_thresholds,
            MATCH_IOU,
        )
        overall_counts[domain] = overall
        values[domain] = {
            "overall": counts_dict(overall),
            "per_class": {
                FIVE_CLASS_NAMES[class_id]: counts_dict(counts)
                for class_id, counts in per_class.items()
            },
        }
    return values, overall_counts


def build_sweep(
    records: dict[str, list[tuple[list[Detection], list[Detection]]]],
) -> tuple[list[dict[str, object]], dict[float, dict[str, object]]]:
    serialized = []
    raw = {}
    for threshold in threshold_grid():
        raw[threshold] = {}
        item = {"threshold": threshold, "domains": {}}
        for domain in DOMAINS:
            overall, per_class = operating_counts(
                records[domain], len(FIVE_CLASS_NAMES), threshold, None, MATCH_IOU
            )
            raw[threshold][domain] = {"overall": overall, "per_class": per_class}
            item["domains"][domain] = {
                "overall": counts_dict(overall),
                "per_class": {
                    FIVE_CLASS_NAMES[class_id]: counts_dict(counts)
                    for class_id, counts in per_class.items()
                },
            }
        serialized.append(item)
    return serialized, raw


def select_policies(
    raw: dict[float, dict[str, object]],
    records: dict[str, list[tuple[list[Detection], list[Detection]]]],
) -> dict[str, object]:
    global_selection = select_threshold(
        [
            (
                threshold,
                values[DOMAINS[0]]["overall"],
                values[DOMAINS[1]]["overall"],
            )
            for threshold, values in raw.items()
        ]
    )
    global_threshold = global_selection["threshold"]
    class_candidates = {}
    f2_candidates = {}
    proposed_thresholds = {}
    for class_id, class_name in enumerate(FIVE_CLASS_NAMES):
        candidates = [
            (
                threshold,
                values[DOMAINS[0]]["per_class"][class_id],
                values[DOMAINS[1]]["per_class"][class_id],
            )
            for threshold, values in raw.items()
        ]
        f1_choice = select_threshold(candidates, beta=1)
        class_candidates[class_name] = f1_choice
        proposed_thresholds[class_id] = f1_choice["threshold"]
        if class_id in VIOLATION_IDS:
            f2_candidates[class_name] = select_threshold(candidates, beta=2)

    no_helmet_id = 3
    global_no_helmet = (
        raw[global_threshold][DOMAINS[0]]["per_class"][no_helmet_id],
        raw[global_threshold][DOMAINS[1]]["per_class"][no_helmet_id],
    )
    candidate_threshold = proposed_thresholds[no_helmet_id]
    candidate_no_helmet = (
        raw[candidate_threshold][DOMAINS[0]]["per_class"][no_helmet_id],
        raw[candidate_threshold][DOMAINS[1]]["per_class"][no_helmet_id],
    )
    no_helmet_guard = {
        "balanced_f1_gain": domain_balanced_score(*candidate_no_helmet)
        - domain_balanced_score(*global_no_helmet),
        "cppe_f1_gain": candidate_no_helmet[0].f_beta() - global_no_helmet[0].f_beta(),
    }
    no_helmet_guard["class_threshold_allowed"] = (
        no_helmet_guard["balanced_f1_gain"] >= 0.01
        and no_helmet_guard["cppe_f1_gain"] >= 0.01
    )
    if not no_helmet_guard["class_threshold_allowed"]:
        proposed_thresholds[no_helmet_id] = global_threshold

    global_metrics, global_domains = evaluate_policy(records, global_threshold)
    class_metrics, class_domains = evaluate_policy(
        records, global_threshold, proposed_thresholds
    )
    global_violation_recall = {}
    class_violation_recall = {}
    for class_id in VIOLATION_IDS:
        class_name = FIVE_CLASS_NAMES[class_id]
        global_violation_recall[class_id] = (
            sum(
                global_metrics[domain]["per_class"][class_name]["recall"]
                for domain in DOMAINS
            )
            / 2
        )
        class_violation_recall[class_id] = (
            sum(
                class_metrics[domain]["per_class"][class_name]["recall"]
                for domain in DOMAINS
            )
            / 2
        )
    gate = class_policy_is_material(
        (global_domains[DOMAINS[0]], global_domains[DOMAINS[1]]),
        (class_domains[DOMAINS[0]], class_domains[DOMAINS[1]]),
        global_violation_recall,
        class_violation_recall,
    )
    chosen_type = "class_specific" if gate["adopt_class_specific"] else "global"
    chosen_thresholds = proposed_thresholds if chosen_type == "class_specific" else {}
    chosen_metrics = (
        class_metrics if chosen_type == "class_specific" else global_metrics
    )
    return {
        "global_selection": global_selection,
        "class_f1_candidates": class_candidates,
        "violation_f2_candidates": f2_candidates,
        "no_helmet_small_sample_guard": no_helmet_guard,
        "proposed_class_thresholds": {
            FIVE_CLASS_NAMES[class_id]: value
            for class_id, value in proposed_thresholds.items()
        },
        "global_policy_metrics": global_metrics,
        "class_policy_metrics": class_metrics,
        "class_policy_gate": gate,
        "chosen_policy_type": chosen_type,
        "chosen_global_threshold": global_threshold,
        "chosen_class_thresholds": {
            FIVE_CLASS_NAMES[class_id]: value
            for class_id, value in chosen_thresholds.items()
        },
        "chosen_policy_metrics": chosen_metrics,
    }


def create_plots(sweep: list[dict[str, object]]) -> list[str]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    outputs = []
    for class_name in FIVE_CLASS_NAMES:
        figure, axis = plt.subplots(figsize=(7, 5))
        for domain in DOMAINS:
            precision = [
                item["domains"][domain]["per_class"][class_name]["precision"]
                for item in sweep
            ]
            recall = [
                item["domains"][domain]["per_class"][class_name]["recall"]
                for item in sweep
            ]
            axis.plot(recall, precision, label=domain)
        axis.set(
            xlabel="Recall",
            ylabel="Precision",
            title=f"{class_name}: operating precision-recall",
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        filename = f"precision_recall_{class_name.lower()}.png"
        figure.savefig(ASSETS / filename, dpi=150)
        plt.close(figure)
        outputs.append(filename)

    thresholds = [item["threshold"] for item in sweep]
    figure, axes = plt.subplots(3, 2, figsize=(12, 13))
    labels = (*FIVE_CLASS_NAMES, "overall")
    for axis, label in zip(axes.flat, labels, strict=True):
        for domain in DOMAINS:
            values = [
                item["domains"][domain][
                    "overall" if label == "overall" else "per_class"
                ][label]["f1"]
                if label != "overall"
                else item["domains"][domain]["overall"]["f1"]
                for item in sweep
            ]
            axis.plot(thresholds, values, label=domain)
        axis.set(title=label, xlabel="Confidence", ylabel="F1")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    filename = "f1_vs_confidence.png"
    figure.savefig(ASSETS / filename, dpi=150)
    plt.close(figure)
    outputs.append(filename)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, class_name in zip(axes, ("no_helmet", "no_vest"), strict=True):
        for domain in DOMAINS:
            axis.plot(
                thresholds,
                [
                    item["domains"][domain]["per_class"][class_name]["f2"]
                    for item in sweep
                ],
                label=domain,
            )
        axis.set(title=class_name, xlabel="Confidence", ylabel="F2")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    filename = "violation_f2_vs_confidence.png"
    figure.savefig(ASSETS / filename, dpi=150)
    plt.close(figure)
    outputs.append(filename)

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row, class_name in enumerate(("no_helmet", "no_vest")):
        for column, metric in enumerate(("fp", "fn")):
            axis = axes[row][column]
            for domain in DOMAINS:
                axis.plot(
                    thresholds,
                    [
                        item["domains"][domain]["per_class"][class_name][metric]
                        for item in sweep
                    ],
                    label=domain,
                )
            axis.set(
                title=f"{class_name} {metric.upper()}",
                xlabel="Confidence",
                ylabel="Count",
            )
            axis.grid(alpha=0.25)
            axis.legend()
    figure.tight_layout()
    filename = "violation_fp_fn_vs_confidence.png"
    figure.savefig(ASSETS / filename, dpi=150)
    plt.close(figure)
    outputs.append(filename)
    return outputs


def write_configs(
    model: YOLO,
    training: dict[str, object],
    data_manifest: dict[str, object],
    policy: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    weight_hash = sha256(WEIGHTS)
    parameter_count = sum(parameter.numel() for parameter in model.model.parameters())
    final_model = {
        "version": "visionops-ppe-detector-v1",
        "architecture": "YOLO26n detect",
        "weights": str(WEIGHTS.relative_to(ROOT)).replace("\\", "/"),
        "weights_sha256": weight_hash,
        "weights_size_bytes": WEIGHTS.stat().st_size,
        "parameter_count": parameter_count,
        "source_run": "runs/experiment_mixed5_yolo26n_960",
        "best_epoch": training["best_epoch"],
        "training_configuration": training["config"],
        "training_data": data_manifest["conditions"]["mixed_domain"],
        "classes": {index: name for index, name in enumerate(FIVE_CLASS_NAMES)},
        "rf100_test_used_for_selection": False,
    }
    FINAL_MODEL_CONFIG.write_text(
        yaml.safe_dump(final_model, sort_keys=False), encoding="utf-8"
    )
    global_threshold = policy["chosen_global_threshold"]
    production = {
        "version": 1,
        "model": {
            "manifest": "configs/final_model.yaml",
            "weights": final_model["weights"],
            "sha256": weight_hash,
            "architecture": "YOLO26n detect",
        },
        "input": {"imgsz": 960, "shape_policy": "letterbox_preserve_aspect_ratio"},
        "classes": final_model["classes"],
        "nms_iou": NMS_IOU,
        "confidence_policy": {
            "type": policy["chosen_policy_type"],
            "global_threshold": global_threshold,
            "per_class": policy["chosen_class_thresholds"],
        },
        "preprocessing": {
            "color_space": "RGB",
            "pixel_scaling": "uint8_0_255_to_float_0_1",
            "layout": "NCHW",
            "letterbox_padding_value": 114,
        },
        "calibration": {
            "domains": ["construction_ppe_validation", "rf100_filtered_validation"],
            "matching_iou": MATCH_IOU,
            "grid": {"start": 0.05, "stop": 0.90, "step": 0.01},
            "rf100_test_used": False,
        },
        "operational_notes": {
            "below_threshold": "ignore detection",
            "single_frame_violation": "detector evidence only; not an incident",
            "deferred": [
                "tracking",
                "PPE-to-person association",
                "temporal persistence",
                "dwell time",
                "cooldown",
            ],
        },
    }
    validate_production_config(production)
    PRODUCTION_CONFIG.write_text(
        yaml.safe_dump(production, sort_keys=False), encoding="utf-8"
    )
    return final_model, production


def write_calibration_report(metrics: dict[str, object]) -> None:
    policy = metrics["policy_selection"]
    global_threshold = policy["global_selection"]["threshold"]
    candidate_rows = []
    for class_name in FIVE_CLASS_NAMES:
        f1 = policy["class_f1_candidates"][class_name]
        f2 = policy["violation_f2_candidates"].get(class_name)
        candidate_rows.append(
            f"| {class_name} | {f1['threshold']:.2f} | {f1['domain_balanced_fbeta']:.4f} | "
            + (
                f"{f2['threshold']:.2f} | {f2['domain_balanced_fbeta']:.4f} |"
                if f2
                else "-- | -- |"
            )
        )
    sweep_by_threshold = {
        item["threshold"]: item["domains"] for item in metrics["threshold_sweep"]
    }
    violation_rows = []
    for class_name in ("no_helmet", "no_vest"):
        choices = (
            ("F1 candidate", policy["class_f1_candidates"][class_name]["threshold"]),
            (
                "F2 candidate",
                policy["violation_f2_candidates"][class_name]["threshold"],
            ),
        )
        for objective, threshold in choices:
            for domain in DOMAINS:
                value = sweep_by_threshold[threshold][domain]["per_class"][class_name]
                violation_rows.append(
                    f"| {class_name} | {objective} | {domain} | {value['tp'] + value['fn']} | "
                    f"{threshold:.2f} | {value['precision']:.4f} | {value['recall']:.4f} | "
                    f"{value['f1']:.4f} | {value['f2']:.4f} | {value['fp']} | {value['fn']} |"
                )
    final_rows = []
    for domain in DOMAINS:
        domain_values = policy["chosen_policy_metrics"][domain]
        overall = domain_values["overall"]
        final_rows.append(
            f"| {domain} | overall | -- | {overall['precision']:.4f} | {overall['recall']:.4f} | {overall['f1']:.4f} | {overall['tp']} | {overall['fp']} | {overall['fn']} |"
        )
        for class_name in FIVE_CLASS_NAMES:
            value = domain_values["per_class"][class_name]
            threshold = policy["chosen_class_thresholds"].get(
                class_name, policy["chosen_global_threshold"]
            )
            final_rows.append(
                f"| {domain} | {class_name} | {threshold:.2f} | {value['precision']:.4f} | {value['recall']:.4f} | {value['f1']:.4f} | {value['tp']} | {value['fp']} | {value['fn']} |"
            )
    global_rows = []
    class_rows = []
    for domain in DOMAINS:
        for source, rows in (
            ("global_policy_metrics", global_rows),
            ("class_policy_metrics", class_rows),
        ):
            value = policy[source][domain]["overall"]
            no_helmet = policy[source][domain]["per_class"]["no_helmet"]
            no_vest = policy[source][domain]["per_class"]["no_vest"]
            rows.append(
                f"| {domain} | {value['precision']:.4f} | {value['recall']:.4f} | {value['f1']:.4f} | {value['fp']} | {value['fn']} | {no_helmet['recall']:.4f} | {no_helmet['fp']} | {no_vest['recall']:.4f} | {no_vest['fp']} |"
            )
    exposure = metrics["reference_metrics"]
    text = f"""# Operating-threshold calibration

## Frozen setup

Candidate C is frozen as production v1. RF100 test was not used. Calibration uses source-domain Construction-PPE validation and held-out target-domain filtered RF100 validation. Predictions were generated once at confidence {CONFIDENCE_FLOOR}, NMS IoU {NMS_IOU}; every threshold uses class-aware one-to-one IoU {MATCH_IOU} assignment.

## Global threshold

The domain-balanced micro-F1 optimum is **{global_threshold:.2f}**. Ties were resolved by mean precision then higher threshold.

## Per-class candidates

| Class | F1 threshold | Balanced F1 | F2 threshold | Balanced F2 |
|---|---:|---:|---:|---:|
{chr(10).join(candidate_rows)}

F2 thresholds for violations are diagnostic alternatives, not automatic production choices. Filtered RF100 contains only 11 `no_helmet` instances; the small-sample guard result is `{policy["no_helmet_small_sample_guard"]["class_threshold_allowed"]}`.

## Violation-class trade-offs

| Class | Objective | Domain | Instances | Threshold | Precision | Recall | F1 | F2 | FP | FN |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(violation_rows)}

The `no_helmet` F1 candidate is not adopted because it loses source-domain F1 and the
11-instance RF100 result cannot establish stability. The F2 choices are shown as
review-cost alternatives only; neither is silently substituted into production.

## Global policy

| Domain | Precision | Recall | F1 | FP | FN | no_helmet R | no_helmet FP | no_vest R | no_vest FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(global_rows)}

## Candidate class-specific policy

| Domain | Precision | Recall | F1 | FP | FN | no_helmet R | no_helmet FP | no_vest R | no_vest FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(class_rows)}

Complexity-gate result: `{policy["class_policy_gate"]["adopt_class_specific"]}`. Final policy: **{policy["chosen_policy_type"]}**. The gate requires a material balanced-F1 or violation-recall gain, FP growth <=20%, and no domain F1 regression beyond 0.02.

## Final operating results

| Domain | Scope | Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(final_rows)}

Confidence calibration does not change mAP. Frozen candidate C retains Construction-PPE mAP50 `{exposure["cppe_validation"]["overall"]["map50"]:.4f}`, mAP50-95 `{exposure["cppe_validation"]["overall"]["map50_95"]:.4f}`, and filtered RF100 mAP50 `{exposure["rf100_filtered_validation"]["overall"]["map50"]:.4f}`, mAP50-95 `{exposure["rf100_filtered_validation"]["overall"]["map50_95"]:.4f}`.

## Detector policy

Detections below their configured threshold are ignored. These outputs are detector-level evidence only. A one-frame `no_helmet` or `no_vest` detection is not an incident; tracking, person-PPE association, temporal persistence, dwell time, and cooldown remain explicitly deferred.
"""
    (DOCS / "calibration.md").write_text(text, encoding="utf-8")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    guarded = [WEIGHTS, CPPE_IMAGES, CPPE_LABELS, RF100_IMAGES, RF100_LABELS]
    assert_rf100_test_excluded(guarded)
    training = json.loads(TRAINING_METADATA.read_text(encoding="utf-8"))
    data_manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    if data_manifest["rf100_test_used"] is not False:
        raise RuntimeError("Training-data manifest violates RF100-test freeze")
    model = YOLO(str(WEIGHTS))
    records = {
        "cppe_validation": collect_records(model, CPPE_IMAGES, CPPE_LABELS),
        "rf100_filtered_validation": collect_records(model, RF100_IMAGES, RF100_LABELS),
    }
    sweep, raw = build_sweep(records)
    policy = select_policies(raw, records)
    plots = create_plots(sweep)
    final_model, production = write_configs(model, training, data_manifest, policy)
    reference = json.loads(EXPOSURE_METRICS.read_text(encoding="utf-8"))["validation"][
        "mixed_full_budget"
    ]
    metrics = {
        "created_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "protocol": {
            "rf100_test_used": False,
            "confidence_grid": list(threshold_grid()),
            "prediction_confidence_floor": CONFIDENCE_FLOOR,
            "nms_iou": NMS_IOU,
            "match_iou": MATCH_IOU,
            "domains": list(DOMAINS),
        },
        "candidate_decision": {
            "selected": "C_full_mixed_source_yolo26n_960",
            "rejected": "D_exposure_matched_mixed_yolo26n_960",
            "reason": "C has higher mAP50-95 in both domains at identical inference cost.",
        },
        "final_model": final_model,
        "production_config": production,
        "reference_metrics": reference,
        "threshold_sweep": sweep,
        "policy_selection": policy,
        "plots": plots,
    }
    (DOCS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_calibration_report(metrics)
    print(
        json.dumps(
            {
                "selected_model": metrics["candidate_decision"]["selected"],
                "weight_sha256": final_model["weights_sha256"],
                "policy": policy["chosen_policy_type"],
                "global_threshold": policy["chosen_global_threshold"],
                "class_thresholds": policy["chosen_class_thresholds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
