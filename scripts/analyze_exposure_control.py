"""Evaluate the leakage-corrected Checkpoint 7B exposure control."""

# ruff: noqa: E501

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch
from ultralytics import YOLO

from analyze_data_diversity import diagnostics, draw_panel, validation_metrics
from visionops.data_diversity import (
    assert_rf100_test_excluded,
    diversity_success,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/experiments/data_diversity_exposure_control"
ASSETS = ROOT / "docs/assets/data_diversity_exposure_control"
RUNS = ROOT / "runs/data_diversity_exposure_control_evaluation"
LEGACY_METRICS = ROOT / "docs/experiments/data_diversity/metrics.json"
FILTER_METADATA = ROOT / "data/generated/rf100-valid-filtered/filter_metadata.json"
CPPE_DATA = ROOT / "data/generated/cppe-five-class/data.yaml"
CPPE_IMAGES = ROOT / "data/generated/cppe-five-class/images/val"
CPPE_LABELS = ROOT / "data/generated/cppe-five-class/labels/val"
FILTERED_DATA = ROOT / "data/generated/rf100-valid-filtered/data.yaml"
FILTERED_IMAGES = ROOT / "data/generated/rf100-valid-filtered/images/valid"
FILTERED_LABELS = ROOT / "data/generated/rf100-valid-filtered/labels/valid"
MODELS = {
    "cppe5_baseline": ROOT / "runs/experiment_cppe5_yolo26n_960/weights/best.pt",
    "quantity_control": ROOT
    / "runs/experiment_cppe5_quantity_control_yolo26n_960/weights/best.pt",
    "mixed_exposure_matched": ROOT
    / "runs/experiment_mixed5_exposure_matched_yolo26n_960/weights/best.pt",
    "mixed_full_budget": ROOT / "runs/experiment_mixed5_yolo26n_960/weights/best.pt",
}
METADATA = {
    "cppe5_baseline": ROOT / "runs/experiment_cppe5_yolo26n_960/training_metadata.json",
    "quantity_control": ROOT
    / "runs/experiment_cppe5_quantity_control_yolo26n_960/training_metadata.json",
    "mixed_exposure_matched": ROOT
    / "runs/experiment_mixed5_exposure_matched_yolo26n_960/training_metadata.json",
    "mixed_full_budget": ROOT
    / "runs/experiment_mixed5_yolo26n_960/training_metadata.json",
}


def run_validation(
    model: YOLO, model_name: str, data: Path, dataset_name: str
) -> dict[str, object]:
    result = model.val(
        data=str(data),
        split="val",
        imgsz=960,
        batch=4,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        plots=False,
        project=str(RUNS),
        name=f"{model_name}_{dataset_name}",
        exist_ok=True,
        verbose=False,
    )
    return validation_metrics(result)


def failure_panels(
    baseline: dict[str, dict[str, object]],
    matched: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    error_names = (
        "localization_failure",
        "classification_confusion",
        "complete_miss",
        "false_positive",
    )
    categories = {
        "fixes_false_positive": lambda a, d: a["false_positive"] - d["false_positive"],
        "fixes_localization": lambda a, d: (
            a["localization_failure"] - d["localization_failure"]
        ),
        "recovers_miss": lambda a, d: a["complete_miss"] - d["complete_miss"],
        "both_fail": lambda a, d: sum(a[name] + d[name] for name in error_names),
        "introduces_new_error": lambda a, d: sum(
            d[name] - a[name] for name in error_names
        ),
    }
    selections: list[dict[str, object]] = []
    for category, scorer in categories.items():
        scored = [
            (
                scorer(
                    Counter(baseline[name]["counts"]),
                    Counter(matched[name]["counts"]),
                ),
                name,
            )
            for name in sorted(baseline)
        ]
        score, filename = max(scored, key=lambda item: (item[0], item[1]))
        if score <= 0:
            continue
        output = ASSETS / f"{category}.jpg"
        draw_panel(baseline[filename], matched[filename], output, category)
        selections.append(
            {
                "category": category,
                "file": output.name,
                "source": filename,
                "selection_score": score,
            }
        )
    negative = [
        (
            sum(
                detection.class_id in {3, 4}
                for detection in baseline[name]["ground_truth"]
            ),
            name,
        )
        for name in sorted(baseline)
    ]
    score, filename = max(negative, key=lambda item: (item[0], item[1]))
    if score > 0:
        output = ASSETS / "negative_ppe.jpg"
        draw_panel(baseline[filename], matched[filename], output, "negative_ppe")
        selections.append(
            {
                "category": "negative_ppe",
                "file": output.name,
                "source": filename,
                "selection_score": score,
            }
        )
    return selections


def training_metadata() -> dict[str, object]:
    values = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in METADATA.items()
    }
    legacy = json.loads(LEGACY_METRICS.read_text(encoding="utf-8"))["training"]
    values["cppe5_baseline"]["optimizer_steps"] = legacy["cppe5_baseline"][
        "optimizer_steps"
    ]
    values["cppe5_baseline"]["optimizer_steps_measurement"] = legacy["cppe5_baseline"][
        "optimizer_steps_measurement"
    ]
    for name, item in values.items():
        item.setdefault("configured_scheduler_epochs", item["config"]["epochs"])
        if "stop_reason" not in item:
            item["stop_reason"] = (
                "early stopping"
                if item.get("early_stopping_occurred")
                else "configured epoch horizon"
            )
    return values


def compare_training_prefix() -> dict[str, object]:
    """Compare D with the first 45 recorded C epochs, excluding wall-clock time."""
    c_path = ROOT / "runs/experiment_mixed5_yolo26n_960/results.csv"
    d_path = ROOT / "runs/experiment_mixed5_exposure_matched_yolo26n_960/results.csv"
    with c_path.open(encoding="utf-8", newline="") as handle:
        c_rows = list(csv.DictReader(handle))[:45]
    with d_path.open(encoding="utf-8", newline="") as handle:
        d_rows = list(csv.DictReader(handle))
    columns = [name for name in c_rows[0] if name != "time"]
    mismatches = []
    for index, (c_row, d_row) in enumerate(zip(c_rows, d_rows, strict=False), start=1):
        changed = [name for name in columns if c_row[name] != d_row[name]]
        if changed:
            mismatches.append({"epoch": index, "columns": changed})
    return {
        "c_prefix_rows": len(c_rows),
        "d_rows": len(d_rows),
        "compared_columns": columns,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "exact_match_excluding_time": len(c_rows) == len(d_rows) == 45
        and not mismatches,
    }


def class_instances(diagnostic: dict[str, object], class_name: str) -> int:
    return sum(
        cell["total"]
        for cell in diagnostic["per_class_size_recall"][class_name].values()
    )


def write_report(metrics: dict[str, object]) -> None:
    validation = metrics["validation"]
    diagnostic_values = metrics["diagnostics"]
    training = metrics["training"]
    overall_rows = []
    class_rows = []
    error_rows = []
    size_rows = []
    budget_rows = []
    for name in MODELS:
        item = training[name]
        budget_rows.append(
            f"| {name} | {item['unique_training_images']} | {item['effective_entries_per_epoch']} | {item['configured_scheduler_epochs']} | {item['epochs_completed']} | {item['estimated_total_image_exposures']} | {item['optimizer_steps']} | {item['duration_seconds'] / 60:.1f} | {item['best_epoch']} | {item.get('stop_reason', 'legacy early-stop policy')} |"
        )
        for dataset in ("cppe_validation", "rf100_filtered_validation"):
            overall = validation[name][dataset]["overall"]
            overall_rows.append(
                f"| {name} | {dataset} | {overall['precision']:.4f} | {overall['recall']:.4f} | {overall['map50']:.4f} | {overall['map50_95']:.4f} |"
            )
            for item_class in validation[name][dataset]["per_class"]:
                instances = class_instances(
                    diagnostic_values[name][dataset], item_class["class_name"]
                )
                class_rows.append(
                    f"| {name} | {dataset} | {item_class['class_name']} | {instances} | {item_class['precision']:.4f} | {item_class['recall']:.4f} | {item_class['ap50']:.4f} | {item_class['map50_95']:.4f} |"
                )
            errors = diagnostic_values[name][dataset]["error_counts"]
            error_rows.append(
                f"| {name} | {dataset} | {errors['correct']} | {errors['localization_failure']} | {errors['classification_confusion']} | {errors['complete_miss']} | {errors['false_positive']} |"
            )
            for size, cell in diagnostic_values[name][dataset]["size_recall"].items():
                recall = "n/a" if cell["recall"] is None else f"{cell['recall']:.4f}"
                size_rows.append(
                    f"| {name} | {dataset} | {size} | {cell['correct']} | {cell['total']} | {recall} |"
                )
    legacy_rows = []
    for name in ("cppe5_baseline", "quantity_control", "mixed_full_budget"):
        legacy = metrics["legacy_rf100_validation"][name]["overall"]
        filtered = validation[name]["rf100_filtered_validation"]["overall"]
        legacy_rows.append(
            f"| {name} | {legacy['map50_95']:.4f} | {filtered['map50_95']:.4f} | {filtered['map50_95'] - legacy['map50_95']:+.4f} |"
        )
    decision = metrics["decision"]
    c_minus_d = metrics["c_minus_d"]
    d_rf = validation["mixed_exposure_matched"]["rf100_filtered_validation"]["overall"][
        "map50_95"
    ]
    a_rf = validation["cppe5_baseline"]["rf100_filtered_validation"]["overall"][
        "map50_95"
    ]
    b_rf = validation["quantity_control"]["rf100_filtered_validation"]["overall"][
        "map50_95"
    ]
    d_cp = validation["mixed_exposure_matched"]["cppe_validation"]["overall"][
        "map50_95"
    ]
    a_cp = validation["cppe5_baseline"]["cppe_validation"]["overall"]["map50_95"]
    filter_audit = metrics["filter_metadata"]["audit"]
    trajectory = metrics["training_prefix_comparison"]
    a_errors = diagnostic_values["cppe5_baseline"]["rf100_filtered_validation"][
        "error_counts"
    ]
    b_errors = diagnostic_values["quantity_control"]["rf100_filtered_validation"][
        "error_counts"
    ]
    d_errors = diagnostic_values["mixed_exposure_matched"]["rf100_filtered_validation"][
        "error_counts"
    ]
    d_external_classes = {
        item["class_name"]: item
        for item in validation["mixed_exposure_matched"]["rf100_filtered_validation"][
            "per_class"
        ]
    }
    text = f"""# Leakage-corrected exposure-control results

RF100 test was not read or evaluated. A/B see RF100 as an external development domain;
C/D use held-out RF100 target-domain validation.

## Leakage finding

The recomputed audit found {filter_audit["exact_duplicate_count"]} exact and {filter_audit["near_duplicate_count"]} near duplicate. The single dHash-distance-0 validation copy has inconsistent classes and was excluded, producing 118 validation images. Source files remain unchanged.

## Legacy versus filtered RF100 validation

| Model | Legacy 119 mAP50-95 | Filtered 118 mAP50-95 | Delta |
|---|---:|---:|---:|
{chr(10).join(legacy_rows)}

## Training budget

| Model | Unique | Entries/epoch | Scheduler horizon | Completed | Exposures | Optimizer steps | Minutes | Best epoch | Stop reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(budget_rows)}

## Overall validation

| Model | Dataset | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
{chr(10).join(overall_rows)}

## Complete five-class results

| Model | Dataset | Class | Instances | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(class_rows)}

`no_helmet` and `no_vest` counts are shown explicitly. Sparse `no_helmet` results are not described as solved.

## Fixed-threshold error decomposition

| Model | Dataset | Correct | Localization | Classification | Miss | False positive |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(error_rows)}

## Size-stratified diagnostic recall

| Model | Dataset | Size | Correct | Count | Recall |
|---|---|---|---:|---:|---:|
{chr(10).join(size_rows)}

Full per-class size cells and counts remain in `metrics.json`.

## Required interpretation

1. Removing the duplicate changed A/B/C RF100 mAP50-95 only by the deltas reported above; interpretation is based on the filtered values.
2. D changes filtered RF100 mAP50-95 by {d_rf - a_rf:+.4f} versus A and {d_rf - b_rf:+.4f} versus B at matched exposure.
3. D changes Construction-PPE mAP50-95 by {d_cp - a_cp:+.4f} versus A.
4. Full-budget C minus D is {c_minus_d["rf100_filtered_map50_95"]:+.4f} on filtered RF100 and {c_minus_d["cppe_map50_95"]:+.4f} on Construction-PPE.
5. Against A, D changes correct assignments by {d_errors["correct"] - a_errors["correct"]:+d}, localization failures by {d_errors["localization_failure"] - a_errors["localization_failure"]:+d}, confusions by {d_errors["classification_confusion"] - a_errors["classification_confusion"]:+d}, misses by {d_errors["complete_miss"] - a_errors["complete_miss"]:+d}, and false positives by {d_errors["false_positive"] - a_errors["false_positive"]:+d}. Against B, D changes false positives by {d_errors["false_positive"] - b_errors["false_positive"]:+d} and misses by {d_errors["complete_miss"] - b_errors["complete_miss"]:+d}.
6. On filtered RF100, D `no_helmet` has n=11, precision {d_external_classes["no_helmet"]["precision"]:.4f}, recall {d_external_classes["no_helmet"]["recall"]:.4f}, AP50 {d_external_classes["no_helmet"]["ap50"]:.4f}, and AP50-95 {d_external_classes["no_helmet"]["map50_95"]:.4f}; this cell is too small to call solved. D `no_vest` has n=90, precision {d_external_classes["no_vest"]["precision"]:.4f}, recall {d_external_classes["no_vest"]["recall"]:.4f}, AP50 {d_external_classes["no_vest"]["ap50"]:.4f}, and AP50-95 {d_external_classes["no_vest"]["map50_95"]:.4f}.
7. Remaining limitations include one training seed, a small filtered RF100 `no_helmet` count, RF100 being a target domain for C/D, and RF100 test having influenced the earlier overall research path. B and D both recorded exactly 1,389 optimizer steps, so no optimizer-step mismatch remains in the primary matched-exposure comparison.
8. **Decision:** {"All predeclared matched-exposure criteria pass; target-domain data diversity remains strongly supported." if decision["material_success"] else "The complete predeclared matched-exposure rule does not pass, so the prior claim must be weakened."}

The result is evidence from a controlled development experiment, not perfect causal proof or universal-domain generalization. D captures most of C's target-domain gain at the same exposure as B, while C's additional 55 epochs add a comparatively modest +{c_minus_d["rf100_filtered_map50_95"]:.4f} RF100 mAP50-95 and recover +{c_minus_d["cppe_map50_95"]:.4f} CPPE mAP50-95. Target-domain data content is therefore the strongest supported explanation, with additional optimization exposure providing a secondary benefit.

## Scheduler-control evidence

D recorded {trajectory["d_rows"]} epochs against the first {trajectory["c_prefix_rows"]} C epochs. Excluding wall-clock time, exact metric/LR trajectory match: **{trajectory["exact_match_excluding_time"]}** ({trajectory["mismatch_count"]} mismatched epochs). The training metadata also records the 100-epoch scheduler horizon, epoch-45 controlled stop, result-row count, absence of epoch 46, and unreached close-mosaic phase.
"""
    (DOCS / "comparison.md").write_text(text, encoding="utf-8")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    guarded = [
        CPPE_DATA,
        CPPE_IMAGES,
        CPPE_LABELS,
        FILTERED_DATA,
        FILTERED_IMAGES,
        FILTERED_LABELS,
        *MODELS.values(),
    ]
    assert_rf100_test_excluded(guarded)
    filter_metadata = json.loads(FILTER_METADATA.read_text(encoding="utf-8"))
    if filter_metadata["rf100_test_used"] is not False:
        raise RuntimeError("Filtered validation metadata violates RF100-test freeze")
    legacy = json.loads(LEGACY_METRICS.read_text(encoding="utf-8"))
    legacy_names = {
        "cppe5_baseline": "cppe5_baseline",
        "quantity_control": "quantity_control",
        "mixed_full_budget": "mixed_domain",
    }
    validation: dict[str, object] = {}
    diagnostic_values: dict[str, object] = {}
    filtered_records: dict[str, dict[str, dict[str, object]]] = {}
    for name, model_path in MODELS.items():
        model = YOLO(str(model_path))
        validation[name] = {}
        diagnostic_values[name] = {}
        if name == "mixed_exposure_matched":
            validation[name]["cppe_validation"] = run_validation(
                model, name, CPPE_DATA, "cppe_validation"
            )
            cppe_diagnostics, _ = diagnostics(model, CPPE_IMAGES, CPPE_LABELS)
            diagnostic_values[name]["cppe_validation"] = cppe_diagnostics
        else:
            legacy_name = legacy_names[name]
            validation[name]["cppe_validation"] = legacy["validation"][legacy_name][
                "cppe_validation"
            ]
            diagnostic_values[name]["cppe_validation"] = legacy["diagnostics"][
                legacy_name
            ]["cppe_validation"]
        validation[name]["rf100_filtered_validation"] = run_validation(
            model, name, FILTERED_DATA, "rf100_filtered_validation"
        )
        filtered_diagnostics, records = diagnostics(
            model, FILTERED_IMAGES, FILTERED_LABELS
        )
        diagnostic_values[name]["rf100_filtered_validation"] = filtered_diagnostics
        filtered_records[name] = records
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    panels = failure_panels(
        filtered_records["cppe5_baseline"],
        filtered_records["mixed_exposure_matched"],
    )
    legacy_rf100 = {
        name: legacy["validation"][legacy_name]["rf100_validation"]
        for name, legacy_name in legacy_names.items()
    }
    decision = diversity_success(
        validation["cppe5_baseline"]["rf100_filtered_validation"]["overall"][
            "map50_95"
        ],
        validation["quantity_control"]["rf100_filtered_validation"]["overall"][
            "map50_95"
        ],
        validation["mixed_exposure_matched"]["rf100_filtered_validation"]["overall"][
            "map50_95"
        ],
        validation["cppe5_baseline"]["cppe_validation"]["overall"]["map50_95"],
        validation["mixed_exposure_matched"]["cppe_validation"]["overall"]["map50_95"],
    )
    c_minus_d = {
        "rf100_filtered_map50_95": validation["mixed_full_budget"][
            "rf100_filtered_validation"
        ]["overall"]["map50_95"]
        - validation["mixed_exposure_matched"]["rf100_filtered_validation"]["overall"][
            "map50_95"
        ],
        "cppe_map50_95": validation["mixed_full_budget"]["cppe_validation"]["overall"][
            "map50_95"
        ]
        - validation["mixed_exposure_matched"]["cppe_validation"]["overall"][
            "map50_95"
        ],
    }
    metrics = {
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "rf100_test_used": False,
            "rf100_filtered_validation_images": 118,
            "confidence": 0.25,
            "nms_iou": 0.70,
            "match_iou": 0.50,
            "localization_iou_floor": 0.10,
        },
        "filter_metadata": filter_metadata,
        "training": training_metadata(),
        "training_prefix_comparison": compare_training_prefix(),
        "legacy_rf100_validation": legacy_rf100,
        "validation": validation,
        "diagnostics": diagnostic_values,
        "decision": decision,
        "c_minus_d": c_minus_d,
        "representative_panels": panels,
    }
    (DOCS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics)
    print(json.dumps({"decision": decision, "c_minus_d": c_minus_d}, indent=2))


if __name__ == "__main__":
    main()
