"""Evaluate the three frozen Checkpoint 7 models without using RF100 test."""

# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

from visionops.data_diversity import (
    FIVE_CLASS_NAMES,
    assert_rf100_test_excluded,
    diversity_success,
    reconstruct_optimizer_steps,
)
from visionops.detection_analysis import SIZE_BINS, Detection, area_bin
from visionops.domain_shift import (
    LOCALIZATION_IOU_FLOOR,
    MATCH_IOU,
    decompose_errors,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/generated/data-diversity-manifest.json"
DOCS = ROOT / "docs/experiments/data_diversity"
ASSETS = ROOT / "docs/assets/data_diversity"
RUNS = ROOT / "runs/data_diversity_evaluation"
CPPE = ROOT / "data/generated/cppe-five-class"
RF100 = ROOT / "data/generated/rf100-five-class"
MODELS = {
    "cppe5_baseline": ROOT / "runs/experiment_cppe5_yolo26n_960/weights/best.pt",
    "quantity_control": ROOT
    / "runs/experiment_cppe5_quantity_control_yolo26n_960/weights/best.pt",
    "mixed_domain": ROOT / "runs/experiment_mixed5_yolo26n_960/weights/best.pt",
}
METADATA = {
    name: path.parents[1] / "training_metadata.json" for name, path in MODELS.items()
}
DATASETS = {
    "cppe_validation": (CPPE / "data.yaml", CPPE / "images/val", CPPE / "labels/val"),
    "rf100_validation": (
        RF100 / "data.yaml",
        RF100 / "images/valid",
        RF100 / "labels/valid",
    ),
}
CONFIDENCE = 0.25
NMS_IOU = 0.70


def rounded(value: object) -> float:
    return round(float(value), 10)


def validation_metrics(result: object) -> dict[str, object]:
    box = result.box
    indices = [int(value) for value in box.ap_class_index]
    by_index = {class_id: position for position, class_id in enumerate(indices)}
    per_class = []
    for class_id, class_name in enumerate(FIVE_CLASS_NAMES):
        position = by_index[class_id]
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": rounded(box.p[position]),
                "recall": rounded(box.r[position]),
                "ap50": rounded(box.ap50[position]),
                "map50_95": rounded(box.ap[position].mean()),
            }
        )
    return {
        "overall": {
            "precision": rounded(box.mp),
            "recall": rounded(box.mr),
            "map50": rounded(box.map50),
            "map50_95": rounded(box.map),
        },
        "per_class": per_class,
    }


def read_labels(path: Path) -> list[Detection]:
    detections = []
    for line in path.read_text(encoding="utf-8").splitlines():
        class_text, x_text, y_text, width_text, height_text = line.split()
        x, y, width, height = map(float, (x_text, y_text, width_text, height_text))
        detections.append(
            Detection(
                int(class_text),
                (x - width / 2, y - height / 2, x + width / 2, y + height / 2),
            )
        )
    return detections


def predictions(result: object) -> list[Detection]:
    boxes = result.boxes
    if boxes is None:
        return []
    return [
        Detection(
            int(boxes.cls[index].item()),
            tuple(float(value) for value in boxes.xyxyn[index].tolist()),
            float(boxes.conf[index].item()),
        )
        for index in range(len(boxes))
    ]


def diagnostics(
    model: YOLO, images: Path, labels: Path
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    results = model.predict(
        source=str(images),
        imgsz=960,
        batch=4,
        conf=CONFIDENCE,
        iou=NMS_IOU,
        device=0 if torch.cuda.is_available() else "cpu",
        stream=True,
        verbose=False,
    )
    counts: Counter[str] = Counter()
    size_counts: dict[str, Counter[str]] = defaultdict(Counter)
    class_size_counts: dict[str, Counter[str]] = defaultdict(Counter)
    records: dict[str, dict[str, object]] = {}
    for result in results:
        image_path = Path(result.path)
        targets = read_labels(labels / f"{image_path.stem}.txt")
        detected = predictions(result)
        decomposition = decompose_errors(targets, detected)
        image_counts: Counter[str] = Counter()
        for outcome in decomposition.outcomes:
            target = targets[outcome.ground_truth_index]
            counts[outcome.category] += 1
            image_counts[outcome.category] += 1
            size = area_bin(target.relative_area)
            size_counts[size]["total"] += 1
            class_size_counts[f"{target.class_id}|{size}"]["total"] += 1
            if outcome.category == "correct":
                size_counts[size]["correct"] += 1
                class_size_counts[f"{target.class_id}|{size}"]["correct"] += 1
        false_positives = len(decomposition.false_positive_indices)
        counts["false_positive"] += false_positives
        image_counts["false_positive"] = false_positives
        records[image_path.name] = {
            "path": image_path,
            "ground_truth": targets,
            "predictions": detected,
            "counts": dict(image_counts),
        }
    size_recall = {
        size: {
            "correct": size_counts[size]["correct"],
            "total": size_counts[size]["total"],
            "recall": (
                size_counts[size]["correct"] / size_counts[size]["total"]
                if size_counts[size]["total"]
                else None
            ),
        }
        for size in SIZE_BINS
    }
    per_class_size = {}
    for class_id, class_name in enumerate(FIVE_CLASS_NAMES):
        per_class_size[class_name] = {}
        for size in SIZE_BINS:
            cell = class_size_counts[f"{class_id}|{size}"]
            per_class_size[class_name][size] = {
                "correct": cell["correct"],
                "total": cell["total"],
                "recall": cell["correct"] / cell["total"] if cell["total"] else None,
            }
    ordered_categories = (
        "correct",
        "localization_failure",
        "classification_confusion",
        "complete_miss",
        "false_positive",
    )
    return (
        {
            "error_counts": {name: counts[name] for name in ordered_categories},
            "size_recall": size_recall,
            "per_class_size_recall": per_class_size,
        },
        records,
    )


def draw_panel(
    record_a: dict[str, object],
    record_c: dict[str, object],
    output: Path,
    title: str,
) -> None:
    panels = []
    for model_name, record in (("baseline", record_a), ("mixed", record_c)):
        with Image.open(record["path"]) as opened:
            panel = ImageOps.exif_transpose(opened).convert("RGB")
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel.width, 30), fill="white")
        draw.text(
            (6, 8), f"{title} | {model_name} | GT blue, prediction red", fill="black"
        )
        for detection in record["ground_truth"]:
            x1, y1, x2, y2 = detection.xyxy
            coordinates = (
                x1 * panel.width,
                y1 * panel.height,
                x2 * panel.width,
                y2 * panel.height,
            )
            draw.rectangle(coordinates, outline="blue", width=3)
            draw.text(
                (coordinates[0], max(30, coordinates[1])),
                FIVE_CLASS_NAMES[detection.class_id],
                fill="blue",
            )
        for detection in record["predictions"]:
            x1, y1, x2, y2 = detection.xyxy
            coordinates = (
                x1 * panel.width,
                y1 * panel.height,
                x2 * panel.width,
                y2 * panel.height,
            )
            draw.rectangle(coordinates, outline="red", width=2)
            draw.text(
                (coordinates[0], max(30, coordinates[1] - 12)),
                f"{FIVE_CLASS_NAMES[detection.class_id]} {detection.confidence:.2f}",
                fill="red",
            )
        panels.append(panel)
    height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), height), "white")
    offset = 0
    for panel in panels:
        canvas.paste(panel, (offset, 0))
        offset += panel.width
    canvas.thumbnail((1800, 1200))
    canvas.save(output, quality=88)


def create_failure_panels(
    baseline: dict[str, dict[str, object]], mixed: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    categories = {
        "fixes_false_positive": lambda a, c: a["false_positive"] - c["false_positive"],
        "fixes_localization": lambda a, c: (
            a["localization_failure"] - c["localization_failure"]
        ),
        "recovers_miss": lambda a, c: a["complete_miss"] - c["complete_miss"],
        "both_fail": lambda a, c: (
            sum(
                a[name]
                for name in (
                    "localization_failure",
                    "classification_confusion",
                    "complete_miss",
                )
            )
            + sum(
                c[name]
                for name in (
                    "localization_failure",
                    "classification_confusion",
                    "complete_miss",
                )
            )
        ),
        "introduces_new_error": lambda a, c: (
            sum(
                c[name]
                for name in (
                    "localization_failure",
                    "classification_confusion",
                    "complete_miss",
                    "false_positive",
                )
            )
            - sum(
                a[name]
                for name in (
                    "localization_failure",
                    "classification_confusion",
                    "complete_miss",
                    "false_positive",
                )
            )
        ),
        "negative_ppe": None,
    }
    outputs = []
    for category, score_function in categories.items():
        scored = []
        for filename in sorted(baseline):
            a_counts = Counter(baseline[filename]["counts"])
            c_counts = Counter(mixed[filename]["counts"])
            if category == "negative_ppe":
                score = sum(
                    detection.class_id in {3, 4}
                    for detection in baseline[filename]["ground_truth"]
                )
            else:
                assert score_function is not None
                score = score_function(a_counts, c_counts)
            scored.append((score, filename))
        score, filename = max(scored, key=lambda item: (item[0], item[1]))
        output = ASSETS / f"{category}.jpg"
        draw_panel(baseline[filename], mixed[filename], output, category)
        outputs.append(
            {
                "category": category,
                "file": output.name,
                "source": filename,
                "selection_score": score,
            }
        )
    return outputs


def load_training_metadata() -> dict[str, object]:
    metadata = {}
    for name, path in METADATA.items():
        value = json.loads(path.read_text(encoding="utf-8"))
        if name == "cppe5_baseline" and value.get("optimizer_steps") == 0:
            value["optimizer_steps"] = reconstruct_optimizer_steps(
                value["effective_entries_per_epoch"],
                value["actual_batch"],
                value["epochs_completed"],
            )
            value["optimizer_steps_measurement"] = (
                "reconstructed from the recorded Ultralytics accumulation schedule; "
                "the original callback was registered too late"
            )
        else:
            value["optimizer_steps_measurement"] = "runtime callback"
        metadata[name] = value
    return metadata


def write_comparison(metrics: dict[str, object]) -> None:
    validation = metrics["validation"]
    diagnostics_data = metrics["diagnostics"]
    decision = metrics["success_criterion"]
    rows = []
    for model_name in MODELS:
        for dataset_name in DATASETS:
            overall = validation[model_name][dataset_name]["overall"]
            rows.append(
                f"| {model_name} | {dataset_name} | {overall['precision']:.4f} | {overall['recall']:.4f} | {overall['map50']:.4f} | {overall['map50_95']:.4f} |"
            )
    class_rows = []
    for model_name in MODELS:
        for dataset_name in DATASETS:
            for item in validation[model_name][dataset_name]["per_class"]:
                class_rows.append(
                    f"| {model_name} | {dataset_name} | {item['class_name']} | {item['precision']:.4f} | {item['recall']:.4f} | {item['ap50']:.4f} | {item['map50_95']:.4f} |"
                )
    error_rows = []
    for model_name in MODELS:
        for dataset_name in DATASETS:
            values = diagnostics_data[model_name][dataset_name]["error_counts"]
            error_rows.append(
                f"| {model_name} | {dataset_name} | {values['correct']} | {values['localization_failure']} | {values['classification_confusion']} | {values['complete_miss']} | {values['false_positive']} |"
            )
    training_rows = []
    for model_name in MODELS:
        item = metrics["training"][model_name]
        training_rows.append(
            f"| {model_name} | {item['unique_training_images']} | "
            f"{item['effective_entries_per_epoch']} | {item['epochs_completed']} | "
            f"{item['estimated_total_image_exposures']} | {item['optimizer_steps']} | "
            f"{item['duration_seconds'] / 60:.1f} |"
        )
    size_rows = []
    class_size_rows = []
    for model_name in MODELS:
        for dataset_name in DATASETS:
            values = diagnostics_data[model_name][dataset_name]
            for size, cell in values["size_recall"].items():
                recall = "n/a" if cell["recall"] is None else f"{cell['recall']:.4f}"
                size_rows.append(
                    f"| {model_name} | {dataset_name} | {size} | "
                    f"{cell['correct']} | {cell['total']} | {recall} |"
                )
            for class_name, sizes in values["per_class_size_recall"].items():
                for size, cell in sizes.items():
                    recall = (
                        "n/a" if cell["recall"] is None else f"{cell['recall']:.4f}"
                    )
                    class_size_rows.append(
                        f"| {model_name} | {dataset_name} | {class_name} | {size} | "
                        f"{cell['correct']} | {cell['total']} | {recall} |"
                    )
    a_rf = validation["cppe5_baseline"]["rf100_validation"]["overall"]["map50_95"]
    b_rf = validation["quantity_control"]["rf100_validation"]["overall"]["map50_95"]
    c_rf = validation["mixed_domain"]["rf100_validation"]["overall"]["map50_95"]
    a_cp = validation["cppe5_baseline"]["cppe_validation"]["overall"]["map50_95"]
    c_cp = validation["mixed_domain"]["cppe_validation"]["overall"]["map50_95"]
    baseline_classes = {
        item["class_name"]: item["map50_95"]
        for item in validation["cppe5_baseline"]["rf100_validation"]["per_class"]
    }
    mixed_classes = {
        item["class_name"]: item["map50_95"]
        for item in validation["mixed_domain"]["rf100_validation"]["per_class"]
    }
    class_deltas = sorted(
        (
            (mixed_classes[name] - baseline_classes[name], name)
            for name in FIVE_CLASS_NAMES
        ),
        reverse=True,
    )
    class_summary = ", ".join(f"{name} {change:+.4f}" for change, name in class_deltas)
    baseline_errors = diagnostics_data["cppe5_baseline"]["rf100_validation"][
        "error_counts"
    ]
    mixed_errors = diagnostics_data["mixed_domain"]["rf100_validation"]["error_counts"]
    error_summary = ", ".join(
        f"{name} {mixed_errors[name] - baseline_errors[name]:+d}"
        for name in (
            "false_positive",
            "localization_failure",
            "classification_confusion",
            "complete_miss",
            "correct",
        )
    )
    regressions = [
        f"{name} {change:+.4f}" for change, name in class_deltas if change < 0
    ]
    text = f"""# Controlled training-data diversity results

RF100 test remained frozen and was not read. RF100 validation is OOD-development data.

## Derived datasets

| Dataset | Split | Images | Retained boxes | Removed out-of-task boxes |
|---|---|---:|---:|---:|
| Construction-PPE | train | {metrics["data_derivation"]["construction_ppe"]["train"]["images"]} | {metrics["data_derivation"]["construction_ppe"]["train"]["retained_boxes"]} | {metrics["data_derivation"]["construction_ppe"]["train"]["removed_boxes"]} |
| Construction-PPE | validation | {metrics["data_derivation"]["construction_ppe"]["val"]["images"]} | {metrics["data_derivation"]["construction_ppe"]["val"]["retained_boxes"]} | {metrics["data_derivation"]["construction_ppe"]["val"]["removed_boxes"]} |
| RF100 | train | {metrics["data_derivation"]["rf100"]["train"]["images"]} | {metrics["data_derivation"]["rf100"]["train"]["retained_boxes"]} | 0 |
| RF100 | validation | {metrics["data_derivation"]["rf100"]["valid"]["images"]} | {metrics["data_derivation"]["rf100"]["valid"]["retained_boxes"]} | 0 |

## Optimization budget

| Condition | Unique images | Entries/epoch | Epochs | Image exposures | Optimizer steps | Minutes |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(training_rows)}

The A-run optimizer-step count is reconstructed from the recorded deterministic
Ultralytics accumulation schedule because its callback was initially registered too late.
The B and C counts are measured by runtime callbacks.

## Overall validation

| Model | Dataset | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Complete per-class validation

| Model | Dataset | Class | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(class_rows)}

## Fixed-threshold error decomposition

Confidence {CONFIDENCE}, NMS IoU {NMS_IOU}, match IoU {MATCH_IOU}, localization floor {LOCALIZATION_IOU_FLOOR}.

| Model | Dataset | Correct | Localization | Classification | Miss | False positive |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(error_rows)}

Size-stratified recall, including every per-class cell count, is recorded in `metrics.json`.

## Size-stratified diagnostic recall

| Model | Dataset | Size | Correct | GT count | Recall |
|---|---|---|---:|---:|---:|
{chr(10).join(size_rows)}

### Per-class size cells

Small cells are retained with their counts and must not be overinterpreted.

| Model | Dataset | Class | Size | Correct | GT count | Recall |
|---|---|---|---|---:|---:|---:|
{chr(10).join(class_size_rows)}

## Required conclusions

1. **Does mixed-domain training improve RF100 validation?** {"Yes" if c_rf > a_rf else "No"}: mAP50-95 changed from {a_rf:.4f} to {c_rf:.4f} ({c_rf - a_rf:+.4f}).
2. **Does it preserve Construction-PPE validation?** {"Yes under the predeclared tolerance" if c_cp - a_cp >= -0.03 else "No"}: mAP50-95 changed from {a_cp:.4f} to {c_cp:.4f} ({c_cp - a_cp:+.4f}).
3. **Does the quantity control achieve similar improvements?** Its RF100 result is {b_rf:.4f}, a change of {b_rf - a_rf:+.4f} from A; mixed exceeds it by {c_rf - b_rf:+.4f}.
4. **Is improvement attributable to diversity rather than only exposure?** {"The numerical control supports diversity" if decision["diversity_gain_pass"] else "Not under the predeclared numerical margin"}. However, B early-stopped after 45 epochs while C ran 100, so C had more total exposures and optimizer steps; this residual confound prevents a strict causal claim that all gain came from diversity alone.
5. **Which failure types improve?** RF100 mixed-minus-baseline count changes are: {error_summary}. Negative error changes are improvements; positive correct change is an improvement.
6. **Which classes benefit most?** RF100 AP50-95 changes, greatest first: {class_summary}.
7. **What regressions appear?** {"RF100 per-class regressions: " + ", ".join(regressions) if regressions else "No RF100 class has a negative AP50-95 change."} In-domain and size-cell changes remain visible above.
8. **Is this strong enough for the final model stage?** {"Yes: all predeclared material-success conditions pass." if decision["material_success"] else "No: the complete predeclared numerical rule does not pass."}

The representative panels are deterministically selected from RF100 validation only. Blue boxes are ground truth and red boxes are predictions. Selection scores are recorded in `metrics.json`; a non-positive score means the requested failure category had no strict improvement/regression example and the closest deterministic case is shown rather than overstated.

The numerical rule and qualitative diagnostics are interpreted separately; disagreement is
reported rather than used to alter the predeclared threshold.

The quantity control matched the mixed condition's entries per epoch and nearly matched
the A baseline's total exposure, but early stopping prevented total-budget matching to C.
This was the predeclared residual confound; training was not extended after results.
"""
    (DOCS / "comparison.md").write_text(text, encoding="utf-8")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    guarded = [
        path for _, images, labels in DATASETS.values() for path in (images, labels)
    ]
    assert_rf100_test_excluded(guarded)
    validation: dict[str, object] = {}
    diagnostic_results: dict[str, object] = {}
    rf100_records: dict[str, dict[str, dict[str, object]]] = {}
    for model_name, model_path in MODELS.items():
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        model = YOLO(str(model_path))
        validation[model_name] = {}
        diagnostic_results[model_name] = {}
        for dataset_name, (data_yaml, images, labels) in DATASETS.items():
            assert_rf100_test_excluded([data_yaml, images, labels])
            result = model.val(
                data=str(data_yaml),
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
            validation[model_name][dataset_name] = validation_metrics(result)
            values, records = diagnostics(model, images, labels)
            diagnostic_results[model_name][dataset_name] = values
            if dataset_name == "rf100_validation":
                rf100_records[model_name] = records
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    panels = create_failure_panels(
        rf100_records["cppe5_baseline"], rf100_records["mixed_domain"]
    )
    decision = diversity_success(
        validation["cppe5_baseline"]["rf100_validation"]["overall"]["map50_95"],
        validation["quantity_control"]["rf100_validation"]["overall"]["map50_95"],
        validation["mixed_domain"]["rf100_validation"]["overall"]["map50_95"],
        validation["cppe5_baseline"]["cppe_validation"]["overall"]["map50_95"],
        validation["mixed_domain"]["cppe_validation"]["overall"]["map50_95"],
    )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("rf100_test_used") is not False:
        raise RuntimeError("Generated-data manifest violates the RF100-test freeze")
    metrics = {
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "rf100_test_used": False,
            "confidence": CONFIDENCE,
            "nms_iou": NMS_IOU,
            "match_iou": MATCH_IOU,
            "localization_iou_floor": LOCALIZATION_IOU_FLOOR,
            "size_bins": {
                "tiny": "area < 0.001",
                "small": "0.001 <= area < 0.01",
                "large": "area >= 0.01",
            },
        },
        "training": load_training_metadata(),
        "data_derivation": manifest,
        "validation": validation,
        "diagnostics": diagnostic_results,
        "success_criterion": decision,
        "representative_panels": panels,
    }
    (DOCS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_comparison(metrics)
    print(json.dumps({"success_criterion": decision, "panels": panels}, indent=2))


if __name__ == "__main__":
    main()
