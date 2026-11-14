"""Evaluate frozen 960-pixel models on leakage-free RF100 Construction Safety."""

# ruff: noqa: E501

from __future__ import annotations

import json
import platform
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import torch
import ultralytics
from PIL import Image, ImageDraw
from ultralytics import YOLO

from visionops.detection_analysis import Detection, area_bin, match_detections
from visionops.external_data import INTERNAL_CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "data/external/generated/construction-safety-gsnvb-v2-remapped"
DOCS = ROOT / "docs/experiments/external_generalization"
ASSETS = ROOT / "docs/assets/external_generalization"
RUNS = ROOT / "runs/external_generalization"
CLASS_IDS = (0, 2, 5, 6, 11)
MODELS = {
    "yolo26n": ROOT / "runs/experiment_yolo26n_960/weights/best.pt",
    "yolo26s": ROOT / "runs/experiment_yolo26s_960/weights/best.pt",
}
CONFIDENCE = 0.25
MATCH_IOU = 0.5
NMS_IOU = 0.7


def rounded(value: float) -> float:
    return round(float(value), 10)


def validation_metrics(result: object) -> dict[str, object]:
    box = result.box
    indices = [int(value) for value in box.ap_class_index]
    per_class = []
    for position, class_id in enumerate(indices):
        if class_id not in CLASS_IDS:
            continue
        per_class.append(
            {
                "class_id": class_id,
                "class_name": INTERNAL_CLASS_NAMES[class_id],
                "precision": rounded(box.p[position]),
                "recall": rounded(box.r[position]),
                "map50": rounded(box.ap50[position]),
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
        "speed_ms_per_image": {
            key: rounded(value) for key, value in result.speed.items()
        },
    }


def run_validation(model: YOLO, model_name: str, scope: str) -> dict[str, object]:
    data = GENERATED / ("data.yaml" if scope == "test" else "data_all.yaml")
    result = model.val(
        data=str(data),
        split="test" if scope == "test" else "val",
        imgsz=960,
        batch=4,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        classes=list(CLASS_IDS),
        plots=False,
        project=str(RUNS),
        name=f"{model_name}_{scope}",
        exist_ok=True,
        verbose=False,
    )
    return validation_metrics(result)


def read_ground_truth(label_path: Path) -> list[Detection]:
    detections = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id_text, x_text, y_text, width_text, height_text = line.split()
        class_id = int(class_id_text)
        x, y, width, height = map(float, (x_text, y_text, width_text, height_text))
        detections.append(
            Detection(
                class_id,
                (x - width / 2, y - height / 2, x + width / 2, y + height / 2),
            )
        )
    return detections


def result_detections(result: object) -> list[Detection]:
    if result.boxes is None:
        return []
    boxes = result.boxes
    return [
        Detection(
            int(boxes.cls[index].item()),
            tuple(float(value) for value in boxes.xyxyn[index].tolist()),
            float(boxes.conf[index].item()),
        )
        for index in range(len(boxes))
        if int(boxes.cls[index].item()) in CLASS_IDS
    ]


def draw_panel(
    source: Path,
    ground_truth: list[Detection],
    predictions: dict[str, list[Detection]],
    output: Path,
    title: str,
) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    panels = []
    for name in ("yolo26n", "yolo26s"):
        panel = image.copy()
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel.width, 28), fill="white")
        draw.text((8, 7), f"{title} | {name} | GT=blue prediction=red", fill="black")
        for detection in ground_truth:
            x1, y1, x2, y2 = detection.xyxy
            coordinates = (
                x1 * panel.width,
                y1 * panel.height,
                x2 * panel.width,
                y2 * panel.height,
            )
            draw.rectangle(coordinates, outline="blue", width=3)
            draw.text(
                (coordinates[0], coordinates[1]),
                INTERNAL_CLASS_NAMES[detection.class_id],
                fill="blue",
            )
        for detection in predictions[name]:
            x1, y1, x2, y2 = detection.xyxy
            coordinates = (
                x1 * panel.width,
                y1 * panel.height,
                x2 * panel.width,
                y2 * panel.height,
            )
            draw.rectangle(coordinates, outline="red", width=2)
            draw.text(
                (coordinates[0], max(28, coordinates[1] - 12)),
                f"{INTERNAL_CLASS_NAMES[detection.class_id]} {detection.confidence:.2f}",
                fill="red",
            )
        panels.append(panel)
    canvas = Image.new("RGB", (panels[0].width * 2, panels[0].height), "white")
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (panels[0].width, 0))
    canvas.thumbnail((1800, 1200))
    canvas.save(output, quality=88)


def diagnostics(
    models: dict[str, YOLO],
) -> tuple[dict[str, object], list[dict[str, str]]]:
    image_paths = sorted((GENERATED / "images/test").glob("*"))
    predictions: dict[str, dict[str, list[Detection]]] = {}
    for model_name, model in models.items():
        results = model.predict(
            source=[str(path) for path in image_paths],
            imgsz=960,
            conf=CONFIDENCE,
            iou=NMS_IOU,
            device=0 if torch.cuda.is_available() else "cpu",
            batch=4,
            classes=list(CLASS_IDS),
            verbose=False,
        )
        predictions[model_name] = {
            Path(result.path).name: result_detections(result) for result in results
        }

    counts: dict[str, Counter[tuple[str, str]]] = {name: Counter() for name in models}
    candidates: dict[str, list[tuple[int, str]]] = defaultdict(list)
    records: dict[str, tuple[Path, list[Detection], dict[str, list[Detection]]]] = {}
    for image_path in image_paths:
        label_path = GENERATED / "labels/test" / f"{image_path.stem}.txt"
        ground_truth = read_ground_truth(label_path)
        image_predictions = {
            name: predictions[name][image_path.name] for name in models
        }
        records[image_path.name] = (image_path, ground_truth, image_predictions)
        model_stats: dict[str, tuple[int, int, int, set[int]]] = {}
        for model_name in models:
            predicted = image_predictions[model_name]
            matched = match_detections(ground_truth, predicted, MATCH_IOU)
            for index, detection in enumerate(ground_truth):
                key = (
                    INTERNAL_CLASS_NAMES[detection.class_id],
                    area_bin(detection.relative_area),
                )
                counts[model_name][(key[0], key[1] + "_gt")] += 1
                counts[model_name][("overall", key[1] + "_gt")] += 1
                if index in matched.matched_ground_truth_indices:
                    counts[model_name][(key[0], key[1] + "_matched")] += 1
                    counts[model_name][("overall", key[1] + "_matched")] += 1
            false_negatives = len(ground_truth) - len(matched.pairs)
            false_positives = len(predicted) - len(matched.pairs)
            missed_classes = {
                ground_truth[index].class_id
                for index in range(len(ground_truth))
                if index not in matched.matched_ground_truth_indices
            }
            model_stats[model_name] = (
                len(matched.pairs),
                false_negatives,
                false_positives,
                missed_classes,
            )
        worst_fn = max(model_stats[name][1] for name in models)
        worst_fp = max(model_stats[name][2] for name in models)
        total_matches = sum(model_stats[name][0] for name in models)
        candidates["correct"].append((total_matches, image_path.name))
        if worst_fn:
            candidates["false_negative"].append((worst_fn, image_path.name))
        if worst_fp:
            candidates["false_positive"].append((worst_fp, image_path.name))
        if any(6 in model_stats[name][3] for name in models):
            candidates["no_helmet_failure"].append((worst_fn, image_path.name))
        if any(11 in model_stats[name][3] for name in models):
            candidates["no_vest_failure"].append((worst_fn, image_path.name))
        tiny_or_small_missed = sum(
            1
            for detection in ground_truth
            if area_bin(detection.relative_area) in {"tiny", "small"}
        )
        if tiny_or_small_missed and worst_fn:
            candidates["small_object_failure"].append(
                (tiny_or_small_missed, image_path.name)
            )

    size_recall: dict[str, object] = {}
    for model_name in models:
        model_result: dict[str, object] = {}
        for class_name in (
            "overall",
            *(INTERNAL_CLASS_NAMES[index] for index in CLASS_IDS),
        ):
            model_result[class_name] = {}
            for size in ("tiny", "small", "large"):
                ground_truth_count = counts[model_name][(class_name, size + "_gt")]
                matched_count = counts[model_name][(class_name, size + "_matched")]
                model_result[class_name][size] = {
                    "ground_truth_count": ground_truth_count,
                    "matched_count": matched_count,
                    "recall": (
                        matched_count / ground_truth_count
                        if ground_truth_count
                        else None
                    ),
                }
        size_recall[model_name] = model_result

    ASSETS.mkdir(parents=True, exist_ok=True)
    examples = []
    for category in (
        "correct",
        "false_negative",
        "false_positive",
        "no_helmet_failure",
        "no_vest_failure",
        "small_object_failure",
    ):
        if not candidates[category]:
            continue
        _, image_name = max(candidates[category], key=lambda item: (item[0], item[1]))
        image_path, ground_truth, image_predictions = records[image_name]
        output_name = f"{category}.jpg"
        draw_panel(
            image_path,
            ground_truth,
            image_predictions,
            ASSETS / output_name,
            category.replace("_", " "),
        )
        examples.append(
            {"category": category, "file": output_name, "source": image_name}
        )
    return size_recall, examples


def internal_overlap_metrics(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    selected = [row for row in document["per_class"] if row["class_id"] in CLASS_IDS]
    return {
        "overall": {
            key: mean(row[key] for row in selected)
            for key in ("precision", "recall", "map50_95")
        }
        | {"map50": mean(row["ap50"] for row in selected)},
        "per_class": selected,
    }


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    models = {name: YOLO(str(path)) for name, path in MODELS.items()}
    primary = {
        name: run_validation(model, name, "test") for name, model in models.items()
    }
    secondary = {
        name: run_validation(model, name, "all") for name, model in models.items()
    }
    size_recall, failure_examples = diagnostics(models)
    internal = {
        "yolo26n": internal_overlap_metrics(
            ROOT / "docs/experiments/resolution_960/metrics.json"
        ),
        "yolo26s": internal_overlap_metrics(
            ROOT / "docs/experiments/model_capacity_960/metrics.json"
        ),
    }
    gaps = {
        name: {
            key: primary[name]["overall"][key] - internal[name]["overall"][key]
            for key in ("precision", "recall", "map50", "map50_95")
        }
        for name in models
    }
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "models": {name: str(path.resolve()) for name, path in MODELS.items()},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ultralytics": ultralytics.__version__,
        },
        "evaluation": {
            "imgsz": 960,
            "classes": [INTERNAL_CLASS_NAMES[index] for index in CLASS_IDS],
            "primary": "official leakage-free test split",
            "secondary": "all leakage-free external images; not independent",
        },
        "primary_test": primary,
        "secondary_all": secondary,
        "construction_ppe_validation_overlapping_classes": internal,
        "generalization_gap_external_minus_internal": gaps,
        "diagnostic_method": {
            "confidence": CONFIDENCE,
            "match_iou": MATCH_IOU,
            "nms_iou": NMS_IOU,
            "size_bins": {
                "tiny": "relative area < 0.001",
                "small": "0.001 <= relative area < 0.01",
                "large": "relative area >= 0.01",
            },
        },
        "size_recall": size_recall,
        "failure_examples": failure_examples,
    }
    (DOCS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = []
    for name in models:
        ext = primary[name]["overall"]
        gap = gaps[name]
        rows.append(
            f"| {name} | {ext['precision']:.4f} | {ext['recall']:.4f} | "
            f"{ext['map50']:.4f} | {ext['map50_95']:.4f} | {gap['map50_95']:+.4f} |"
        )
    delta = {
        key: primary["yolo26s"]["overall"][key] - primary["yolo26n"]["overall"][key]
        for key in ("precision", "recall", "map50", "map50_95")
    }
    comparison = f"""# External generalization comparison

## Protocol

Both frozen best checkpoints were evaluated at 960 pixels on the same five mapped classes.
The official leakage-free 90-image RF100 test split is primary. Validation AP uses the
Ultralytics defaults; diagnostic recall alone uses fixed confidence {CONFIDENCE}, NMS IoU
{NMS_IOU}, and class-aware match IoU {MATCH_IOU}. No training, fine-tuning, threshold
tuning, annotation repair, or augmentation change was performed.

## Primary external test results

| Model | Precision | Recall | mAP50 | mAP50-95 | mAP50-95 gap vs internal |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

External YOLO26s minus YOLO26n: precision {delta["precision"]:+.4f}, recall
{delta["recall"]:+.4f}, mAP50 {delta["map50"]:+.4f}, and mAP50-95
{delta["map50_95"]:+.4f}.

## Interpretation

The generalization gap compares each external primary result with the arithmetic mean of
the same five classes on Construction-PPE validation, avoiding contamination by the seven
non-overlapping classes. The official test set is independent but only 90 images; the
secondary all-image metrics in `metrics.json` are descriptive and not independent.

Size-stratified recall and representative correct, false-negative, false-positive,
missing-helmet, missing-vest, and small-object examples are recorded in `metrics.json` and
`docs/assets/external_generalization/`.
"""
    (DOCS / "comparison.md").write_text(comparison, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
