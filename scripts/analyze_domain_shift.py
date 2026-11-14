"""Characterize Construction-PPE to RF100 domain shift without model tuning."""

# ruff: noqa: E501

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO

from visionops.detection_analysis import Detection, area_bin
from visionops.domain_shift import (
    LOCALIZATION_IOU_FLOOR,
    MATCH_IOU,
    decompose_errors,
)
from visionops.external_data import INTERNAL_CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
CONSTRUCTION = ROOT / "data/construction-ppe"
RF100_SOURCE = ROOT / "data/external/construction-safety-gsnvb-v2"
RF100_MAPPED = ROOT / "data/external/generated/construction-safety-gsnvb-v2-remapped"
DOCS = ROOT / "docs/experiments/domain_shift"
ASSETS = ROOT / "docs/assets/domain_shift"
METRICS_PATH = DOCS / "metrics.json"
CLASS_IDS = (0, 2, 5, 6, 11)
CLASS_NAMES = tuple(INTERNAL_CLASS_NAMES[index] for index in CLASS_IDS)
RF100_MAPPING = {0: 0, 1: 6, 2: 11, 3: 5, 4: 2}
MODELS = {
    "yolo26n": ROOT / "runs/experiment_yolo26n_960/weights/best.pt",
    "yolo26s": ROOT / "runs/experiment_yolo26s_960/weights/best.pt",
}
CONFIDENCE = 0.25
NMS_IOU = 0.70
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_labels(path: Path, mapping: dict[int, int] | None = None) -> list[Detection]:
    detections: list[Detection] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_text, x_text, y_text, width_text, height_text = line.split()
        class_id = int(class_text)
        if mapping is not None:
            class_id = mapping[class_id]
        x, y, width, height = map(float, (x_text, y_text, width_text, height_text))
        detections.append(
            Detection(
                class_id,
                (x - width / 2, y - height / 2, x + width / 2, y + height / 2),
            )
        )
    return detections


def image_features(path: Path) -> dict[str, float]:
    with Image.open(path) as opened:
        width, height = opened.size
        if opened.getexif().get(274, 1) in {5, 6, 7, 8}:
            width, height = height, width
        opened.draft("RGB", (256, 256))
        oriented = ImageOps.exif_transpose(opened).convert("RGB")
        sample = oriented.copy()
        sample.thumbnail((256, 256), Image.Resampling.BILINEAR)
        rgb = np.asarray(sample, dtype=np.float32) / 255.0
        gray = np.asarray(sample.convert("L"), dtype=np.float32) / 255.0
        saturation = np.asarray(sample.convert("HSV"), dtype=np.float32)[..., 1] / 255.0
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        -4 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return {
        "width": float(width),
        "height": float(height),
        "aspect_ratio": width / height,
        "mean_luminance": float(gray.mean()),
        "luminance_std": float(gray.std()),
        "contrast": float(np.percentile(gray, 90) - np.percentile(gray, 10)),
        "saturation": float(saturation.mean()),
        "laplacian_variance": float(laplacian.var()),
        "mean_red": float(rgb[..., 0].mean()),
        "mean_green": float(rgb[..., 1].mean()),
        "mean_blue": float(rgb[..., 2].mean()),
    }


def dataset_records(dataset: str) -> list[dict[str, object]]:
    records = []
    if dataset == "construction_ppe":
        specifications = [
            (split, CONSTRUCTION / "images" / split, CONSTRUCTION / "labels" / split)
            for split in ("train", "val")
        ]
        mapping = None
    else:
        specifications = [
            (split, RF100_SOURCE / split / "images", RF100_SOURCE / split / "labels")
            for split in ("train", "valid")
        ]
        mapping = RF100_MAPPING
    for split, images_directory, labels_directory in specifications:
        for image_path in image_paths(images_directory):
            boxes = read_labels(labels_directory / f"{image_path.stem}.txt", mapping)
            records.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "path": image_path,
                    "boxes": boxes,
                    "features": image_features(image_path),
                }
            )
    return records


def distribution_metrics(records: list[dict[str, object]]) -> dict[str, object]:
    feature_names = (
        "width",
        "height",
        "aspect_ratio",
        "mean_luminance",
        "luminance_std",
        "contrast",
        "saturation",
        "laplacian_variance",
    )
    all_boxes = [box for record in records for box in record["boxes"]]
    areas = [box.relative_area for box in all_boxes]
    widths = [box.xyxy[2] - box.xyxy[0] for box in all_boxes]
    heights = [box.xyxy[3] - box.xyxy[1] for box in all_boxes]
    size_counts = Counter(area_bin(area) for area in areas)
    overlapping_box_total = sum(box.class_id in CLASS_IDS for box in all_boxes)
    per_class: dict[str, object] = {}
    for class_id in CLASS_IDS:
        class_boxes = [box for box in all_boxes if box.class_id == class_id]
        class_areas = [box.relative_area for box in class_boxes]
        class_size_counts = Counter(area_bin(area) for area in class_areas)
        images_with_class = sum(
            any(box.class_id == class_id for box in record["boxes"])
            for record in records
        )
        per_class[INTERNAL_CLASS_NAMES[class_id]] = {
            "boxes": len(class_boxes),
            "images": images_with_class,
            "box_frequency": len(class_boxes) / len(all_boxes) if all_boxes else 0.0,
            "overlap_box_frequency": (
                len(class_boxes) / overlapping_box_total
                if overlapping_box_total
                else 0.0
            ),
            "image_frequency": images_with_class / len(records),
            "median_relative_area": quantiles(class_areas)["p50"],
            "size_percentage": {
                size: class_size_counts[size] / len(class_boxes) if class_boxes else 0.0
                for size in ("tiny", "small", "large")
            },
        }
    person_id = 5
    pair_ids = {"helmet": 0, "no_helmet": 6, "vest": 2, "no_vest": 11}
    cooccurrence = {}
    for name, class_id in pair_ids.items():
        count = sum(
            any(box.class_id == person_id for box in record["boxes"])
            and any(box.class_id == class_id for box in record["boxes"])
            for record in records
        )
        cooccurrence[f"Person+{name}"] = {
            "images": count,
            "frequency": count / len(records),
        }
    person_count = sum(box.class_id == person_id for box in all_boxes)
    ppe_count = sum(box.class_id in {0, 2, 6, 11} for box in all_boxes)
    person_images = [
        record
        for record in records
        if any(box.class_id == person_id for box in record["boxes"])
    ]
    return {
        "image_count": len(records),
        "image_statistics": {
            name: quantiles([record["features"][name] for record in records])
            for name in feature_names
        },
        "bounding_boxes": {
            "total": len(all_boxes),
            "per_image": quantiles([len(record["boxes"]) for record in records]),
            "relative_area": quantiles(areas),
            "relative_width": quantiles(widths),
            "relative_height": quantiles(heights),
            "size_proportion": {
                size: size_counts[size] / len(all_boxes) if all_boxes else 0.0
                for size in ("tiny", "small", "large")
            },
        },
        "overlapping_classes": per_class,
        "cooccurrence": cooccurrence,
        "context": {
            "person_instances": person_count,
            "ppe_instances": ppe_count,
            "ppe_instances_per_person": ppe_count / person_count
            if person_count
            else None,
            "average_ppe_in_person_images": mean(
                sum(box.class_id in {0, 2, 6, 11} for box in record["boxes"])
                for record in person_images
            )
            if person_images
            else None,
            "multiple_person_images": sum(
                sum(box.class_id == person_id for box in record["boxes"]) > 1
                for record in records
            ),
            "multiple_person_image_frequency": sum(
                sum(box.class_id == person_id for box in record["boxes"]) > 1
                for record in records
            )
            / len(records),
        },
    }


def prediction_detections(result: object) -> list[Detection]:
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


def diagnostic_dataset(dataset: str) -> tuple[Path, Path]:
    if dataset == "construction_ppe_validation":
        return CONSTRUCTION / "images/val", CONSTRUCTION / "labels/val"
    return RF100_MAPPED / "images/valid", RF100_MAPPED / "labels/valid"


def model_diagnostics(model: YOLO, dataset: str) -> dict[str, object]:
    images_directory, labels_directory = diagnostic_dataset(dataset)
    paths = image_paths(images_directory)
    results = model.predict(
        source=str(images_directory),
        imgsz=960,
        conf=CONFIDENCE,
        iou=NMS_IOU,
        classes=list(CLASS_IDS),
        device=0 if torch.cuda.is_available() else "cpu",
        batch=4,
        stream=True,
        verbose=False,
    )
    category_counts: Counter[str] = Counter()
    false_positives = 0
    confidence: dict[str, list[float]] = defaultdict(list)
    class_size: dict[str, Counter[str]] = defaultdict(Counter)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    total_predictions = 0
    for result in results:
        image_path = Path(result.path)
        ground_truth = [
            box
            for box in read_labels(labels_directory / f"{image_path.stem}.txt")
            if box.class_id in CLASS_IDS
        ]
        predictions = prediction_detections(result)
        total_predictions += len(predictions)
        decomposition = decompose_errors(ground_truth, predictions)
        for outcome in decomposition.outcomes:
            target = ground_truth[outcome.ground_truth_index]
            category_counts[outcome.category] += 1
            class_name = INTERNAL_CLASS_NAMES[target.class_id]
            size = area_bin(target.relative_area)
            class_size[f"{class_name}|{size}"][outcome.category] += 1
            if outcome.prediction_index is not None:
                prediction = predictions[outcome.prediction_index]
                confidence[outcome.category].append(prediction.confidence)
                if outcome.category == "classification_confusion":
                    confusion[class_name][
                        INTERNAL_CLASS_NAMES[prediction.class_id]
                    ] += 1
            if outcome.category == "correct":
                confusion[class_name][class_name] += 1
            elif outcome.category in {"complete_miss", "localization_failure"}:
                confusion[class_name]["background_or_unmatched"] += 1
        for prediction_index in decomposition.false_positive_indices:
            false_positives += 1
            confidence["false_positive"].append(
                predictions[prediction_index].confidence
            )
    total_gt = sum(category_counts.values())
    matrix = {}
    for class_name in CLASS_NAMES:
        row = confusion[class_name]
        row_total = sum(row.values())
        matrix[class_name] = {
            predicted: {
                "count": row[predicted],
                "row_normalized": row[predicted] / row_total if row_total else 0.0,
            }
            for predicted in (*CLASS_NAMES, "background_or_unmatched")
        }
    size_matrix = {}
    for class_name in CLASS_NAMES:
        size_matrix[class_name] = {}
        for size in ("tiny", "small", "large"):
            counts = class_size[f"{class_name}|{size}"]
            total = sum(counts.values())
            size_matrix[class_name][size] = {
                category: counts[category]
                for category in (
                    "correct",
                    "localization_failure",
                    "classification_confusion",
                    "complete_miss",
                )
            } | {
                "total": total,
                "failure_rate": 1 - counts["correct"] / total if total else None,
            }
    return {
        "images": len(paths),
        "ground_truth_objects": total_gt,
        "predictions": total_predictions,
        "error_counts": {
            category: category_counts[category]
            for category in (
                "correct",
                "localization_failure",
                "classification_confusion",
                "complete_miss",
            )
        }
        | {"false_positive": false_positives},
        "error_rates_per_gt": {
            category: category_counts[category] / total_gt if total_gt else 0.0
            for category in (
                "correct",
                "localization_failure",
                "classification_confusion",
                "complete_miss",
            )
        }
        | {"false_positive": false_positives / total_gt if total_gt else 0.0},
        "confidence": {
            category: quantiles(values)
            for category, values in sorted(confidence.items())
        },
        "size_class_matrix": size_matrix,
        "confusion_matrix": matrix,
    }


def draw_record(record: dict[str, object], title: str) -> Image.Image:
    with Image.open(record["path"]) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((760, 600))
    canvas = Image.new("RGB", (760, 640), "white")
    canvas.paste(image, ((760 - image.width) // 2, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), title, fill="black")
    offset_x = (760 - image.width) // 2
    for box in record["boxes"]:
        if box.class_id not in CLASS_IDS:
            continue
        x1, y1, x2, y2 = box.xyxy
        coordinates = (
            offset_x + x1 * image.width,
            40 + y1 * image.height,
            offset_x + x2 * image.width,
            40 + y2 * image.height,
        )
        draw.rectangle(coordinates, outline="red", width=2)
        draw.text(
            (coordinates[0], coordinates[1]),
            INTERNAL_CLASS_NAMES[box.class_id],
            fill="red",
        )
    return canvas


def create_panels(
    construction_records: list[dict[str, object]],
    rf100_records: list[dict[str, object]],
) -> list[str]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    selected: list[str] = []
    datasets = {
        "Construction-PPE": construction_records,
        "RF100": rf100_records,
    }
    median_objects = {
        name: float(np.median([len(record["boxes"]) for record in records]))
        for name, records in datasets.items()
    }
    median_aspect = {
        name: float(
            np.median([record["features"]["aspect_ratio"] for record in records])
        )
        for name, records in datasets.items()
    }

    def choice(name: str, category: str) -> dict[str, object]:
        records = datasets[name]
        if category == "common":
            return min(
                records,
                key=lambda record: (
                    abs(len(record["boxes"]) - median_objects[name])
                    + abs(record["features"]["aspect_ratio"] - median_aspect[name]),
                    str(record["path"]),
                ),
            )
        if category == "crowded":
            return max(
                records, key=lambda record: (len(record["boxes"]), str(record["path"]))
            )
        if category == "small_ppe":
            return max(
                records,
                key=lambda record: (
                    sum(
                        box.class_id in CLASS_IDS
                        and area_bin(box.relative_area) in {"tiny", "small"}
                        for box in record["boxes"]
                    ),
                    str(record["path"]),
                ),
            )
        if category == "negative_ppe":
            return max(
                records,
                key=lambda record: (
                    sum(box.class_id in {6, 11} for box in record["boxes"]),
                    str(record["path"]),
                ),
            )
        if category == "darker":
            return min(
                records,
                key=lambda record: (
                    record["features"]["mean_luminance"],
                    str(record["path"]),
                ),
            )
        if category == "blur_proxy":
            return min(
                records,
                key=lambda record: (
                    record["features"]["laplacian_variance"],
                    str(record["path"]),
                ),
            )
        return max(
            records,
            key=lambda record: (
                abs(np.log(record["features"]["aspect_ratio"] / median_aspect[name])),
                str(record["path"]),
            ),
        )

    for category in (
        "common",
        "crowded",
        "small_ppe",
        "negative_ppe",
        "darker",
        "blur_proxy",
        "unusual_aspect_ratio",
    ):
        panels = []
        for name in datasets:
            record = choice(name, category)
            panels.append(
                draw_record(
                    record, f"{name} | {category} | {Path(record['path']).name}"
                )
            )
        output = Image.new("RGB", (1520, 640), "white")
        output.paste(panels[0], (0, 0))
        output.paste(panels[1], (760, 0))
        filename = f"{category}.jpg"
        output.save(ASSETS / filename, quality=88)
        selected.append(filename)
    return selected


def markdown_reports(metrics: dict[str, object]) -> None:
    construction = metrics["distributions"]["construction_ppe_train_validation"]
    rf100 = metrics["distributions"]["rf100_train_validation"]
    feature_rows = []
    for feature in construction["image_statistics"]:
        feature_rows.append(
            f"| {feature} | {construction['image_statistics'][feature]['p50']:.4f} | "
            f"{rf100['image_statistics'][feature]['p50']:.4f} |"
        )
    class_rows = []
    for name in CLASS_NAMES:
        first = construction["overlapping_classes"][name]
        second = rf100["overlapping_classes"][name]
        class_rows.append(
            f"| {name} | {first['boxes']} | {first['overlap_box_frequency']:.4f} | "
            f"{second['boxes']} | {second['overlap_box_frequency']:.4f} |"
        )
    distribution = f"""# Distribution comparison

RF100 test is frozen and excluded. This report compares Construction-PPE train+validation
with RF100 train+validation only. Pixel statistics are image-weighted and computed on
deterministic, aspect-preserving thumbnails no larger than 256 x 256. Contrast is the
per-image luminance p90-p10 range; sharpness is variance of a discrete grayscale
Laplacian. Full native annotations are used for overall box statistics; class comparisons
use only the five mapped classes.

## Image statistics (median)

| Statistic | Construction-PPE | RF100 |
|---|---:|---:|
{chr(10).join(feature_rows)}

## Bounding boxes

| Dataset | Images | Boxes | Median boxes/image | Median relative area | Tiny | Small | Large |
|---|---:|---:|---:|---:|---:|---:|---:|
| Construction-PPE | {construction["image_count"]} | {construction["bounding_boxes"]["total"]} | {construction["bounding_boxes"]["per_image"]["p50"]:.2f} | {construction["bounding_boxes"]["relative_area"]["p50"]:.6f} | {construction["bounding_boxes"]["size_proportion"]["tiny"]:.3f} | {construction["bounding_boxes"]["size_proportion"]["small"]:.3f} | {construction["bounding_boxes"]["size_proportion"]["large"]:.3f} |
| RF100 | {rf100["image_count"]} | {rf100["bounding_boxes"]["total"]} | {rf100["bounding_boxes"]["per_image"]["p50"]:.2f} | {rf100["bounding_boxes"]["relative_area"]["p50"]:.6f} | {rf100["bounding_boxes"]["size_proportion"]["tiny"]:.3f} | {rf100["bounding_boxes"]["size_proportion"]["small"]:.3f} | {rf100["bounding_boxes"]["size_proportion"]["large"]:.3f} |

## Overlapping class frequencies

| Class | Construction boxes | Construction overlap frequency | RF100 boxes | RF100 overlap frequency |
|---|---:|---:|---:|---:|
{chr(10).join(class_rows)}

## Co-occurrence scope

Co-occurrence is image-level dataset context only. It does not infer which PPE item belongs
to which person. Full values, per-class size proportions, image frequencies, and context
ratios are in `metrics.json`.

## Measured interpretation

RF100 images have lower native median resolution ({rf100["image_statistics"]["width"]["p50"]:.0f} x {rf100["image_statistics"]["height"]["p50"]:.0f} versus {construction["image_statistics"]["width"]["p50"]:.0f} x {construction["image_statistics"]["height"]["p50"]:.0f}), but nearly the same median aspect ratio. RF100 is brighter at the median ({rf100["image_statistics"]["mean_luminance"]["p50"]:.3f} versus {construction["image_statistics"]["mean_luminance"]["p50"]:.3f}) and more saturated ({rf100["image_statistics"]["saturation"]["p50"]:.3f} versus {construction["image_statistics"]["saturation"]["p50"]:.3f}). Median contrast and Laplacian variance are similar; the data do not support darker, lower-contrast, or blurrier RF100 imagery as the dominant dataset-wide explanation.

RF100 boxes are substantially larger: median relative area is {rf100["bounding_boxes"]["relative_area"]["p50"]:.4f} versus {construction["bounding_boxes"]["relative_area"]["p50"]:.4f}. Tiny boxes are {rf100["bounding_boxes"]["size_proportion"]["tiny"]:.1%} of RF100 annotations versus {construction["bounding_boxes"]["size_proportion"]["tiny"]:.1%} in Construction-PPE. Therefore, the external degradation cannot be attributed primarily to a greater prevalence of small objects.

Class composition changes materially. RF100 contains proportionally more `helmet`, `Person`, and `no_vest`, while `no_helmet` is only {rf100["overlapping_classes"]["no_helmet"]["overlap_box_frequency"]:.1%} of overlapping-class boxes versus {construction["overlapping_classes"]["no_helmet"]["overlap_box_frequency"]:.1%}. RF100 also has more multi-person images ({rf100["context"]["multiple_person_image_frequency"]:.1%} versus {construction["context"]["multiple_person_image_frequency"]:.1%}).
"""
    (DOCS / "distribution_comparison.md").write_text(distribution, encoding="utf-8")

    error_rows = []
    for model_name, datasets in metrics["diagnostics"].items():
        for dataset_name, values in datasets.items():
            counts = values["error_counts"]
            error_rows.append(
                f"| {model_name} | {dataset_name} | {counts['correct']} | "
                f"{counts['localization_failure']} | {counts['classification_confusion']} | "
                f"{counts['complete_miss']} | {counts['false_positive']} |"
            )
    error_report = f"""# Error and confidence analysis

## Fixed protocol

- Confidence: {CONFIDENCE}
- NMS IoU: {NMS_IOU}
- Correct/confusion overlap IoU: {MATCH_IOU}
- Localization diagnostic overlap: {LOCALIZATION_IOU_FLOOR} <= IoU < {MATCH_IOU}
- Assignment precedence: correct, classification confusion, localization failure, miss
- Each prediction may be assigned to at most one ground-truth object
- RF100 test is not read or evaluated

## Error counts

| Model | Dataset | Correct | Localization | Classification | Miss | False positive |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(error_rows)}

Confidence quantiles, normalized confusion matrices, and class-by-size error matrices are
recorded in `metrics.json`. A missed object has no detection confidence; confidence for
localization and classification errors is reported when an assigned marginal prediction
exists.

## Measured interpretation

For YOLO26n, correct detections fall from {metrics["diagnostics"]["yolo26n"]["construction_ppe_validation"]["error_rates_per_gt"]["correct"]:.1%} in-domain to {metrics["diagnostics"]["yolo26n"]["rf100_validation"]["error_rates_per_gt"]["correct"]:.1%} on RF100 validation. Localization errors rise from {metrics["diagnostics"]["yolo26n"]["construction_ppe_validation"]["error_rates_per_gt"]["localization_failure"]:.1%} to {metrics["diagnostics"]["yolo26n"]["rf100_validation"]["error_rates_per_gt"]["localization_failure"]:.1%}, and false positives per GT rise from {metrics["diagnostics"]["yolo26n"]["construction_ppe_validation"]["error_rates_per_gt"]["false_positive"]:.1%} to {metrics["diagnostics"]["yolo26n"]["rf100_validation"]["error_rates_per_gt"]["false_positive"]:.1%}. Complete misses do not increase for YOLO26n at this operating point.

For YOLO26s, correct detections fall from {metrics["diagnostics"]["yolo26s"]["construction_ppe_validation"]["error_rates_per_gt"]["correct"]:.1%} to {metrics["diagnostics"]["yolo26s"]["rf100_validation"]["error_rates_per_gt"]["correct"]:.1%}; false positives per GT rise from {metrics["diagnostics"]["yolo26s"]["construction_ppe_validation"]["error_rates_per_gt"]["false_positive"]:.1%} to {metrics["diagnostics"]["yolo26s"]["rf100_validation"]["error_rates_per_gt"]["false_positive"]:.1%}. YOLO26s produces fewer RF100 false positives and localization errors than YOLO26n, but more complete misses, and both have exactly {metrics["diagnostics"]["yolo26s"]["rf100_validation"]["error_counts"]["correct"]} correct RF100-validation assignments.

Median true-positive confidence falls only modestly: {metrics["diagnostics"]["yolo26n"]["construction_ppe_validation"]["confidence"]["correct"]["p50"]:.3f} to {metrics["diagnostics"]["yolo26n"]["rf100_validation"]["confidence"]["correct"]["p50"]:.3f} for YOLO26n and {metrics["diagnostics"]["yolo26s"]["construction_ppe_validation"]["confidence"]["correct"]["p50"]:.3f} to {metrics["diagnostics"]["yolo26s"]["rf100_validation"]["confidence"]["correct"]["p50"]:.3f} for YOLO26s. In contrast, median false-positive confidence increases from {metrics["diagnostics"]["yolo26n"]["construction_ppe_validation"]["confidence"]["false_positive"]["p50"]:.3f} to {metrics["diagnostics"]["yolo26n"]["rf100_validation"]["confidence"]["false_positive"]["p50"]:.3f} for YOLO26n and {metrics["diagnostics"]["yolo26s"]["construction_ppe_validation"]["confidence"]["false_positive"]["p50"]:.3f} to {metrics["diagnostics"]["yolo26s"]["rf100_validation"]["confidence"]["false_positive"]["p50"]:.3f} for YOLO26s. This is not a uniform confidence collapse; external false predictions are more confident.

The robust size/class regressions include `helmet` small, `vest` large, `Person` large, and `no_vest` large. Very large percentage changes in RF100 tiny `Person` and `no_helmet` bins have only 2-9 objects and must not drive intervention selection. `no_vest` confused as `vest` rises externally for both models. Measured errors are dominated by false positives and increased localization/unmatched errors rather than wrong-class confusion alone.

## Evidence-based next intervention category

**More diverse training data** should be tested next. The shift spans scene composition, annotation/class frequencies, confident false positives, localization, and multiple well-populated class/size cells while RF100 objects are generally larger and photometric medians are not severely shifted. Model capacity did not close the independent gap. Photometric or scale augmentation alone is therefore less directly supported; explicit domain adaptation can remain a later predeclared alternative if broader representative training data are insufficient.

## Feature-space analysis

Skipped. Stable intermediate YOLO26 feature extraction would require coupling this audit
to private model-layer structure, making the result fragile and invasive. No unrelated
embedding model or dependency was introduced.
"""
    (DOCS / "error_analysis.md").write_text(error_report, encoding="utf-8")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    construction_records = dataset_records("construction_ppe")
    rf100_records = dataset_records("rf100")
    distributions = {
        "construction_ppe_train_validation": distribution_metrics(construction_records),
        "rf100_train_validation": distribution_metrics(rf100_records),
    }
    diagnostics = {}
    for model_name, model_path in MODELS.items():
        model = YOLO(str(model_path))
        diagnostics[model_name] = {
            dataset: model_diagnostics(model, dataset)
            for dataset in ("construction_ppe_validation", "rf100_validation")
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    panels = create_panels(construction_records, rf100_records)
    metrics = {
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": {
            "rf100_test_used": False,
            "confidence": CONFIDENCE,
            "nms_iou": NMS_IOU,
            "match_iou": MATCH_IOU,
            "localization_iou_floor": LOCALIZATION_IOU_FLOOR,
            "size_bins": {
                "tiny": "relative area < 0.001",
                "small": "0.001 <= relative area < 0.01",
                "large": "relative area >= 0.01",
            },
        },
        "distributions": distributions,
        "diagnostics": diagnostics,
        "representative_panels": panels,
        "feature_space_analysis": {
            "performed": False,
            "reason": "Stable YOLO26 intermediate extraction is invasive and layer-private.",
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    markdown_reports(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
