"""Audit the Ultralytics Construction-PPE dataset without modifying it."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val")
VERY_SMALL_AREA_THRESHOLD = 0.01
TINY_AREA_THRESHOLD = 0.001


@dataclass
class AuditResult:
    """JSON summary plus raw values used by the visualization script."""

    summary: dict[str, Any]
    image_sizes: list[tuple[int, int]]
    box_areas: list[float]


def numeric_stats(values: list[float | int]) -> dict[str, float | int | None]:
    """Return stable descriptive statistics for a numeric sequence."""
    if not values:
        return {key: None for key in ("count", "min", "max", "mean", "median", "std")}
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "std": statistics.pstdev(numeric),
    }


def load_dataset_config(dataset_root: Path) -> tuple[Path, list[str]]:
    """Locate the dataset YAML and read its ordered class names."""
    candidates = [dataset_root / "construction-ppe.yaml", dataset_root / "data.yaml"]
    candidates.extend(sorted(dataset_root.glob("*.yaml")))
    config_path = next((path for path in candidates if path.is_file()), None)
    if config_path is None:
        raise FileNotFoundError(f"No dataset YAML found under {dataset_root}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = config.get("names")
    if isinstance(names, dict):
        class_names = [str(names[index]) for index in sorted(names, key=int)]
    elif isinstance(names, list):
        class_names = [str(name) for name in names]
    else:
        raise ValueError(f"Invalid or missing class names in {config_path}")
    return config_path, class_names


def image_files(directory: Path) -> list[Path]:
    """List supported images in deterministic path order."""
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )


def label_files(directory: Path) -> list[Path]:
    """List YOLO label files in deterministic path order."""
    return sorted(directory.rglob("*.txt")) if directory.is_dir() else []


def relative_stem(path: Path, base: Path) -> str:
    """Return a case-normalized relative path without its suffix."""
    return path.relative_to(base).with_suffix("").as_posix().casefold()


def issue(path: Path, dataset_root: Path, **details: Any) -> dict[str, Any]:
    """Build a machine-readable issue record."""
    return {"file": path.relative_to(dataset_root).as_posix(), **details}


def audit_dataset(dataset_root: Path, output_dir: Path) -> AuditResult:
    """Inspect images and YOLO labels and write a complete JSON report."""
    dataset_root = dataset_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path, class_names = load_dataset_config(dataset_root)

    split_image_counts: dict[str, int] = {}
    split_label_counts: dict[str, int] = {}
    all_images: list[Path] = []
    all_labels: list[Path] = []
    missing_labels: list[str] = []
    missing_images: list[str] = []

    for split in SPLITS:
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        images = image_files(images_dir)
        labels = label_files(labels_dir)
        split_image_counts[split] = len(images)
        split_label_counts[split] = len(labels)
        all_images.extend(images)
        all_labels.extend(labels)

        images_by_stem = {relative_stem(path, images_dir): path for path in images}
        labels_by_stem = {relative_stem(path, labels_dir): path for path in labels}
        missing_labels.extend(
            images_by_stem[stem].relative_to(dataset_root).as_posix()
            for stem in sorted(images_by_stem.keys() - labels_by_stem.keys())
        )
        missing_images.extend(
            labels_by_stem[stem].relative_to(dataset_root).as_posix()
            for stem in sorted(labels_by_stem.keys() - images_by_stem.keys())
        )

    corrupt_images: list[dict[str, Any]] = []
    image_sizes: list[tuple[int, int]] = []
    for image_path in all_images:
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
                image.load()
            if width <= 0 or height <= 0:
                raise ValueError(f"invalid dimensions {width}x{height}")
            image_sizes.append((width, height))
        except (OSError, UnidentifiedImageError, ValueError) as error:
            corrupt_images.append(
                issue(
                    image_path,
                    dataset_root,
                    error=f"{type(error).__name__}: {error}",
                )
            )

    malformed_annotations: list[dict[str, Any]] = []
    normalized_values_out_of_range: list[dict[str, Any]] = []
    boxes_outside_image_bounds: list[dict[str, Any]] = []
    invalid_boxes: list[dict[str, Any]] = []
    boxes_per_class: Counter[int] = Counter()
    images_per_class: defaultdict[int, set[str]] = defaultdict(set)
    box_areas: list[float] = []
    parsed_box_count = 0

    for label_path in all_labels:
        image_classes: set[int] = set()
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            malformed_annotations.append(
                issue(label_path, dataset_root, line=None, reason=str(error))
            )
            continue

        for line_number, raw_line in enumerate(lines, start=1):
            fields = raw_line.split()
            location = {
                "line": line_number,
                "content": raw_line,
            }
            if len(fields) != 5:
                malformed_annotations.append(
                    issue(
                        label_path,
                        dataset_root,
                        **location,
                        reason=f"expected 5 fields, found {len(fields)}",
                    )
                )
                continue
            try:
                class_value = float(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError:
                malformed_annotations.append(
                    issue(
                        label_path,
                        dataset_root,
                        **location,
                        reason="non-numeric field",
                    )
                )
                continue
            if not math.isfinite(class_value) or not all(
                math.isfinite(value) for value in coordinates
            ):
                malformed_annotations.append(
                    issue(
                        label_path,
                        dataset_root,
                        **location,
                        reason="non-finite numeric field",
                    )
                )
                continue
            class_id = int(class_value)
            if class_value != class_id or not 0 <= class_id < len(class_names):
                malformed_annotations.append(
                    issue(
                        label_path,
                        dataset_root,
                        **location,
                        reason=f"invalid class ID {fields[0]}",
                    )
                )
                continue

            x_center, y_center, width, height = coordinates
            parsed_box_count += 1
            boxes_per_class[class_id] += 1
            image_classes.add(class_id)
            area = width * height
            box_areas.append(area)

            if any(value < 0 or value > 1 for value in coordinates):
                normalized_values_out_of_range.append(
                    issue(label_path, dataset_root, **location, values=coordinates)
                )
            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
                boxes_outside_image_bounds.append(
                    issue(
                        label_path,
                        dataset_root,
                        **location,
                        bounds=[x_min, y_min, x_max, y_max],
                    )
                )
            if width <= 0 or height <= 0:
                invalid_boxes.append(
                    issue(
                        label_path,
                        dataset_root,
                        **location,
                        reason=f"non-positive box size {width}x{height}",
                    )
                )

        image_key = label_path.relative_to(dataset_root).with_suffix("").as_posix()
        for class_id in image_classes:
            images_per_class[class_id].add(image_key)

    class_box_counts = {
        class_names[class_id]: boxes_per_class[class_id]
        for class_id in range(len(class_names))
    }
    class_image_counts = {
        class_names[class_id]: len(images_per_class[class_id])
        for class_id in range(len(class_names))
    }
    count_values = list(class_box_counts.values())
    nonzero_counts = [count for count in count_values if count > 0]
    imbalance = {
        **numeric_stats(count_values),
        "max_to_min_ratio": (
            max(nonzero_counts) / min(nonzero_counts) if nonzero_counts else None
        ),
        "coefficient_of_variation": (
            statistics.pstdev(count_values) / statistics.fmean(count_values)
            if count_values and statistics.fmean(count_values)
            else None
        ),
        "majority_class": max(class_box_counts, key=class_box_counts.get),
        "minority_class": min(class_box_counts, key=class_box_counts.get),
    }
    widths = [width for width, _ in image_sizes]
    heights = [height for _, height in image_sizes]
    aspect_ratios = [width / height for width, height in image_sizes]
    valid_areas = [area for area in box_areas if area > 0]
    very_small_count = sum(area < VERY_SMALL_AREA_THRESHOLD for area in valid_areas)
    tiny_count = sum(area < TINY_AREA_THRESHOLD for area in valid_areas)

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "dataset_config": config_path.relative_to(dataset_root).as_posix(),
        "classes": [
            {"id": class_id, "name": name} for class_id, name in enumerate(class_names)
        ],
        "image_counts": {
            **split_image_counts,
            "total": sum(split_image_counts.values()),
        },
        "label_file_counts": {
            **split_label_counts,
            "total": sum(split_label_counts.values()),
        },
        "pairing": {
            "images_without_labels_count": len(missing_labels),
            "images_without_labels": missing_labels,
            "labels_without_images_count": len(missing_images),
            "labels_without_images": missing_images,
        },
        "corrupt_images": {
            "count": len(corrupt_images),
            "items": corrupt_images,
        },
        "annotations": {
            "total_bounding_boxes": parsed_box_count,
            "malformed_count": len(malformed_annotations),
            "malformed": malformed_annotations,
            "normalized_values_out_of_range_count": len(normalized_values_out_of_range),
            "normalized_values_out_of_range": normalized_values_out_of_range,
            "boxes_outside_image_bounds_count": len(boxes_outside_image_bounds),
            "boxes_outside_image_bounds": boxes_outside_image_bounds,
            "zero_area_or_invalid_count": len(invalid_boxes),
            "zero_area_or_invalid": invalid_boxes,
        },
        "box_count_per_class": class_box_counts,
        "images_containing_class": class_image_counts,
        "class_imbalance": imbalance,
        "image_dimensions": {
            "width": numeric_stats(widths),
            "height": numeric_stats(heights),
        },
        "aspect_ratio": numeric_stats(aspect_ratios),
        "box_relative_area": numeric_stats(valid_areas),
        "very_small_objects": {
            "definition": f"relative box area < {VERY_SMALL_AREA_THRESHOLD}",
            "count": very_small_count,
            "percentage": (
                100 * very_small_count / len(valid_areas) if valid_areas else 0.0
            ),
            "tiny_definition": f"relative box area < {TINY_AREA_THRESHOLD}",
            "tiny_count": tiny_count,
            "tiny_percentage": (
                100 * tiny_count / len(valid_areas) if valid_areas else 0.0
            ),
        },
    }
    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return AuditResult(summary=summary, image_sizes=image_sizes, box_areas=valid_areas)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/construction-ppe"),
        help="Dataset root containing images/, labels/, and its YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dataset_audit"),
        help="Directory for dataset_summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the audit and print its JSON summary."""
    args = parse_args()
    result = audit_dataset(args.dataset_root, args.output_dir)
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
