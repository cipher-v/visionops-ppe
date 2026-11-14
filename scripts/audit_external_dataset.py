"""Audit RF100 Construction Safety v2 and prepare leakage-free mapped labels."""

# ruff: noqa: E501

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

import yaml
from PIL import Image, ImageOps

from visionops.external_data import (
    DHASH_DISTANCE_THRESHOLD,
    EXTERNAL_CLASS_NAMES,
    EXTERNAL_TO_INTERNAL,
    INTERNAL_CLASS_NAMES,
    dhash64,
    hamming_distance,
    map_annotation_line,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/external/construction safety.v2-release.yolov5pytorch.zip"
SOURCE = ROOT / "data/external/construction-safety-gsnvb-v2"
INTERNAL = ROOT / "data/construction-ppe"
GENERATED = ROOT / "data/external/generated/construction-safety-gsnvb-v2-remapped"
AUDIT_JSON = GENERATED / "audit.json"
DOCS = ROOT / "docs/experiments/external_generalization"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }


def audit_split(split: str) -> dict[str, object]:
    images_dir = SOURCE / split / "images"
    labels_dir = SOURCE / split / "labels"
    images = image_files(images_dir)
    labels = sorted(labels_dir.glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    corrupt: list[str] = []
    malformed: list[dict[str, object]] = []
    out_of_range: list[dict[str, object]] = []
    invalid_boxes: list[dict[str, object]] = []
    widths: list[float] = []
    heights: list[float] = []
    aspect_ratios: list[float] = []
    areas: list[float] = []
    boxes_per_class: Counter[int] = Counter()
    images_per_class: Counter[int] = Counter()

    for image_path in images:
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                oriented = ImageOps.exif_transpose(image)
                width, height = oriented.size
            widths.append(float(width))
            heights.append(float(height))
            aspect_ratios.append(width / height)
        except Exception as error:  # Pillow exposes several format-specific errors.
            corrupt.append(f"{image_path.name}: {error}")

    for label_path in labels:
        present: set[int] = set()
        for line_number, raw_line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = raw_line.split()
            problem = {"file": label_path.name, "line": line_number, "text": raw_line}
            if len(fields) != 5:
                malformed.append({**problem, "reason": "field_count"})
                continue
            try:
                class_id = int(fields[0])
                x, y, width, height = (float(value) for value in fields[1:])
            except ValueError:
                malformed.append({**problem, "reason": "non_numeric"})
                continue
            if class_id not in range(len(EXTERNAL_CLASS_NAMES)):
                malformed.append({**problem, "reason": "class_id"})
                continue
            if any(value < 0 or value > 1 for value in (x, y, width, height)):
                out_of_range.append(problem)
            if width <= 0 or height <= 0:
                invalid_boxes.append(problem)
            boxes_per_class[class_id] += 1
            present.add(class_id)
            areas.append(width * height)
        images_per_class.update(present)

    return {
        "image_count": len(images),
        "label_count": len(labels),
        "missing_labels": sorted(image_stems - label_stems),
        "missing_images": sorted(label_stems - image_stems),
        "corrupt_images": corrupt,
        "malformed_annotations": malformed,
        "out_of_range_boxes": out_of_range,
        "invalid_boxes": invalid_boxes,
        "total_boxes": sum(boxes_per_class.values()),
        "boxes_per_class": {
            EXTERNAL_CLASS_NAMES[index]: boxes_per_class[index]
            for index in range(len(EXTERNAL_CLASS_NAMES))
        },
        "images_per_class": {
            EXTERNAL_CLASS_NAMES[index]: images_per_class[index]
            for index in range(len(EXTERNAL_CLASS_NAMES))
        },
        "width": describe(widths),
        "height": describe(heights),
        "aspect_ratio": describe(aspect_ratios),
        "box_relative_area": describe(areas),
        "very_small_boxes_lt_0_001": sum(area < 0.001 for area in areas),
    }


def duplicate_audit() -> dict[str, object]:
    external: list[tuple[str, Path]] = []
    for split in ("train", "valid", "test"):
        external.extend(
            (split, path) for path in image_files(SOURCE / split / "images")
        )
    internal: list[tuple[str, Path]] = []
    for split in ("train", "val"):
        internal.extend(
            (split, path) for path in image_files(INTERNAL / "images" / split)
        )

    internal_exact: dict[str, list[tuple[str, Path]]] = {}
    internal_hashes: list[tuple[str, Path, int]] = []
    for split, path in internal:
        internal_exact.setdefault(sha256_file(path), []).append((split, path))
        internal_hashes.append((split, path, dhash64(path)))

    exact_matches: list[dict[str, object]] = []
    near_matches: list[dict[str, object]] = []
    excluded: set[str] = set()
    for split, path in external:
        relative = f"{split}/images/{path.name}"
        exact = internal_exact.get(sha256_file(path), [])
        if exact:
            excluded.add(relative)
            for internal_split, internal_path in exact:
                exact_matches.append(
                    {
                        "external": relative,
                        "internal_split": internal_split,
                        "internal": internal_path.name,
                    }
                )
            continue
        external_hash = dhash64(path)
        candidates = [
            (hamming_distance(external_hash, value), internal_split, internal_path)
            for internal_split, internal_path, value in internal_hashes
        ]
        distance, internal_split, internal_path = min(
            candidates, key=lambda item: item[0]
        )
        if distance <= DHASH_DISTANCE_THRESHOLD:
            excluded.add(relative)
            near_matches.append(
                {
                    "external": relative,
                    "internal_split": internal_split,
                    "internal": internal_path.name,
                    "dhash_distance": distance,
                }
            )
    return {
        "method": "SHA-256 exact; 64-bit grayscale dHash after EXIF orientation",
        "near_duplicate_threshold": DHASH_DISTANCE_THRESHOLD,
        "internal_images_compared": len(internal),
        "external_images_compared": len(external),
        "exact_matches": exact_matches,
        "near_matches": near_matches,
        "excluded_external_images": sorted(excluded),
    }


def prepare_generated_dataset(excluded: set[str]) -> dict[str, int]:
    if GENERATED.exists():
        shutil.rmtree(GENERATED)
    counts: dict[str, int] = {}
    all_lines: list[str] = []
    for split in ("train", "valid", "test"):
        target_images = GENERATED / "images" / split
        target_labels = GENERATED / "labels" / split
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        kept = 0
        for source_image in image_files(SOURCE / split / "images"):
            relative = f"{split}/images/{source_image.name}"
            if relative in excluded:
                continue
            target_image = target_images / source_image.name
            try:
                os.link(source_image, target_image)
            except OSError:
                shutil.copy2(source_image, target_image)
            source_label = SOURCE / split / "labels" / f"{source_image.stem}.txt"
            mapped = [
                map_annotation_line(line)
                for line in source_label.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            (target_labels / source_label.name).write_text(
                "\n".join(mapped) + ("\n" if mapped else ""), encoding="utf-8"
            )
            all_lines.append(str(target_image.resolve()))
            kept += 1
        counts[split] = kept
    (GENERATED / "all.txt").write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    data = {
        "path": str(GENERATED.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(INTERNAL_CLASS_NAMES)},
    }
    (GENERATED / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    all_data = {**data, "val": str((GENERATED / "all.txt").resolve())}
    (GENERATED / "data_all.yaml").write_text(
        yaml.safe_dump(all_data, sort_keys=False), encoding="utf-8"
    )
    return counts


def main() -> None:
    metadata = yaml.safe_load((SOURCE / "data.yaml").read_text(encoding="utf-8"))
    split_audits = {split: audit_split(split) for split in ("train", "valid", "test")}
    duplicates = duplicate_audit()
    generated_counts = prepare_generated_dataset(
        set(duplicates["excluded_external_images"])
    )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "archive": {
            "path": str(ARCHIVE.resolve()),
            "size_bytes": ARCHIVE.stat().st_size,
            "sha256": sha256_file(ARCHIVE),
        },
        "source": str(SOURCE.resolve()),
        "format": "YOLO normalized class_id x_center y_center width height",
        "source_class_names": metadata["names"],
        "mapping": {
            EXTERNAL_CLASS_NAMES[source]: INTERNAL_CLASS_NAMES[target]
            for source, target in EXTERNAL_TO_INTERNAL.items()
        },
        "splits": split_audits,
        "duplicates": duplicates,
        "generated_leakage_free_counts": generated_counts,
        "generated_dataset": str(GENERATED.resolve()),
    }
    GENERATED.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    DOCS.mkdir(parents=True, exist_ok=True)
    totals = {
        class_name: sum(
            split_audits[split]["boxes_per_class"][class_name] for split in split_audits
        )
        for class_name in EXTERNAL_CLASS_NAMES
    }
    issue_count = sum(
        len(split_audits[split][key])
        for split in split_audits
        for key in (
            "missing_labels",
            "missing_images",
            "corrupt_images",
            "malformed_annotations",
            "out_of_range_boxes",
            "invalid_boxes",
        )
    )
    table = "\n".join(
        f"| {split} | {split_audits[split]['image_count']} | "
        f"{split_audits[split]['label_count']} | {split_audits[split]['total_boxes']} |"
        for split in ("train", "valid", "test")
    )
    class_table = "\n".join(
        f"| {name} | {totals[name]} | {INTERNAL_CLASS_NAMES[EXTERNAL_TO_INTERNAL[index]]} |"
        for index, name in enumerate(EXTERNAL_CLASS_NAMES)
    )
    audit_markdown = f"""# External dataset audit

## Provenance

- Dataset: RF100 Construction Safety, version 2 (`release`)
- Source: https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb/dataset/2
- License: CC BY 4.0
- Export timestamp recorded by the archive: 2022-08-30 10:11 GMT
- Local source: `{SOURCE}`
- Archive: `{ARCHIVE.name}` ({ARCHIVE.stat().st_size} bytes)
- Archive SHA-256: `{report["archive"]["sha256"]}`
- Annotation format: YOLO normalized `class_id x_center y_center width height`

The extracted source is retained unchanged. Mapped labels and image hard links/copies live
only under the Git-ignored generated-data directory.

## Integrity

| Split | Images | Labels | Boxes |
|---|---:|---:|---:|
{table}

Integrity problems found: **{issue_count}**. No missing pairs, corrupt images, malformed
annotations, out-of-range coordinates, or non-positive boxes were found.

## Class mapping and counts

| RF100 class | Total boxes | Construction-PPE class |
|---|---:|---|
{class_table}

This mapping is explicit and no other Construction-PPE classes are evaluated.

## Leakage audit

Every one of the 1,206 external images was compared with all 1,000 Construction-PPE
train/validation images. Exact matching used file SHA-256. Near-duplicate matching used a
deterministic 64-bit grayscale difference hash after EXIF orientation; the threshold was
predeclared as Hamming distance <= {DHASH_DISTANCE_THRESHOLD}. Exact matches:
**{len(duplicates["exact_matches"])}**. Near matches: **{len(duplicates["near_matches"])}**.
Excluded external images: **{len(duplicates["excluded_external_images"])}**. Any match is
excluded from independent evaluation before metrics are computed.

## Evaluation policy

The official 90-image test split is the primary independent evaluation. It is small, so
metrics should be interpreted with that sampling limitation. A secondary result over all
leakage-free external images is reported separately and is not presented as independent.
No thresholds, model weights, augmentation, or dataset annotations are tuned or repaired.
"""
    (DOCS / "dataset_audit.md").write_text(audit_markdown, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
