"""Five-class data-diversity experiment definitions and safeguards."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from math import ceil
from pathlib import Path

FIVE_CLASS_NAMES = ("helmet", "vest", "Person", "no_helmet", "no_vest")
CPPE_TO_FIVE = {0: 0, 2: 1, 5: 2, 6: 3, 11: 4}
RF100_TO_FIVE = {0: 0, 4: 1, 3: 2, 1: 3, 2: 4}


@dataclass(frozen=True)
class DerivationCounts:
    """Counts produced while deriving one split."""

    images: int
    retained_boxes: int
    removed_boxes: int


def remap_annotation_lines(
    lines: list[str], mapping: dict[int, int]
) -> tuple[list[str], int]:
    """Retain mapped YOLO annotations and count intentionally removed boxes."""
    mapped: list[str] = []
    removed = 0
    for line in lines:
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Expected five YOLO fields, found {len(fields)}")
        source_class = int(fields[0])
        if source_class not in mapping:
            removed += 1
            continue
        mapped.append(" ".join((str(mapping[source_class]), *fields[1:])))
    return mapped, removed


def derive_split(
    source_images: Path,
    source_labels: Path,
    output_images: Path,
    output_labels: Path,
    mapping: dict[int, int],
) -> DerivationCounts:
    """Create hard-linked images and remapped labels without changing sources."""
    output_images.mkdir(parents=True, exist_ok=False)
    output_labels.mkdir(parents=True, exist_ok=False)
    retained = 0
    removed = 0
    images = sorted(
        path
        for path in source_images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    for source_image in images:
        source_label = source_labels / f"{source_image.stem}.txt"
        if not source_label.is_file():
            raise FileNotFoundError(f"Missing label for {source_image}")
        target_image = output_images / source_image.name
        try:
            os.link(source_image, target_image)
        except OSError:
            shutil.copy2(source_image, target_image)
        mapped, removed_count = remap_annotation_lines(
            source_label.read_text(encoding="utf-8").splitlines(), mapping
        )
        (output_labels / source_label.name).write_text(
            "\n".join(mapped) + ("\n" if mapped else ""), encoding="utf-8"
        )
        retained += len(mapped)
        removed += removed_count
    return DerivationCounts(len(images), retained, removed)


def exposure_matched_entries(
    unique_cppe_entries: list[str], target_count: int
) -> list[str]:
    """Repeat CPPE entries deterministically until the target exposure is reached."""
    if not unique_cppe_entries:
        raise ValueError("At least one CPPE training entry is required")
    if target_count < len(unique_cppe_entries):
        raise ValueError("Target cannot be smaller than the unique CPPE set")
    ordered = sorted(unique_cppe_entries)
    return [ordered[index % len(ordered)] for index in range(target_count)]


def assert_rf100_test_excluded(paths: list[Path]) -> None:
    """Reject any RF100 test path before derivation, training, or evaluation."""
    for path in paths:
        lowered = tuple(part.lower() for part in path.resolve().parts)
        is_rf100 = any(
            "rf100" in part or "construction-safety-gsnvb" in part for part in lowered
        )
        if is_rf100 and "test" in lowered:
            raise ValueError(f"RF100 test is frozen and excluded: {path}")


def diversity_success(
    cppe_baseline_rf100_map: float,
    quantity_rf100_map: float,
    mixed_rf100_map: float,
    cppe_baseline_cppe_map: float,
    mixed_cppe_map: float,
) -> dict[str, bool | float]:
    """Apply the immutable Checkpoint 7 material-success decision rule."""
    rf100_gain = mixed_rf100_map - cppe_baseline_rf100_map
    diversity_gain = mixed_rf100_map - quantity_rf100_map
    cppe_change = mixed_cppe_map - cppe_baseline_cppe_map
    tolerance = 1e-12
    return {
        "rf100_gain": rf100_gain,
        "diversity_gain_over_quantity": diversity_gain,
        "cppe_change": cppe_change,
        "rf100_gain_pass": rf100_gain + tolerance >= 0.03,
        "diversity_gain_pass": diversity_gain + tolerance >= 0.02,
        "cppe_preservation_pass": cppe_change + tolerance >= -0.03,
        "material_success": (
            rf100_gain + tolerance >= 0.03
            and diversity_gain + tolerance >= 0.02
            and cppe_change + tolerance >= -0.03
        ),
    }


def reconstruct_optimizer_steps(
    images_per_epoch: int,
    batch_size: int,
    epochs: int,
    *,
    nominal_batch_size: int = 64,
    warmup_epochs: float = 3.0,
) -> int:
    """Reproduce Ultralytics' accumulated optimizer-step schedule."""
    if min(images_per_epoch, batch_size, epochs, nominal_batch_size) <= 0:
        raise ValueError("Step reconstruction inputs must be positive")
    batches = ceil(images_per_epoch / batch_size)
    warmup_iterations = max(round(warmup_epochs * batches), 100)
    target_accumulation = max(round(nominal_batch_size / batch_size), 1)
    last_step = -1
    steps = 0
    for epoch in range(epochs):
        for batch_index in range(batches):
            iteration = batch_index + batches * epoch
            if iteration <= warmup_iterations:
                fraction = iteration / warmup_iterations
                accumulation = max(
                    1,
                    round(1 + fraction * (target_accumulation - 1)),
                )
            else:
                accumulation = target_accumulation
            if iteration - last_step >= accumulation:
                steps += 1
                last_step = iteration
    return steps
