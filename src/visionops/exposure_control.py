"""Leakage filtering and controlled-stop logic for Checkpoint 7B."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from visionops.external_data import dhash64, hamming_distance, sha256_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SplitDuplicate:
    """One validation image matched to a training image."""

    train_file: str
    validation_file: str
    match_type: str
    dhash_distance: int
    train_annotation_classes: tuple[int, ...]
    validation_annotation_classes: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def image_files(directory: Path) -> list[Path]:
    """Return deterministically ordered supported image files."""
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def annotation_classes(path: Path) -> tuple[int, ...]:
    """Return sorted unique YOLO class IDs from one annotation file."""
    return tuple(
        sorted(
            {
                int(line.split()[0])
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        )
    )


def audit_train_validation_duplicates(
    train_images: Path,
    train_labels: Path,
    validation_images: Path,
    validation_labels: Path,
    threshold: int,
) -> list[SplitDuplicate]:
    """Find exact and perceptual RF100 train-to-validation duplicates."""
    if not 0 <= threshold <= 64:
        raise ValueError("dHash threshold must be between 0 and 64")
    train = image_files(train_images)
    validation = image_files(validation_images)
    exact: dict[str, list[Path]] = {}
    train_hashes: list[tuple[Path, int]] = []
    for path in train:
        exact.setdefault(sha256_file(path), []).append(path)
        train_hashes.append((path, dhash64(path)))

    matches: list[SplitDuplicate] = []
    for validation_path in validation:
        exact_candidates = exact.get(sha256_file(validation_path), [])
        if exact_candidates:
            train_path = min(exact_candidates, key=lambda path: path.name)
            distance = 0
            match_type = "exact_sha256"
        else:
            validation_hash = dhash64(validation_path)
            distance, train_path = min(
                (
                    (hamming_distance(validation_hash, train_hash), train_path)
                    for train_path, train_hash in train_hashes
                ),
                key=lambda item: (item[0], item[1].name),
            )
            if distance > threshold:
                continue
            match_type = "near_duplicate_dhash"
        matches.append(
            SplitDuplicate(
                train_file=train_path.name,
                validation_file=validation_path.name,
                match_type=match_type,
                dhash_distance=distance,
                train_annotation_classes=annotation_classes(
                    train_labels / f"{train_path.stem}.txt"
                ),
                validation_annotation_classes=annotation_classes(
                    validation_labels / f"{validation_path.stem}.txt"
                ),
            )
        )
    return matches


def create_filtered_validation_view(
    source_images: Path,
    source_labels: Path,
    output_images: Path,
    output_labels: Path,
    excluded_names: set[str],
) -> int:
    """Hard-link a deterministic validation view while leaving sources unchanged."""
    if output_images.exists() or output_labels.exists():
        raise FileExistsError("Refusing to overwrite filtered validation data")
    source = image_files(source_images)
    known = {path.name for path in source}
    unknown = excluded_names - known
    if unknown:
        raise ValueError(f"Excluded validation files do not exist: {sorted(unknown)}")
    output_images.mkdir(parents=True)
    output_labels.mkdir(parents=True)
    retained = 0
    for source_image in source:
        if source_image.name in excluded_names:
            continue
        source_label = source_labels / f"{source_image.stem}.txt"
        if not source_label.is_file():
            raise FileNotFoundError(source_label)
        for source_path, output_path in (
            (source_image, output_images / source_image.name),
            (source_label, output_labels / source_label.name),
        ):
            try:
                os.link(source_path, output_path)
            except OSError:
                shutil.copy2(source_path, output_path)
        retained += 1
    return retained


def should_controlled_stop(completed_epoch: int, stop_epoch: int) -> bool:
    """Return whether a completed human-readable epoch reaches the hard stop."""
    if completed_epoch < 1 or stop_epoch < 1:
        raise ValueError("Epoch numbers must be positive")
    return completed_epoch >= stop_epoch


def exposure_count(entries_per_epoch: int, completed_epochs: int) -> int:
    """Return deterministic total image-entry exposure."""
    if entries_per_epoch <= 0 or completed_epochs <= 0:
        raise ValueError("Exposure inputs must be positive")
    return entries_per_epoch * completed_epochs
