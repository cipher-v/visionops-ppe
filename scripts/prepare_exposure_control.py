"""Audit RF100 train/validation leakage and build a filtered validation view."""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import yaml

from visionops.data_diversity import FIVE_CLASS_NAMES, assert_rf100_test_excluded
from visionops.exposure_control import (
    audit_train_validation_duplicates,
    create_filtered_validation_view,
    image_files,
)
from visionops.external_data import DHASH_DISTANCE_THRESHOLD, EXTERNAL_CLASS_NAMES

ROOT = Path(__file__).resolve().parents[1]
RF100_SOURCE = ROOT / "data/external/construction-safety-gsnvb-v2"
RF100_MAPPED = ROOT / "data/generated/rf100-five-class"
OUTPUT = ROOT / "data/generated/rf100-valid-filtered"
METADATA = OUTPUT / "filter_metadata.json"


def main() -> None:
    guarded = [
        RF100_SOURCE / "train/images",
        RF100_SOURCE / "train/labels",
        RF100_SOURCE / "valid/images",
        RF100_SOURCE / "valid/labels",
        RF100_MAPPED / "images/valid",
        RF100_MAPPED / "labels/valid",
    ]
    assert_rf100_test_excluded(guarded)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite filtered data: {OUTPUT}")
    matches = audit_train_validation_duplicates(
        RF100_SOURCE / "train/images",
        RF100_SOURCE / "train/labels",
        RF100_SOURCE / "valid/images",
        RF100_SOURCE / "valid/labels",
        DHASH_DISTANCE_THRESHOLD,
    )
    excluded = {match.validation_file for match in matches}
    source_count = len(image_files(RF100_SOURCE / "valid/images"))
    retained = create_filtered_validation_view(
        RF100_MAPPED / "images/valid",
        RF100_MAPPED / "labels/valid",
        OUTPUT / "images/valid",
        OUTPUT / "labels/valid",
        excluded,
    )
    data = {
        "path": str(OUTPUT.resolve()),
        "train": str((RF100_MAPPED / "images/train").resolve()),
        "val": "images/valid",
        "names": {index: name for index, name in enumerate(FIVE_CLASS_NAMES)},
    }
    (OUTPUT / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    details = []
    for match in matches:
        item = match.as_dict()
        item["train_annotation_names"] = [
            EXTERNAL_CLASS_NAMES[index] for index in match.train_annotation_classes
        ]
        item["validation_annotation_names"] = [
            EXTERNAL_CLASS_NAMES[index] for index in match.validation_annotation_classes
        ]
        item["annotation_classes_consistent"] = (
            match.train_annotation_classes == match.validation_annotation_classes
        )
        details.append(item)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "rf100_test_used": False,
        "audit": {
            "method": "SHA-256 exact, then deterministic 64-bit grayscale dHash",
            "dhash_hamming_threshold": DHASH_DISTANCE_THRESHOLD,
            "train_image_count": len(image_files(RF100_SOURCE / "train/images")),
            "source_validation_image_count": source_count,
            "exact_duplicate_count": sum(
                match.match_type == "exact_sha256" for match in matches
            ),
            "near_duplicate_count": sum(
                match.match_type == "near_duplicate_dhash" for match in matches
            ),
            "matches": details,
        },
        "filter": {
            "rule": "exclude validation copy when matched to RF100 train",
            "excluded_validation_files": sorted(excluded),
            "resulting_validation_image_count": retained,
        },
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
