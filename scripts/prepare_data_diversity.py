"""Build deterministic five-class datasets for Checkpoint 7."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from visionops.data_diversity import (
    CPPE_TO_FIVE,
    FIVE_CLASS_NAMES,
    RF100_TO_FIVE,
    assert_rf100_test_excluded,
    derive_split,
    exposure_matched_entries,
)

ROOT = Path(__file__).resolve().parents[1]
CPPE_SOURCE = ROOT / "data/construction-ppe"
RF100_SOURCE = ROOT / "data/external/construction-safety-gsnvb-v2"
GENERATED = ROOT / "data/generated"
CPPE_OUTPUT = GENERATED / "cppe-five-class"
RF100_OUTPUT = GENERATED / "rf100-five-class"
MIXED_OUTPUT = GENERATED / "mixed-five-class"
MANIFEST = GENERATED / "data-diversity-manifest.json"


def write_yaml(path: Path, values: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def require_new(paths: tuple[Path, ...]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite derived data: {existing}")


def absolute_images(directory: Path) -> list[str]:
    return sorted(str(path.resolve()) for path in directory.iterdir() if path.is_file())


def main() -> None:
    assert_rf100_test_excluded(
        [
            RF100_SOURCE / "train",
            RF100_SOURCE / "valid",
            CPPE_SOURCE / "images/train",
            CPPE_SOURCE / "images/val",
        ]
    )
    require_new((CPPE_OUTPUT, RF100_OUTPUT, MIXED_OUTPUT, MANIFEST))
    GENERATED.mkdir(parents=True, exist_ok=True)

    cppe_counts = {}
    for split in ("train", "val"):
        cppe_counts[split] = derive_split(
            CPPE_SOURCE / "images" / split,
            CPPE_SOURCE / "labels" / split,
            CPPE_OUTPUT / "images" / split,
            CPPE_OUTPUT / "labels" / split,
            CPPE_TO_FIVE,
        )
    write_yaml(
        CPPE_OUTPUT / "data.yaml",
        {
            "path": str(CPPE_OUTPUT.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {index: name for index, name in enumerate(FIVE_CLASS_NAMES)},
        },
    )

    rf100_counts = {}
    for split in ("train", "valid"):
        rf100_counts[split] = derive_split(
            RF100_SOURCE / split / "images",
            RF100_SOURCE / split / "labels",
            RF100_OUTPUT / "images" / split,
            RF100_OUTPUT / "labels" / split,
            RF100_TO_FIVE,
        )
    write_yaml(
        RF100_OUTPUT / "data.yaml",
        {
            "path": str(RF100_OUTPUT.resolve()),
            "train": "images/train",
            "val": "images/valid",
            "names": {index: name for index, name in enumerate(FIVE_CLASS_NAMES)},
        },
    )

    MIXED_OUTPUT.mkdir(parents=True)
    cppe_train = absolute_images(CPPE_OUTPUT / "images/train")
    rf100_train = absolute_images(RF100_OUTPUT / "images/train")
    target_entries = len(cppe_train) + len(rf100_train)
    quantity_entries = exposure_matched_entries(cppe_train, target_entries)
    mixed_entries = [*cppe_train, *rf100_train]
    (MIXED_OUTPUT / "train_quantity.txt").write_text(
        "\n".join(quantity_entries) + "\n", encoding="utf-8"
    )
    (MIXED_OUTPUT / "train_mixed.txt").write_text(
        "\n".join(mixed_entries) + "\n", encoding="utf-8"
    )
    common = {
        "path": str(ROOT.resolve()),
        "val": str((CPPE_OUTPUT / "images/val").resolve()),
        "names": {index: name for index, name in enumerate(FIVE_CLASS_NAMES)},
    }
    write_yaml(
        MIXED_OUTPUT / "quantity_data.yaml",
        {**common, "train": str((MIXED_OUTPUT / "train_quantity.txt").resolve())},
    )
    write_yaml(
        MIXED_OUTPUT / "mixed_data.yaml",
        {**common, "train": str((MIXED_OUTPUT / "train_mixed.txt").resolve())},
    )

    manifest = {
        "classes": list(FIVE_CLASS_NAMES),
        "mapping": {
            "construction_ppe": {
                str(key): value for key, value in CPPE_TO_FIVE.items()
            },
            "rf100": {str(key): value for key, value in RF100_TO_FIVE.items()},
        },
        "construction_ppe": {
            split: counts.__dict__ for split, counts in cppe_counts.items()
        },
        "rf100": {split: counts.__dict__ for split, counts in rf100_counts.items()},
        "conditions": {
            "cppe5_baseline": {
                "unique_training_images": len(cppe_train),
                "effective_entries_per_epoch": len(cppe_train),
            },
            "quantity_control": {
                "unique_training_images": len(set(quantity_entries)),
                "effective_entries_per_epoch": len(quantity_entries),
                "repeat_entries": len(quantity_entries) - len(set(quantity_entries)),
            },
            "mixed_domain": {
                "unique_training_images": len(set(mixed_entries)),
                "effective_entries_per_epoch": len(mixed_entries),
                "cppe_images": len(cppe_train),
                "rf100_images": len(rf100_train),
            },
        },
        "rf100_test_used": False,
        "duplicate_entry_verification": {
            "ultralytics_file": ".venv/Lib/site-packages/ultralytics/data/base.py",
            "behavior": "get_img_files sorts text entries without set/deduplication",
            "quantity_list_entries": len(quantity_entries),
            "quantity_list_unique": len(set(quantity_entries)),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
