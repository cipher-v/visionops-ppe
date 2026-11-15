import shutil
import uuid
from pathlib import Path

import pytest
from PIL import Image

from visionops.exposure_control import (
    audit_train_validation_duplicates,
    create_filtered_validation_view,
    exposure_count,
    should_controlled_stop,
)


def _image(path: Path, invert: bool = False) -> None:
    image = Image.new("L", (12, 12))
    image.putdata(
        [
            (255 - (row * 12 + column) if invert else row * 12 + column)
            for row in range(12)
            for column in range(12)
        ]
    )
    image.save(path)


def test_duplicate_audit_reports_classes_and_distance() -> None:
    root = Path("results") / f"test-exposure-audit-{uuid.uuid4().hex}"
    try:
        for split in ("train", "valid"):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)
        _image(root / "train/images/a.png")
        _image(root / "valid/images/b.png")
        (root / "train/labels/a.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.4 0.4\n", encoding="utf-8"
        )
        (root / "valid/labels/b.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n4 0.5 0.5 0.3 0.3\n", encoding="utf-8"
        )
        matches = audit_train_validation_duplicates(
            root / "train/images",
            root / "train/labels",
            root / "valid/images",
            root / "valid/labels",
            4,
        )
        assert len(matches) == 1
        assert matches[0].dhash_distance == 0
        assert matches[0].train_annotation_classes == (0, 3)
        assert matches[0].validation_annotation_classes == (0, 4)
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_filtered_view_is_deterministic_and_preserves_source() -> None:
    root = Path("results") / f"test-filtered-view-{uuid.uuid4().hex}"
    try:
        source_images = root / "source/images"
        source_labels = root / "source/labels"
        source_images.mkdir(parents=True)
        source_labels.mkdir(parents=True)
        for name in ("a", "b"):
            _image(source_images / f"{name}.png", invert=name == "b")
            (source_labels / f"{name}.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )
        before = (source_labels / "a.txt").read_bytes()
        retained = create_filtered_validation_view(
            source_images,
            source_labels,
            root / "filtered/images",
            root / "filtered/labels",
            {"b.png"},
        )
        assert retained == 1
        assert [path.name for path in (root / "filtered/images").iterdir()] == ["a.png"]
        assert (source_labels / "a.txt").read_bytes() == before
        assert (source_images / "b.png").is_file()
    finally:
        if root.exists():
            shutil.rmtree(root)


def test_controlled_stop_and_exposure_accounting() -> None:
    assert not should_controlled_stop(44, 45)
    assert should_controlled_stop(45, 45)
    assert exposure_count(1797, 45) == 80865
    with pytest.raises(ValueError):
        should_controlled_stop(0, 45)
    with pytest.raises(ValueError):
        exposure_count(1797, 0)
