import shutil
import uuid
from pathlib import Path

import pytest

from visionops.data_diversity import (
    CPPE_TO_FIVE,
    FIVE_CLASS_NAMES,
    RF100_TO_FIVE,
    assert_rf100_test_excluded,
    derive_split,
    diversity_success,
    exposure_matched_entries,
    reconstruct_optimizer_steps,
    remap_annotation_lines,
)


def test_five_class_mappings_are_exact() -> None:
    assert FIVE_CLASS_NAMES == ("helmet", "vest", "Person", "no_helmet", "no_vest")
    assert CPPE_TO_FIVE == {0: 0, 2: 1, 5: 2, 6: 3, 11: 4}
    assert RF100_TO_FIVE == {0: 0, 4: 1, 3: 2, 1: 3, 2: 4}
    mapped, removed = remap_annotation_lines(
        ["0 0.5 0.5 0.2 0.2", "1 0.5 0.5 0.1 0.1", "11 0.4 0.4 0.2 0.2"],
        CPPE_TO_FIVE,
    )
    assert mapped == ["0 0.5 0.5 0.2 0.2", "4 0.4 0.4 0.2 0.2"]
    assert removed == 1


def test_derived_split_remaps_without_modifying_source() -> None:
    tmp_path = Path("results") / f"test-data-diversity-{uuid.uuid4().hex}"
    try:
        source_images = tmp_path / "source/images"
        source_labels = tmp_path / "source/labels"
        source_images.mkdir(parents=True)
        source_labels.mkdir(parents=True)
        (source_images / "sample.jpg").write_bytes(b"not-decoded-by-derivation")
        original = "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n"
        (source_labels / "sample.txt").write_text(original, encoding="utf-8")
        counts = derive_split(
            source_images,
            source_labels,
            tmp_path / "derived/images",
            tmp_path / "derived/labels",
            CPPE_TO_FIVE,
        )
        assert counts.images == 1
        assert counts.retained_boxes == 1
        assert counts.removed_boxes == 1
        assert (source_labels / "sample.txt").read_text(encoding="utf-8") == original
        derived = tmp_path / "derived/labels/sample.txt"
        assert derived.read_text(encoding="utf-8") == "0 0.5 0.5 0.2 0.2\n"
    finally:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def test_exposure_control_preserves_repeated_entries() -> None:
    entries = exposure_matched_entries(["b.jpg", "a.jpg"], 5)
    assert entries == ["a.jpg", "b.jpg", "a.jpg", "b.jpg", "a.jpg"]
    assert len(entries) == 5
    assert len(set(entries)) == 2


def test_rf100_test_exclusion_guard() -> None:
    assert_rf100_test_excluded([Path("data/external/rf100/train/image.jpg")])
    with pytest.raises(ValueError, match="RF100 test is frozen"):
        assert_rf100_test_excluded([Path("data/external/rf100/test/image.jpg")])


def test_predeclared_decision_rule() -> None:
    passing = diversity_success(0.30, 0.31, 0.34, 0.65, 0.62)
    assert passing["material_success"] is True
    failing = diversity_success(0.30, 0.33, 0.34, 0.65, 0.63)
    assert failing["diversity_gain_pass"] is False
    assert failing["material_success"] is False


def test_reconstruct_optimizer_steps() -> None:
    assert reconstruct_optimizer_steps(800, 8, 1) > 0
    assert reconstruct_optimizer_steps(800, 8, 100) > 100
    with pytest.raises(ValueError):
        reconstruct_optimizer_steps(0, 8, 100)
