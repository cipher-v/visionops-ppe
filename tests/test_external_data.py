from visionops.external_data import (
    DHASH_DISTANCE_THRESHOLD,
    EXTERNAL_TO_INTERNAL,
    hamming_distance,
    map_annotation_line,
)


def test_external_class_mapping_is_explicit_and_complete() -> None:
    assert EXTERNAL_TO_INTERNAL == {0: 0, 1: 6, 2: 11, 3: 5, 4: 2}


def test_annotation_mapping_preserves_coordinates() -> None:
    assert map_annotation_line("2 0.5 0.4 0.3 0.2") == "11 0.5 0.4 0.3 0.2"


def test_hamming_distance_and_threshold() -> None:
    assert hamming_distance(0b1010, 0b0011) == 2
    assert DHASH_DISTANCE_THRESHOLD == 4


def test_annotation_mapping_rejects_bad_shape() -> None:
    try:
        map_annotation_line("0 0.5 0.5")
    except ValueError as error:
        assert "five YOLO fields" in str(error)
    else:
        raise AssertionError("Malformed annotation was accepted")
