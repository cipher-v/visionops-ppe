from copy import deepcopy
from pathlib import Path

import pytest

from visionops.baseline import (
    EXPECTED_BASELINE_VALUES,
    EXPECTED_CLASSES,
    EXPECTED_MODEL_CAPACITY_960_VALUES,
    EXPECTED_RESOLUTION_960_VALUES,
    ordered_class_names,
    parse_baseline_config,
    parse_experiment_config,
)


def test_parse_baseline_config_accepts_checkpoint_specification() -> None:
    config = parse_baseline_config(deepcopy(EXPECTED_BASELINE_VALUES), Path("."))

    assert config.model == "yolo26n.pt"
    assert config.epochs == 100
    assert config.batch == 8
    assert config.imgsz == 640


def test_parse_baseline_config_rejects_unplanned_tuning() -> None:
    values = deepcopy(EXPECTED_BASELINE_VALUES)
    values["imgsz"] = 1280

    with pytest.raises(ValueError, match="Experiment configuration differs"):
        parse_baseline_config(values)


def test_parse_resolution_experiment_changes_only_resolution_and_name() -> None:
    config = parse_experiment_config(
        deepcopy(EXPECTED_RESOLUTION_960_VALUES), Path(".")
    )

    assert config.imgsz == 960
    assert config.name == "experiment_yolo26n_960"
    assert config.batch == 8


def test_parse_capacity_experiment_changes_only_model_and_name() -> None:
    config = parse_experiment_config(
        deepcopy(EXPECTED_MODEL_CAPACITY_960_VALUES), Path(".")
    )

    assert config.model == "yolo26s.pt"
    assert config.imgsz == 960
    assert config.name == "experiment_yolo26s_960"
    assert config.training_kwargs(Path("runtime.yaml"), batch=2)["batch"] == 2


def test_ordered_class_names_requires_all_expected_classes() -> None:
    names = {index: name for index, name in enumerate(EXPECTED_CLASSES)}

    assert ordered_class_names(names) == EXPECTED_CLASSES
