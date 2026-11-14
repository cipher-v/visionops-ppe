"""Repository-owned configuration helpers for the YOLO26n baseline."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

EXPECTED_CLASSES = (
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
    "forklift",
    "no_vest",
)

EXPECTED_BASELINE_VALUES: dict[str, Any] = {
    "task": "detect",
    "model": "yolo26n.pt",
    "data": "data/construction-ppe/data.yaml",
    "pretrained": True,
    "epochs": 100,
    "imgsz": 640,
    "batch": 8,
    "device": 0,
    "seed": 42,
    "deterministic": True,
    "patience": 20,
    "amp": True,
    "workers": 4,
    "cache": False,
    "project": "runs",
    "name": "baseline_yolo26n_640",
}


@dataclass(frozen=True)
class BaselineConfig:
    """Validated, resolved baseline configuration."""

    task: str
    model: str
    data: Path
    pretrained: bool
    epochs: int
    imgsz: int
    batch: int
    device: int
    seed: int
    deterministic: bool
    patience: int
    amp: bool
    workers: int
    cache: bool
    project: Path
    name: str

    def as_serializable_dict(self) -> dict[str, Any]:
        """Return JSON-safe resolved configuration values."""
        values = asdict(self)
        values["data"] = str(self.data)
        values["project"] = str(self.project)
        return values

    def training_kwargs(
        self, runtime_data: Path, batch: int | None = None
    ) -> dict[str, Any]:
        """Return only the explicitly controlled Ultralytics training arguments."""
        actual_batch = self.batch if batch is None else batch
        if actual_batch not in {4, 8}:
            raise ValueError(
                "The baseline permits only batch 8 or the OOM fallback batch 4"
            )
        return {
            "data": str(runtime_data),
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": actual_batch,
            "device": self.device,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "patience": self.patience,
            "amp": self.amp,
            "workers": self.workers,
            "cache": self.cache,
            "pretrained": self.pretrained,
            "project": str(self.project),
            "name": self.name,
            "exist_ok": True,
            "plots": True,
            "save": True,
        }


def parse_baseline_config(
    values: dict[str, Any], repo_root: Path = Path(".")
) -> BaselineConfig:
    """Validate the immutable Checkpoint 2 baseline specification."""
    missing = EXPECTED_BASELINE_VALUES.keys() - values.keys()
    if missing:
        raise ValueError(f"Missing baseline configuration keys: {sorted(missing)}")

    mismatches = {
        key: {"expected": expected, "actual": values[key]}
        for key, expected in EXPECTED_BASELINE_VALUES.items()
        if values[key] != expected or type(values[key]) is not type(expected)
    }
    if mismatches:
        raise ValueError(
            f"Baseline configuration differs from specification: {mismatches}"
        )

    root = repo_root.resolve()
    return BaselineConfig(
        task=values["task"],
        model=values["model"],
        data=(root / values["data"]).resolve(),
        pretrained=values["pretrained"],
        epochs=values["epochs"],
        imgsz=values["imgsz"],
        batch=values["batch"],
        device=values["device"],
        seed=values["seed"],
        deterministic=values["deterministic"],
        patience=values["patience"],
        amp=values["amp"],
        workers=values["workers"],
        cache=values["cache"],
        project=(root / values["project"]).resolve(),
        name=values["name"],
    )


def load_baseline_config(config_path: Path) -> BaselineConfig:
    """Load and validate the baseline YAML relative to the repository root."""
    config_path = config_path.resolve()
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    repo_root = config_path.parent.parent
    return parse_baseline_config(values, repo_root)


def ordered_class_names(names: dict[int, str] | list[str]) -> tuple[str, ...]:
    """Normalize YOLO class names while preserving numeric class order."""
    if isinstance(names, list):
        return tuple(str(name) for name in names)
    if isinstance(names, dict):
        try:
            return tuple(str(names[key]) for key in sorted(names, key=int))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Dataset class names must have numeric keys") from error
    raise ValueError("Dataset class names must be a list or numeric-keyed mapping")


def prepare_runtime_dataset_yaml(source_yaml: Path, destination_yaml: Path) -> Path:
    """Write an ignored runtime YAML that points at the existing local dataset."""
    source_yaml = source_yaml.resolve()
    dataset = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise ValueError(f"Expected a YAML mapping in {source_yaml}")

    for split in ("train", "val"):
        if split not in dataset or not isinstance(dataset[split], str):
            raise ValueError(f"Dataset YAML must define a string '{split}' split")

    names = ordered_class_names(dataset.get("names"))
    if names != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected all 12 Construction-PPE classes in order, found {names}"
        )

    configured_path = Path(str(dataset.get("path", ".")))
    root_candidates = [source_yaml.parent]
    if configured_path.is_absolute():
        root_candidates.insert(0, configured_path)
    else:
        root_candidates.append(source_yaml.parent / configured_path)

    dataset_root = next(
        (
            root.resolve()
            for root in root_candidates
            if all((root / dataset[split]).is_dir() for split in ("train", "val"))
        ),
        None,
    )
    if dataset_root is None:
        raise FileNotFoundError(
            f"Could not resolve train and validation images from {source_yaml}"
        )

    runtime_dataset = {
        "path": dataset_root.as_posix(),
        "train": dataset["train"],
        "val": dataset["val"],
        "names": {index: name for index, name in enumerate(names)},
    }
    destination_yaml = destination_yaml.resolve()
    destination_yaml.parent.mkdir(parents=True, exist_ok=True)
    destination_yaml.write_text(
        yaml.safe_dump(runtime_dataset, sort_keys=False), encoding="utf-8"
    )
    return destination_yaml


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
