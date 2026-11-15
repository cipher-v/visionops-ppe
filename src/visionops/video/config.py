"""Load and verify the frozen production detector configuration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

EXPECTED_CLASSES = {
    0: "helmet",
    1: "vest",
    2: "Person",
    3: "no_helmet",
    4: "no_vest",
}
EXPECTED_THRESHOLDS = {
    "helmet": 0.32,
    "vest": 0.19,
    "Person": 0.41,
    "no_helmet": 0.27,
    "no_vest": 0.11,
}


@dataclass(frozen=True)
class FrozenDetectorConfig:
    """Validated, repository-owned view of the production detector contract."""

    architecture: str
    manifest_path: Path
    weights_path: Path
    weights_sha256: str
    imgsz: int
    classes: dict[int, str]
    confidence_thresholds: dict[str, float]
    global_confidence_threshold: float
    nms_iou: float
    preprocessing: dict[str, object]

    @property
    def inference_confidence_floor(self) -> float:
        """Lowest threshold Ultralytics may use before class-aware filtering."""
        return min(self.confidence_thresholds.values())


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA-256 digest without modifying the file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_weights(path: Path, expected_sha256: str) -> str:
    """Fail loudly when frozen weights are absent or do not match the manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Frozen detector weights do not exist: {path}")
    actual = sha256_file(path)
    expected = expected_sha256.upper()
    if actual != expected:
        raise ValueError(
            "Frozen detector weight hash mismatch: "
            f"expected {expected}, found {actual} at {path}"
        )
    return actual


def load_frozen_detector_config(
    config_path: Path, repository_root: Path | None = None
) -> FrozenDetectorConfig:
    """Load and strictly validate the immutable production-v1 detector settings."""
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Production detector config does not exist: {config_path}"
        )
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Production detector config must be a YAML mapping")

    try:
        model = values["model"]
        input_config = values["input"]
        classes = {int(key): value for key, value in values["classes"].items()}
        policy = values["confidence_policy"]
        thresholds = {
            str(name): float(value) for name, value in policy["per_class"].items()
        }
        architecture = str(model["architecture"])
        manifest_value = str(model["manifest"])
        weights_value = str(model["weights"])
        weights_sha256 = str(model["sha256"]).upper()
        imgsz = int(input_config["imgsz"])
        nms_iou = float(values["nms_iou"])
        global_threshold = float(policy["global_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Malformed production detector config: {error}") from error

    if architecture != "YOLO26n detect":
        raise ValueError(f"Frozen architecture changed: {architecture}")
    if imgsz != 960:
        raise ValueError(f"Frozen input resolution changed: {imgsz}")
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Frozen class vocabulary changed: {classes}")
    if policy.get("type") != "class_specific":
        raise ValueError("Frozen confidence policy must be class_specific")
    if thresholds != EXPECTED_THRESHOLDS:
        raise ValueError(f"Frozen confidence thresholds changed: {thresholds}")
    if global_threshold != 0.27:
        raise ValueError(f"Frozen global threshold changed: {global_threshold}")
    if nms_iou != 0.70:
        raise ValueError(f"Frozen NMS IoU changed: {nms_iou}")

    root = repository_root.resolve() if repository_root else config_path.parents[1]
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Frozen model manifest does not exist: {manifest_path}"
        )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Frozen model manifest must be a YAML mapping")
    if str(manifest.get("weights_sha256", "")).upper() != weights_sha256:
        raise ValueError("Production config and model manifest hashes differ")
    if str(manifest.get("weights", "")) != weights_value:
        raise ValueError("Production config and model manifest weight paths differ")
    if str(manifest.get("architecture", "")) != architecture:
        raise ValueError("Production config and model manifest architectures differ")

    weights_path = Path(weights_value)
    if not weights_path.is_absolute():
        weights_path = root / weights_path
    preprocessing = values.get("preprocessing", {})
    if not isinstance(preprocessing, dict):
        raise ValueError("Preprocessing configuration must be a mapping")
    return FrozenDetectorConfig(
        architecture=architecture,
        manifest_path=manifest_path.resolve(),
        weights_path=weights_path.resolve(),
        weights_sha256=weights_sha256,
        imgsz=imgsz,
        classes=classes,
        confidence_thresholds=thresholds,
        global_confidence_threshold=global_threshold,
        nms_iou=nms_iou,
        preprocessing=preprocessing,
    )
