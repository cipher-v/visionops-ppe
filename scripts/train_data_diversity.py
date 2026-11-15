"""Train one immutable Checkpoint 7 five-class condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version

from visionops.data_diversity import FIVE_CLASS_NAMES, assert_rf100_test_excluded

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/generated/data-diversity-manifest.json"
COMMON = {
    "task": "detect",
    "model": "yolo26n.pt",
    "pretrained": True,
    "epochs": 100,
    "imgsz": 960,
    "batch": 8,
    "device": 0,
    "seed": 42,
    "deterministic": True,
    "patience": 20,
    "amp": True,
    "workers": 4,
    "cache": False,
    "project": "runs",
}
CONDITIONS = {
    "experiment_cppe5_yolo26n_960": {
        "config": "configs/experiment_cppe5_yolo26n_960.yaml",
        "data": "data/generated/cppe-five-class/data.yaml",
        "manifest_key": "cppe5_baseline",
    },
    "experiment_cppe5_quantity_control_yolo26n_960": {
        "config": "configs/experiment_cppe5_quantity_control_yolo26n_960.yaml",
        "data": "data/generated/mixed-five-class/quantity_data.yaml",
        "manifest_key": "quantity_control",
    },
    "experiment_mixed5_yolo26n_960": {
        "config": "configs/experiment_mixed5_yolo26n_960.yaml",
        "data": "data/generated/mixed-five-class/mixed_data.yaml",
        "manifest_key": "mixed_domain",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, values: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", choices=tuple(CONDITIONS))
    parser.add_argument("--batch-override", type=int, choices=(4, 2))
    return parser.parse_args()


def load_config(condition: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    specification = CONDITIONS[condition]
    config_path = ROOT / specification["config"]
    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {**COMMON, "data": specification["data"], "name": condition}
    if values != expected:
        raise ValueError(f"Config differs from immutable specification: {values}")
    return config_path, values, specification


def main() -> None:
    args = parse_args()
    config_path, config, specification = load_config(args.condition)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for controlled training")
    data_path = (ROOT / config["data"]).resolve()
    assert_rf100_test_excluded([data_path])
    dataset = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if tuple(dataset["names"].values()) != FIVE_CLASS_NAMES:
        raise ValueError("Dataset does not use the immutable five-class vocabulary")
    assert_rf100_test_excluded([Path(dataset["train"]), Path(dataset["val"])])

    source_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if source_manifest["rf100_test_used"] is not False:
        raise RuntimeError("Generated-data manifest violates the RF100-test freeze")
    composition = source_manifest["conditions"][specification["manifest_key"]]
    run_dir = ROOT / config["project"] / config["name"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    batch = args.batch_override or config["batch"]
    model = YOLO(str(ROOT / config["model"]))
    optimizer_steps = 0

    def instrument_optimizer_steps(trainer: Any) -> None:
        nonlocal optimizer_steps
        original_optimizer_step = trainer.optimizer_step

        def counted_optimizer_step() -> None:
            nonlocal optimizer_steps
            original_optimizer_step()
            optimizer_steps += 1

        trainer.optimizer_step = counted_optimizer_step

    model.add_callback("on_train_start", instrument_optimizer_steps)
    metadata: dict[str, Any] = {
        "status": "initialized",
        "created_at": datetime.now(UTC).isoformat(),
        "condition": specification["manifest_key"],
        "config": config,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "data_path": str(data_path),
        "data_sha256": sha256(data_path),
        "rf100_test_used": False,
        "unique_training_images": composition["unique_training_images"],
        "effective_entries_per_epoch": composition["effective_entries_per_epoch"],
        "requested_batch": batch,
        "batch_override_due_to_oom": args.batch_override is not None,
        "pretrained_weights": str((ROOT / config["model"]).resolve()),
        "pretrained_weights_sha256": sha256(ROOT / config["model"]),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "ultralytics": ultralytics_version,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    metadata_path = run_dir / "training_metadata.json"
    write_json(metadata_path, metadata)
    training_arguments = {
        key: config[key]
        for key in (
            "epochs",
            "imgsz",
            "device",
            "seed",
            "deterministic",
            "patience",
            "amp",
            "workers",
            "cache",
            "pretrained",
        )
    } | {
        "data": str(data_path),
        "batch": batch,
        "project": str((ROOT / config["project"]).resolve()),
        "name": config["name"],
        "exist_ok": True,
        "plots": True,
        "save": True,
    }
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    start = time.perf_counter()
    try:
        metadata["status"] = "training"
        write_json(metadata_path, metadata)
        model.train(**training_arguments)
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "duration_seconds": time.perf_counter() - start,
                "optimizer_steps": optimizer_steps,
            }
        )
        write_json(metadata_path, metadata)
        raise

    trainer = model.trainer
    epochs_completed = int(trainer.epoch) + 1
    metadata.update(
        {
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "duration_seconds": time.perf_counter() - start,
            "epochs_completed": epochs_completed,
            "early_stopping_occurred": epochs_completed < config["epochs"],
            "actual_batch": int(trainer.batch_size),
            "estimated_total_image_exposures": (
                composition["effective_entries_per_epoch"] * epochs_completed
            ),
            "optimizer_steps": optimizer_steps,
            "best_epoch": int(trainer.stopper.best_epoch) + 1,
            "best_checkpoint": str(Path(trainer.best).resolve()),
            "last_checkpoint": str(Path(trainer.last).resolve()),
            "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(0),
        }
    )
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
