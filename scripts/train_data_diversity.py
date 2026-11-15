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
from visionops.exposure_control import exposure_count, should_controlled_stop

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
    "experiment_mixed5_exposure_matched_yolo26n_960": {
        "config": "configs/experiment_mixed5_exposure_matched_yolo26n_960.yaml",
        "data": "data/generated/mixed-five-class/mixed_data.yaml",
        "manifest_key": "mixed_domain",
        "config_overrides": {"patience": 0, "controlled_stop_epoch": 45},
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
    expected = {
        **COMMON,
        "data": specification["data"],
        "name": condition,
        **specification.get("config_overrides", {}),
    }
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
    reference_c_metadata = None
    if controlled_stop_epoch := config.get("controlled_stop_epoch"):
        reference_path = (
            ROOT / "runs/experiment_mixed5_yolo26n_960/training_metadata.json"
        )
        reference_c_metadata = json.loads(reference_path.read_text(encoding="utf-8"))
    run_dir = ROOT / config["project"] / config["name"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    batch = args.batch_override or config["batch"]
    model = YOLO(str(ROOT / config["model"]))
    optimizer_steps = 0
    controlled_stop_epoch = config.get("controlled_stop_epoch")
    controlled_stop_requested = False

    def instrument_optimizer_steps(trainer: Any) -> None:
        nonlocal optimizer_steps
        original_optimizer_step = trainer.optimizer_step

        def counted_optimizer_step() -> None:
            nonlocal optimizer_steps
            original_optimizer_step()
            optimizer_steps += 1

        trainer.optimizer_step = counted_optimizer_step

    model.add_callback("on_train_start", instrument_optimizer_steps)

    def request_controlled_stop(trainer: Any) -> None:
        nonlocal controlled_stop_requested
        if controlled_stop_epoch is None:
            return
        completed_epoch = int(trainer.epoch) + 1
        if should_controlled_stop(completed_epoch, controlled_stop_epoch):
            controlled_stop_requested = True
            trainer.stop = True

    if controlled_stop_epoch is not None:
        model.add_callback("on_fit_epoch_end", request_controlled_stop)
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
        "configured_scheduler_epochs": config["epochs"],
        "controlled_stop_epoch": controlled_stop_epoch,
        "early_stopping_disabled": config["patience"] == 0,
        "controlled_stop_callback": (
            "on_fit_epoch_end" if controlled_stop_epoch is not None else None
        ),
        "controlled_stop_lifecycle": (
            "after validation, metric writing, and checkpoint save"
            if controlled_stop_epoch is not None
            else None
        ),
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
    results_path = Path(trainer.csv)
    result_lines = results_path.read_text(encoding="utf-8").splitlines()
    result_rows = len(result_lines) - 1
    human_epochs = [int(line.split(",", 1)[0].strip()) for line in result_lines[1:]]
    close_mosaic = int(trainer.args.close_mosaic)
    close_mosaic_start_epoch = int(trainer.args.epochs) - close_mosaic
    acceptance = {
        "configured_scheduler_epochs_is_100": int(trainer.args.epochs) == 100,
        "completed_epochs_is_controlled_stop": (
            controlled_stop_epoch is None or epochs_completed == controlled_stop_epoch
        ),
        "results_rows_match_completed_epochs": result_rows == epochs_completed,
        "last_human_epoch_matches_completed": (
            bool(human_epochs) and human_epochs[-1] == epochs_completed
        ),
        "epoch_46_absent": (controlled_stop_epoch is None or 46 not in human_epochs),
        "controlled_stop_requested": (
            controlled_stop_epoch is None or controlled_stop_requested
        ),
        "close_mosaic_not_triggered": (
            controlled_stop_epoch is None
            or close_mosaic_start_epoch > controlled_stop_epoch
        ),
        "rf100_test_used_is_false": metadata["rf100_test_used"] is False,
        "training_data_matches_config": str(data_path) == metadata["data_path"],
        "training_data_matches_c": (
            reference_c_metadata is None
            or metadata["data_sha256"] == reference_c_metadata["data_sha256"]
        ),
        "pretrained_weights_match_c": (
            reference_c_metadata is None
            or metadata["pretrained_weights_sha256"]
            == reference_c_metadata["pretrained_weights_sha256"]
        ),
        "seed_model_resolution_match_c": (
            reference_c_metadata is None
            or all(
                config[key] == reference_c_metadata["config"][key]
                for key in ("seed", "model", "imgsz")
            )
        ),
    }
    if not all(acceptance.values()):
        metadata.update(
            {
                "status": "invalid_control",
                "acceptance_checks": acceptance,
                "epochs_completed": epochs_completed,
                "optimizer_steps": optimizer_steps,
            }
        )
        write_json(metadata_path, metadata)
        raise RuntimeError(f"Controlled-stop acceptance failed: {acceptance}")
    metadata.update(
        {
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "duration_seconds": time.perf_counter() - start,
            "epochs_completed": epochs_completed,
            "early_stopping_occurred": False,
            "controlled_stop_occurred": controlled_stop_requested,
            "stop_reason": (
                f"controlled stop after epoch {controlled_stop_epoch}"
                if controlled_stop_requested
                else "configured epoch horizon or early stopping"
            ),
            "actual_batch": int(trainer.batch_size),
            "estimated_total_image_exposures": exposure_count(
                composition["effective_entries_per_epoch"], epochs_completed
            ),
            "optimizer_steps": optimizer_steps,
            "best_epoch": int(trainer.stopper.best_epoch) + 1,
            "best_checkpoint": str(Path(trainer.best).resolve()),
            "last_checkpoint": str(Path(trainer.last).resolve()),
            "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "results_csv": str(results_path.resolve()),
            "results_csv_epoch_rows": result_rows,
            "last_human_readable_epoch": human_epochs[-1],
            "close_mosaic_epochs": close_mosaic,
            "close_mosaic_start_epoch": close_mosaic_start_epoch,
            "mosaic_closed_during_training": (
                epochs_completed > close_mosaic_start_epoch
            ),
            "acceptance_checks": acceptance,
        }
    )
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
