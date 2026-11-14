"""Train one exact, deterministic VisionOps PPE experiment configuration."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import sys
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version
from ultralytics.utils.torch_utils import get_flops

from visionops.baseline import (
    load_experiment_config,
    prepare_runtime_dataset_yaml,
    sha256_file,
)


def package_versions() -> dict[str, str]:
    """Collect the direct environment versions relevant to this experiment."""
    packages = (
        "matplotlib",
        "numpy",
        "Pillow",
        "PyYAML",
        "torch",
        "torchvision",
        "ultralytics",
    )
    return {package: metadata.version(package) for package in packages}


def write_json(path: Path, values: dict[str, Any]) -> None:
    """Write stable, human-readable JSON."""
    path.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_yolo26n_640.yaml"),
    )
    parser.add_argument(
        "--batch-override",
        type=int,
        choices=(2, 4),
        help="Use only after a genuine CUDA OOM at the next larger allowed batch.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the environment, train once, and record raw run metadata."""
    args = parse_args()
    config_path = args.config.resolve()
    config = load_experiment_config(config_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for controlled training experiments")
    if torch.cuda.device_count() <= config.device:
        raise RuntimeError(
            f"Requested CUDA device {config.device}, but found "
            f"{torch.cuda.device_count()} CUDA devices"
        )

    run_dir = config.project / config.name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix a new experiment with existing artifacts in {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_data = prepare_runtime_dataset_yaml(
        config.data, run_dir / "dataset_runtime.yaml"
    )

    model = YOLO(config.model)
    weights_path = Path(model.ckpt_path).resolve()
    gpu_name = torch.cuda.get_device_name(config.device)
    requested_batch = (
        config.batch if args.batch_override is None else args.batch_override
    )
    manifest: dict[str, Any] = {
        "status": "initialized",
        "created_at": datetime.now(UTC).isoformat(),
        "config_file": str(config_path),
        "config_sha256": sha256_file(config_path),
        "resolved_config": config.as_serializable_dict(),
        "requested_batch": requested_batch,
        "batch_override_due_to_oom": args.batch_override is not None,
        "source_dataset_yaml": str(config.data),
        "source_dataset_yaml_sha256": sha256_file(config.data),
        "runtime_dataset_yaml": str(runtime_data),
        "model_weights": str(weights_path),
        "model_weights_sha256": sha256_file(weights_path),
        "model_weights_size_bytes": weights_path.stat().st_size,
        "model_parameter_count": sum(
            parameter.numel() for parameter in model.model.parameters()
        ),
        "model_gflops": float(get_flops(model.model, imgsz=config.imgsz)),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "packages": package_versions(),
            "torch_cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "ultralytics": ultralytics_version,
            "gpu_name": gpu_name,
            "gpu_count": torch.cuda.device_count(),
        },
    }
    manifest_path = run_dir / "training_metadata.json"
    write_json(manifest_path, manifest)

    def enforce_allowed_batch(trainer: Any) -> None:
        if trainer.batch_size < 2:
            raise RuntimeError(
                "Batch 2 also exhausted CUDA memory; smaller batches are outside "
                "the controlled experiment specification"
            )

    model.add_callback("on_train_batch_start", enforce_allowed_batch)
    torch.cuda.set_device(config.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(config.device)
    start = time.perf_counter()
    try:
        manifest["status"] = "training"
        write_json(manifest_path, manifest)
        model.train(**config.training_kwargs(runtime_data, requested_batch))
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now(UTC).isoformat(),
                "error": f"{type(error).__name__}: {error}",
                "duration_seconds": time.perf_counter() - start,
                "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(
                    config.device
                ),
                "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(
                    config.device
                ),
            }
        )
        write_json(manifest_path, manifest)
        raise

    duration = time.perf_counter() - start
    trainer = model.trainer
    final_epoch = int(trainer.epoch) + 1
    best_epoch = int(trainer.stopper.best_epoch) + 1
    actual_batch = int(trainer.batch_size)
    manifest.update(
        {
            "status": "completed",
            "completed_at": datetime.now(UTC).isoformat(),
            "duration_seconds": duration,
            "requested_epochs": config.epochs,
            "final_epoch": final_epoch,
            "best_epoch": best_epoch,
            "early_stopping_occurred": final_epoch < config.epochs,
            "actual_batch": actual_batch,
            "batch_reduced_after_cuda_oom": actual_batch < requested_batch,
            "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(
                config.device
            ),
            "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(
                config.device
            ),
            "best_checkpoint": str(Path(trainer.best).resolve()),
            "last_checkpoint": str(Path(trainer.last).resolve()),
            "output_directory": str(Path(trainer.save_dir).resolve()),
        }
    )
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Final output directory: {Path(trainer.save_dir).resolve()}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
