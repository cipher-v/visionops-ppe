"""Evaluate the best checkpoint from an exact controlled experiment."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version
from ultralytics.utils.torch_utils import get_flops

from visionops.baseline import (
    EXPECTED_BASELINE_VALUES,
    load_experiment_config,
    prepare_runtime_dataset_yaml,
)


def json_safe(value: Any) -> Any:
    """Convert NumPy-like scalars and containers to JSON-compatible values."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, values: dict[str, Any]) -> None:
    """Write stable, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(values), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def format_duration(seconds: float) -> str:
    """Format seconds as an auditable human-readable duration."""
    hours, remainder = divmod(int(round(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def copy_curated_plots(
    training_dir: Path, validation_dir: Path, assets_dir: Path
) -> list[str]:
    """Copy only lightweight plots requested for version control."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    candidates = {
        "results.png": training_dir / "results.png",
        "confusion_matrix.png": validation_dir / "confusion_matrix.png",
        "confusion_matrix_normalized.png": validation_dir
        / "confusion_matrix_normalized.png",
        "BoxPR_curve.png": validation_dir / "BoxPR_curve.png",
        "BoxF1_curve.png": validation_dir / "BoxF1_curve.png",
        "BoxP_curve.png": validation_dir / "BoxP_curve.png",
        "BoxR_curve.png": validation_dir / "BoxR_curve.png",
    }
    copied: list[str] = []
    for output_name, source in candidates.items():
        if source.is_file():
            shutil.copy2(source, assets_dir / output_name)
            copied.append(output_name)
    return copied


def build_markdown(metrics: dict[str, Any]) -> str:
    """Build the curated baseline experiment report from measured values only."""
    training = metrics["training"]
    overall = metrics["overall"]
    per_class = metrics["per_class"]
    speed = metrics["validation_speed_ms_per_image"]
    config = metrics["configuration"]
    environment = metrics["environment"]
    ranked = sorted(per_class, key=lambda row: row["map50_95"], reverse=True)
    best = ranked[:3]
    worst = list(reversed(ranked[-3:]))

    lines = [
        "# YOLO26n 640 baseline",
        "",
        "## 1. Objective",
        "",
        "Establish one reproducible pretrained YOLO26n baseline on all 12 "
        "Construction-PPE classes. No tuning or class rebalancing was performed.",
        "",
        "## 2. Exact training configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
    ]
    for key in (
        "model",
        "data",
        "epochs",
        "imgsz",
        "batch",
        "device",
        "seed",
        "deterministic",
        "patience",
        "amp",
        "workers",
        "cache",
    ):
        lines.append(f"| `{key}` | `{config[key]}` |")
    lines.extend(
        [
            "",
            "All unlisted optimizer, learning-rate schedule, augmentation, loss, "
            "and mosaic settings used Ultralytics 8.4.212 defaults.",
            "",
            "## 3. Hardware and software environment",
            "",
            f"- GPU: {environment['gpu_name']}",
            f"- Python: {environment['python_version']}",
            f"- PyTorch: {environment['torch_version']}",
            f"- PyTorch CUDA runtime: {environment['torch_cuda_runtime']}",
            f"- Ultralytics: {environment['ultralytics_version']}",
            f"- Platform: {environment['platform']}",
            "",
            "## 4. Overall validation metrics",
            "",
            "| Precision | Recall | mAP50 | mAP50-95 |",
            "| ---: | ---: | ---: | ---: |",
            f"| {overall['precision']:.6f} | {overall['recall']:.6f} | "
            f"{overall['map50']:.6f} | {overall['map50_95']:.6f} |",
            "",
            "## 5. Per-class metrics",
            "",
            "| Class | Precision | Recall | AP50 | AP50-95 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in per_class:
        lines.append(
            f"| `{row['class_name']}` | {row['precision']:.6f} | "
            f"{row['recall']:.6f} | {row['ap50']:.6f} | "
            f"{row['map50_95']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 6. Training behavior",
            "",
            f"- Actual batch size: {training['actual_batch']}",
            f"- Final epoch reached: {training['final_epoch']}",
            f"- Best epoch: {training['best_epoch']}",
            f"- Early stopping occurred: {training['early_stopping_occurred']}",
            f"- Training duration: {format_duration(training['duration_seconds'])}",
            f"- Peak allocated GPU memory: "
            f"{training['peak_gpu_memory_allocated_gib']:.3f} GiB",
            f"- Validation speed: preprocess {speed.get('preprocess', 0.0):.3f} "
            f"ms, inference {speed.get('inference', 0.0):.3f} ms, "
            f"loss {speed.get('loss', 0.0):.3f} ms, postprocess "
            f"{speed.get('postprocess', 0.0):.3f} ms per image",
            "",
            "## 7. Best-performing classes",
            "",
        ]
    )
    lines.extend(
        f"- `{row['class_name']}`: AP50-95 {row['map50_95']:.6f}" for row in best
    )
    lines.extend(["", "## 8. Worst-performing classes", ""])
    lines.extend(
        f"- `{row['class_name']}`: AP50-95 {row['map50_95']:.6f}" for row in worst
    )
    lines.extend(
        [
            "",
            "## 9. Observations",
            "",
            f"- The measured validation mAP50-95 is {overall['map50_95']:.6f}.",
            f"- `{best[0]['class_name']}` has the highest measured AP50-95; "
            f"`{worst[0]['class_name']}` has the lowest.",
            "- The earlier dataset audit found that 67.34% of boxes have relative "
            "area below 1%, and 27.17% have relative area below 0.1%. This "
            "baseline does not attempt to address that distribution.",
            "",
            "## 10. Evaluation scope",
            "",
            "These metrics use the official validation split, which is development "
            "data. It is not an independent final test set and must not be treated "
            "as a deployment-performance estimate.",
            "",
        ]
    )
    if metrics["curated_plots"]:
        lines.extend(["## Curated plots", ""])
        for plot in metrics["curated_plots"]:
            lines.append(f"![{plot}](../../assets/baseline/{plot})")
            lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_yolo26n_640.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    """Evaluate best.pt and write curated metrics and plots."""
    args = parse_args()
    config = load_experiment_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for experiment validation")

    training_dir = config.project / config.name
    metadata_path = training_dir / "training_metadata.json"
    best_checkpoint = training_dir / "weights" / "best.pt"
    if not metadata_path.is_file() or not best_checkpoint.is_file():
        raise FileNotFoundError(
            "Completed training metadata and weights/best.pt are required"
        )
    training_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if training_metadata.get("status") != "completed":
        raise RuntimeError("Training metadata does not report successful completion")

    validation_name = f"{config.name}_best_val"
    validation_dir = config.project / validation_name
    if validation_dir.exists() and any(validation_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to mix validation artifacts in existing {validation_dir}"
        )
    validation_dir.mkdir(parents=True, exist_ok=True)
    runtime_data = prepare_runtime_dataset_yaml(
        config.data, validation_dir / "dataset_runtime.yaml"
    )

    model = YOLO(best_checkpoint)
    started = time.perf_counter()
    validation_metrics = model.val(
        data=str(runtime_data),
        split="val",
        imgsz=config.imgsz,
        batch=config.batch,
        device=config.device,
        workers=config.workers,
        plots=True,
        project=str(config.project),
        name=validation_name,
        exist_ok=True,
        seed=config.seed,
        deterministic=config.deterministic,
    )
    validation_duration = time.perf_counter() - started

    precision, recall, map50, map50_95 = map(float, validation_metrics.mean_results())
    per_class = []
    for class_id, row in enumerate(validation_metrics.summary(decimals=10)):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": row["Class"],
                "images": int(row["Images"]),
                "instances": int(row["Instances"]),
                "precision": float(row["Box-P"]),
                "recall": float(row["Box-R"]),
                "ap50": float(row["mAP50"]),
                "map50_95": float(row["mAP50-95"]),
            }
        )

    is_baseline = config.name == EXPECTED_BASELINE_VALUES["name"]
    output_names = {
        "baseline_yolo26n_640": "baseline",
        "experiment_yolo26n_960": "resolution_960",
        "experiment_yolo26s_960": "model_capacity_960",
    }
    asset_name = output_names[config.name]
    assets_dir = Path("docs/assets", asset_name).resolve()
    curated_plots = copy_curated_plots(training_dir, validation_dir, assets_dir)
    metrics: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": str(best_checkpoint.resolve()),
        "configuration": {
            **config.as_serializable_dict(),
            "batch": int(training_metadata["actual_batch"]),
        },
        "environment": {
            "gpu_name": torch.cuda.get_device_name(config.device),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "ultralytics_version": ultralytics_version,
        },
        "training": {
            "actual_batch": int(training_metadata["actual_batch"]),
            "requested_epochs": int(training_metadata["requested_epochs"]),
            "final_epoch": int(training_metadata["final_epoch"]),
            "best_epoch": int(training_metadata["best_epoch"]),
            "early_stopping_occurred": bool(
                training_metadata["early_stopping_occurred"]
            ),
            "duration_seconds": float(training_metadata["duration_seconds"]),
            "peak_gpu_memory_allocated_gib": float(
                training_metadata["peak_gpu_memory_allocated_bytes"]
            )
            / 1024**3,
            "peak_gpu_memory_reserved_gib": float(
                training_metadata["peak_gpu_memory_reserved_bytes"]
            )
            / 1024**3,
            "model_parameter_count": sum(
                parameter.numel() for parameter in model.model.parameters()
            ),
            "model_gflops": float(get_flops(model.model, imgsz=config.imgsz)),
            "pretrained_weights_size_bytes": int(
                training_metadata["model_weights_size_bytes"]
            ),
            "best_checkpoint_size_bytes": best_checkpoint.stat().st_size,
        },
        "overall": {
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
        },
        "per_class": per_class,
        "validation_speed_ms_per_image": json_safe(validation_metrics.speed),
        "validation_duration_seconds": validation_duration,
        "validation_output_directory": str(validation_dir.resolve()),
        "curated_plots": curated_plots,
    }

    output_name = output_names[config.name]
    output_dir = Path("docs/experiments", output_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_name = "baseline_metrics.json" if is_baseline else "metrics.json"
    write_json(output_dir / metrics_name, metrics)
    if is_baseline:
        (output_dir / "baseline.md").write_text(
            build_markdown(metrics), encoding="utf-8"
        )
    print(json.dumps(json_safe(metrics), indent=2, sort_keys=True))
    print(f"Curated experiment results: {output_dir}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
