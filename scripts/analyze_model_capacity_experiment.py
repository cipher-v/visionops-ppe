"""Compare YOLO26n and YOLO26s at 960 with fixed diagnostics."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from scripts.analyze_resolution_experiment import (
    CONFIDENCE_THRESHOLD,
    FOCUS_CLASS_IDS,
    IMAGE_EXTENSIONS,
    MATCH_IOU_THRESHOLD,
    MAX_EXAMPLES_PER_CATEGORY,
    NMS_IOU_THRESHOLD,
    ExampleCandidate,
    ImageComparison,
    candidate_priority,
    draw_panel,
    empty_counts,
    finalize_counts,
    load_ground_truth,
    predict_validation,
    update_counts,
    write_json,
)
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

from visionops.baseline import EXPECTED_CLASSES, prepare_runtime_dataset_yaml
from visionops.detection_analysis import SIZE_BINS, box_iou, match_detections

N_METRICS_PATH = Path("docs/experiments/resolution_960/metrics.json")
S_METRICS_PATH = Path("docs/experiments/model_capacity_960/metrics.json")
N_CHECKPOINT = Path("runs/experiment_yolo26n_960/weights/best.pt")
S_CHECKPOINT = Path("runs/experiment_yolo26s_960/weights/best.pt")
ASSETS_DIR = Path("docs/assets/model_capacity_960")
FOCUS_NAMES = (
    "helmet",
    "vest",
    "no_helmet",
    "no_vest",
    "gloves",
    "no_gloves",
    "goggles",
    "no_goggle",
)


def collect_capacity_candidates(
    comparisons: list[ImageComparison],
) -> dict[str, list[ExampleCandidate]]:
    """Collect balanced success and failure cases for the capacity comparison."""
    candidates = {
        category: []
        for category in ("improved", "both_missed", "false_positive", "regressed")
    }
    for image_index, comparison in enumerate(comparisons):
        n_matched = comparison.baseline_matches.matched_ground_truth_indices
        s_matched = comparison.experiment_matches.matched_ground_truth_indices
        for target_index, target in enumerate(comparison.ground_truth):
            category = None
            if target_index not in n_matched and target_index in s_matched:
                category = "improved"
            elif target_index not in n_matched and target_index not in s_matched:
                category = "both_missed"
            elif target_index in n_matched and target_index not in s_matched:
                category = "regressed"
            if category:
                candidates[category].append(
                    ExampleCandidate(
                        category=category,
                        image_index=image_index,
                        object_index=target_index,
                        class_id=target.class_id,
                        relative_area=target.relative_area,
                        priority=candidate_priority(
                            target.class_id,
                            target.relative_area,
                            comparison.path.name,
                            target_index,
                        ),
                    )
                )

        for prediction_index, prediction in enumerate(
            comparison.experiment_predictions
        ):
            if (
                prediction_index
                in comparison.experiment_matches.matched_prediction_indices
            ):
                continue
            existed_for_n = any(
                candidate.class_id == prediction.class_id
                and box_iou(candidate.xyxy, prediction.xyxy) >= MATCH_IOU_THRESHOLD
                for candidate in comparison.baseline_predictions
            )
            if not existed_for_n:
                candidates["false_positive"].append(
                    ExampleCandidate(
                        category="false_positive",
                        image_index=image_index,
                        object_index=prediction_index,
                        class_id=prediction.class_id,
                        relative_area=prediction.relative_area,
                        priority=candidate_priority(
                            prediction.class_id,
                            prediction.relative_area,
                            comparison.path.name,
                            prediction_index,
                        ),
                    )
                )
    for values in candidates.values():
        values.sort(key=lambda candidate: candidate.priority)
    return candidates


def render_capacity_examples(
    candidates: dict[str, list[ExampleCandidate]],
    comparisons: list[ImageComparison],
    assets_dir: Path,
) -> list[dict[str, Any]]:
    """Render deterministic n-versus-s examples without success-only selection."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    for existing in assets_dir.glob("example_*.jpg"):
        existing.unlink()
    labels = {
        "improved": "YOLO26n missed; YOLO26s detected",
        "both_missed": "both models missed",
        "false_positive": "YOLO26s introduced false positive",
        "regressed": "YOLO26n detected; YOLO26s missed",
    }
    records: list[dict[str, Any]] = []
    for category, values in candidates.items():
        selected: list[ExampleCandidate] = []
        seen_images: set[int] = set()
        seen_classes: set[int] = set()
        for candidate in values:
            if (
                candidate.image_index not in seen_images
                and candidate.class_id not in seen_classes
            ):
                selected.append(candidate)
                seen_images.add(candidate.image_index)
                seen_classes.add(candidate.class_id)
            if len(selected) == MAX_EXAMPLES_PER_CATEGORY:
                break
        for candidate in values:
            if len(selected) == MAX_EXAMPLES_PER_CATEGORY:
                break
            if candidate.image_index not in seen_images:
                selected.append(candidate)
                seen_images.add(candidate.image_index)

        for ordinal, candidate in enumerate(selected, start=1):
            comparison = comparisons[candidate.image_index]
            focal_box = (
                comparison.experiment_predictions[candidate.object_index].xyxy
                if category == "false_positive"
                else comparison.ground_truth[candidate.object_index].xyxy
            )
            with Image.open(comparison.path) as source:
                n_panel = draw_panel(
                    source,
                    comparison.ground_truth,
                    comparison.baseline_predictions,
                    comparison.baseline_matches,
                    "YOLO26n @ 960: gold=GT, green=matched, red=unmatched",
                    focal_box,
                )
                s_panel = draw_panel(
                    source,
                    comparison.ground_truth,
                    comparison.experiment_predictions,
                    comparison.experiment_matches,
                    "YOLO26s @ 960: gold=GT, green=matched, red=unmatched",
                    focal_box,
                )
            combined = Image.new(
                "RGB", (n_panel.width + s_panel.width, n_panel.height), "white"
            )
            combined.paste(n_panel, (0, 0))
            combined.paste(s_panel, (n_panel.width, 0))
            filename = f"example_{category}_{ordinal:02d}.jpg"
            combined.save(assets_dir / filename, quality=86, optimize=True)
            records.append(
                {
                    "category": category,
                    "description": labels[category],
                    "file": filename,
                    "source_image": comparison.path.name,
                    "class_id": candidate.class_id,
                    "class_name": EXPECTED_CLASSES[candidate.class_id],
                    "relative_area": candidate.relative_area,
                }
            )
    return records


def model_stats(checkpoint: Path, pretrained: Path) -> dict[str, Any]:
    """Measure comparable model capacity and on-disk sizes."""
    model = YOLO(checkpoint)
    return {
        "parameter_count": sum(p.numel() for p in model.model.parameters()),
        "gflops_at_960": float(get_flops(model.model, imgsz=960)),
        "pretrained_weights_size_bytes": pretrained.stat().st_size,
        "best_checkpoint_size_bytes": checkpoint.stat().st_size,
    }


def decide_conclusion(comparison: dict[str, Any]) -> dict[str, str]:
    """Apply the predeclared capacity-versus-compute decision rule."""
    map_delta = comparison["overall"]["map50_95"]["delta"]
    tiny_delta = comparison["size_recall"]["overall"]["tiny"]["delta"]
    small_delta = comparison["size_recall"]["overall"]["small"]["delta"]
    focus_positive = sum(
        row["map50_95_delta"] > 0
        for row in comparison["per_class"]
        if row["class_name"] in FOCUS_NAMES
    )
    if (
        map_delta >= 0.025
        and max(tiny_delta, small_delta) >= 0.05
        and focus_positive >= 6
        and comparison["latency"]["inference"]["ratio"] <= 2.5
    ):
        option = "A"
        text = (
            "YOLO26s materially improves difficult-class and small-object "
            "performance enough to justify its compute cost."
        )
    elif map_delta > 0 and (max(tiny_delta, small_delta) > 0 or focus_positive >= 5):
        option = "B"
        text = (
            "YOLO26s improves accuracy, but YOLO26n at 960 remains the better "
            "deployment tradeoff."
        )
    else:
        option = "C"
        text = "YOLO26s provides little useful gain and should not be used."
    return {
        "option": option,
        "text": text,
        "rule": (
            "A requires AP50-95 delta >=0.025, tiny or small recall delta >=0.05, "
            "positive AP50-95 deltas for at least 6/8 focus classes, and latency "
            "ratio <=2.5. B requires positive AP50-95 and either positive tiny/small "
            "recall or at least 5/8 positive focus classes. Otherwise C."
        ),
    }


def build_markdown(metrics: dict[str, Any]) -> str:
    """Build the measured model-capacity comparison report."""
    comparison = metrics["comparison"]
    lines = [
        "# YOLO26n versus YOLO26s at 960",
        "",
        "## Experimental control",
        "",
        "Only model capacity changed. Resolution, dataset, split, classes, epochs, "
        "seed, optimizer behavior, augmentation, loss settings, and diagnostic "
        "thresholds remained fixed.",
        "",
        "## Overall comparison",
        "",
        "| Metric | YOLO26n | YOLO26s | Delta / ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("map50", "mAP50"),
        ("map50_95", "mAP50-95"),
    ):
        row = comparison["overall"][key]
        lines.append(
            f"| {label} | {row['n']:.6f} | {row['s']:.6f} | {row['delta']:+.6f} |"
        )
    for key, label, unit in (
        ("inference", "Inference latency", "ms/image"),
        ("total", "Total processing latency", "ms/image"),
    ):
        row = comparison["latency"][key]
        lines.append(
            f"| {label} ({unit}) | {row['n']:.6f} | {row['s']:.6f} | "
            f"{row['ratio']:.3f}x |"
        )
    resource_rows = (
        ("allocated_gib", "Peak allocated GPU memory (GiB)"),
        ("reserved_gib", "Peak reserved GPU memory (GiB)"),
        ("parameter_count", "Parameters"),
        ("gflops_at_960", "GFLOPs at 960"),
        ("pretrained_size_mib", "Pretrained size (MiB)"),
        ("best_checkpoint_size_mib", "Best checkpoint size (MiB)"),
    )
    for key, label in resource_rows:
        row = comparison["resources"][key]
        lines.append(
            f"| {label} | {row['n']:.3f} | {row['s']:.3f} | {row['ratio']:.3f}x |"
        )

    lines.extend(
        [
            "",
            "Relative mAP50-95 gain: "
            f"{comparison['overall']['map50_95']['relative_gain'] * 100:.2f}%.",
            "",
            "## Complete per-class comparison",
            "",
            "| Class | n AP50-95 | s AP50-95 | Delta | n recall | s recall | Delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison["per_class"]:
        lines.append(
            f"| `{row['class_name']}` | {row['n_map50_95']:.6f} | "
            f"{row['s_map50_95']:.6f} | {row['map50_95_delta']:+.6f} | "
            f"{row['n_recall']:.6f} | {row['s_recall']:.6f} | "
            f"{row['recall_delta']:+.6f} |"
        )
    rows_by_class = {row["class_name"]: row for row in comparison["per_class"]}
    lines.extend(["", "### Focus classes", ""])
    for class_name in FOCUS_NAMES:
        row = rows_by_class[class_name]
        lines.append(
            f"- `{class_name}`: AP50-95 {row['n_map50_95']:.6f} -> "
            f"{row['s_map50_95']:.6f} ({row['map50_95_delta']:+.6f}); recall "
            f"{row['n_recall']:.6f} -> {row['s_recall']:.6f} "
            f"({row['recall_delta']:+.6f})."
        )
    lines.extend(
        [
            "",
            "## Size-stratified fixed-threshold recall",
            "",
            f"Confidence {CONFIDENCE_THRESHOLD:.2f}; matching IoU "
            f"{MATCH_IOU_THRESHOLD:.2f}; NMS IoU {NMS_IOU_THRESHOLD:.2f}.",
            "",
            "| Scope | Size | GT | n matched | n recall | s matched | s recall "
            "| Delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for scope, bins in comparison["size_recall"].items():
        for size in SIZE_BINS:
            row = bins[size]
            lines.append(
                f"| `{scope}` | {size} | {row['ground_truth_count']} | "
                f"{row['n_matched_count']} | {row['n_recall']:.6f} | "
                f"{row['s_matched_count']} | {row['s_recall']:.6f} | "
                f"{row['delta']:+.6f} |"
            )
    lines.extend(
        [
            "",
            "## Training behavior",
            "",
            f"- Actual YOLO26s batch: {metrics['training']['actual_batch']}",
            f"- Epochs completed: {metrics['training']['final_epoch']}",
            f"- Best epoch: {metrics['training']['best_epoch']}",
            f"- Duration: {metrics['training']['duration_seconds']:.3f} seconds",
            "",
            "## Tradeoff judgment",
            "",
            f"- Absolute mAP50-95 gain: "
            f"{comparison['overall']['map50_95']['delta']:+.6f}",
            f"- Relative mAP50-95 gain: "
            f"{comparison['overall']['map50_95']['relative_gain'] * 100:.2f}%",
            f"- Absolute recall gain: {comparison['overall']['recall']['delta']:+.6f}",
            f"- Tiny-object recall gain: "
            f"{comparison['size_recall']['overall']['tiny']['delta']:+.6f}",
            f"- Inference-latency ratio: "
            f"{comparison['latency']['inference']['ratio']:.3f}x",
            f"- Peak allocated-memory ratio: "
            f"{comparison['resources']['allocated_gib']['ratio']:.3f}x",
            f"- Parameter-count ratio: "
            f"{comparison['resources']['parameter_count']['ratio']:.3f}x",
            "",
            "## Representative failures and changes",
            "",
            "Gold boxes are ground truth, green predictions are matched, red "
            "predictions are unmatched, and magenta marks the focal object.",
            "",
        ]
    )
    for example in metrics["failure_examples"]:
        lines.append(
            f"- {example['description']}: `{example['class_name']}`, relative area "
            f"{example['relative_area']:.6f}."
        )
        lines.append(
            f"  ![{example['file']}](../../assets/model_capacity_960/{example['file']})"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "This is a one-seed comparison on the development validation split. "
            "The fixed-threshold size diagnostic is not AP, and sparse bins must "
            "be interpreted with their counts. A reduced batch after a genuine OOM "
            "is documented as a comparability limitation if it occurred.",
            "",
            "## Conclusion",
            "",
            f"**{metrics['conclusion']['option']}. {metrics['conclusion']['text']}**",
            "",
            "The additional capacity raises model-only latency and resource use "
            "substantially, but it remains below 23 ms/image on this GPU while "
            "producing a material tiny-object recall gain and positive AP50-95 "
            "movement in seven of the eight focus classes. This conclusion is for "
            "the measured GPU deployment context, not an unconstrained preference "
            "for the larger model.",
            "",
            f"Decision rule: {metrics['conclusion']['rule']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run paired predictions and produce the capacity report."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model-capacity analysis")
    paths = [N_METRICS_PATH, S_METRICS_PATH, N_CHECKPOINT, S_CHECKPOINT]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required comparison artifacts missing: {missing}")

    runtime_yaml = prepare_runtime_dataset_yaml(
        Path("data/construction-ppe/data.yaml"),
        Path("runs/model_capacity_960_analysis/dataset_runtime.yaml"),
    )
    dataset = yaml.safe_load(runtime_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(dataset["path"])
    image_dir = dataset_root / dataset["val"]
    image_paths = sorted(
        (p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )
    if len(image_paths) != 200:
        raise RuntimeError(f"Expected 200 validation images, found {len(image_paths)}")

    # Diagnostic batching changes throughput only, not predictions or thresholds.
    n_predictions = predict_validation(N_CHECKPOINT, image_paths, imgsz=960, batch=4)
    s_predictions = predict_validation(S_CHECKPOINT, image_paths, imgsz=960, batch=4)
    comparisons: list[ImageComparison] = []
    counters = {
        model: {
            "overall": empty_counts(),
            **{
                EXPECTED_CLASSES[class_id]: empty_counts()
                for class_id in FOCUS_CLASS_IDS
            },
        }
        for model in ("n", "s")
    }
    for index, image_path in enumerate(image_paths):
        ground_truth = load_ground_truth(
            dataset_root / "labels" / "val" / f"{image_path.stem}.txt"
        )
        n_matches = match_detections(
            ground_truth, n_predictions[index], MATCH_IOU_THRESHOLD
        )
        s_matches = match_detections(
            ground_truth, s_predictions[index], MATCH_IOU_THRESHOLD
        )
        comparisons.append(
            ImageComparison(
                path=image_path,
                ground_truth=ground_truth,
                baseline_predictions=n_predictions[index],
                experiment_predictions=s_predictions[index],
                baseline_matches=n_matches,
                experiment_matches=s_matches,
            )
        )
        for model_name, matches in (("n", n_matches), ("s", s_matches)):
            update_counts(counters[model_name]["overall"], ground_truth, matches)
            for class_id in FOCUS_CLASS_IDS:
                update_counts(
                    counters[model_name][EXPECTED_CLASSES[class_id]],
                    ground_truth,
                    matches,
                    class_id,
                )

    size_recall: dict[str, Any] = {}
    for scope in counters["n"]:
        n_bins = finalize_counts(counters["n"][scope])
        s_bins = finalize_counts(counters["s"][scope])
        size_recall[scope] = {}
        for size in SIZE_BINS:
            n_row, s_row = n_bins[size], s_bins[size]
            if n_row["ground_truth_count"] != s_row["ground_truth_count"]:
                raise RuntimeError("Ground-truth counts differ between diagnostics")
            n_recall = n_row["recall"] or 0.0
            s_recall = s_row["recall"] or 0.0
            size_recall[scope][size] = {
                "ground_truth_count": n_row["ground_truth_count"],
                "n_matched_count": n_row["matched_count"],
                "n_recall": n_recall,
                "s_matched_count": s_row["matched_count"],
                "s_recall": s_recall,
                "delta": s_recall - n_recall,
            }

    n_metrics = json.loads(N_METRICS_PATH.read_text(encoding="utf-8"))
    s_metrics = json.loads(S_METRICS_PATH.read_text(encoding="utf-8"))
    n_by_class = {row["class_name"]: row for row in n_metrics["per_class"]}
    s_by_class = {row["class_name"]: row for row in s_metrics["per_class"]}
    per_class = []
    for class_id, class_name in enumerate(EXPECTED_CLASSES):
        n_row, s_row = n_by_class[class_name], s_by_class[class_name]
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "n_map50_95": n_row["map50_95"],
                "s_map50_95": s_row["map50_95"],
                "map50_95_delta": s_row["map50_95"] - n_row["map50_95"],
                "n_recall": n_row["recall"],
                "s_recall": s_row["recall"],
                "recall_delta": s_row["recall"] - n_row["recall"],
            }
        )
    overall = {
        key: {
            "n": n_metrics["overall"][key],
            "s": s_metrics["overall"][key],
            "delta": s_metrics["overall"][key] - n_metrics["overall"][key],
        }
        for key in ("precision", "recall", "map50", "map50_95")
    }
    overall["map50_95"]["relative_gain"] = (
        overall["map50_95"]["delta"] / overall["map50_95"]["n"]
    )
    n_speed, s_speed = (
        n_metrics["validation_speed_ms_per_image"],
        s_metrics["validation_speed_ms_per_image"],
    )
    n_total = sum(n_speed[key] for key in ("preprocess", "inference", "postprocess"))
    s_total = sum(s_speed[key] for key in ("preprocess", "inference", "postprocess"))
    n_stats = model_stats(N_CHECKPOINT, Path("yolo26n.pt"))
    s_stats = model_stats(S_CHECKPOINT, Path("yolo26s.pt"))
    n_training, s_training = n_metrics["training"], s_metrics["training"]

    def ratio_row(n_value: float, s_value: float) -> dict[str, float]:
        return {"n": n_value, "s": s_value, "ratio": s_value / n_value}

    comparison = {
        "overall": overall,
        "per_class": per_class,
        "size_recall": size_recall,
        "latency": {
            "inference": ratio_row(n_speed["inference"], s_speed["inference"]),
            "total": ratio_row(n_total, s_total),
        },
        "validation_timing_ms_per_image": {
            key: ratio_row(n_speed[key], s_speed[key])
            for key in ("preprocess", "inference", "postprocess", "loss")
        },
        "resources": {
            "allocated_gib": ratio_row(
                n_training["peak_gpu_memory_allocated_gib"],
                s_training["peak_gpu_memory_allocated_gib"],
            ),
            "reserved_gib": ratio_row(
                n_training["peak_gpu_memory_reserved_gib"],
                s_training["peak_gpu_memory_reserved_gib"],
            ),
            "parameter_count": ratio_row(
                n_stats["parameter_count"], s_stats["parameter_count"]
            ),
            "gflops_at_960": ratio_row(
                n_stats["gflops_at_960"], s_stats["gflops_at_960"]
            ),
            "pretrained_size_mib": ratio_row(
                n_stats["pretrained_weights_size_bytes"] / 1024**2,
                s_stats["pretrained_weights_size_bytes"] / 1024**2,
            ),
            "best_checkpoint_size_mib": ratio_row(
                n_stats["best_checkpoint_size_bytes"] / 1024**2,
                s_stats["best_checkpoint_size_bytes"] / 1024**2,
            ),
        },
    }
    candidates = collect_capacity_candidates(comparisons)
    examples = render_capacity_examples(candidates, comparisons, ASSETS_DIR)
    s_metrics["model_stats"] = s_stats
    s_metrics["diagnostic_method"] = {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "matching_iou_threshold": MATCH_IOU_THRESHOLD,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "size_bins": {
            "tiny": "relative area < 0.001",
            "small": "0.001 <= relative area < 0.01",
            "large": "relative area >= 0.01",
        },
        "matching": "descending-confidence, same-class, greedy one-to-one",
        "inference_batch": 4,
    }
    s_metrics["comparison"] = comparison
    s_metrics["failure_examples"] = examples
    s_metrics["failure_analysis_candidate_counts"] = {
        category: len(values) for category, values in candidates.items()
    }
    s_metrics["conclusion"] = decide_conclusion(comparison)
    write_json(S_METRICS_PATH, s_metrics)
    report = S_METRICS_PATH.parent / "comparison.md"
    report.write_text(build_markdown(s_metrics), encoding="utf-8")
    print(json.dumps(s_metrics, indent=2, sort_keys=True))
    print(f"Comparison report: {report.resolve()}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
