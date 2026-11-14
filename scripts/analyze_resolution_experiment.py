"""Compare 640 and 960 detections with fixed size-stratified diagnostics."""

from __future__ import annotations

import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

from visionops.baseline import EXPECTED_CLASSES, prepare_runtime_dataset_yaml
from visionops.detection_analysis import (
    SIZE_BINS,
    Detection,
    MatchResult,
    area_bin,
    box_iou,
    match_detections,
)

CONFIDENCE_THRESHOLD = 0.25
MATCH_IOU_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.70
FOCUS_CLASS_IDS = (0, 6, 2, 11, 4, 7)
MAX_EXAMPLES_PER_CATEGORY = 2
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageComparison:
    """Predictions and one-to-one matches for one validation image."""

    path: Path
    ground_truth: list[Detection]
    baseline_predictions: list[Detection]
    experiment_predictions: list[Detection]
    baseline_matches: MatchResult
    experiment_matches: MatchResult


@dataclass(frozen=True)
class ExampleCandidate:
    """A deterministic failure-analysis candidate."""

    category: str
    image_index: int
    object_index: int
    class_id: int
    relative_area: float
    priority: tuple[int, int, str, int]


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write stable, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_ground_truth(label_path: Path) -> list[Detection]:
    """Load audited YOLO labels as normalized xyxy boxes."""
    boxes: list[Detection] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Malformed label {label_path}:{line_number}")
        class_id = int(fields[0])
        center_x, center_y, width, height = map(float, fields[1:])
        boxes.append(
            Detection(
                class_id=class_id,
                xyxy=(
                    center_x - width / 2,
                    center_y - height / 2,
                    center_x + width / 2,
                    center_y + height / 2,
                ),
            )
        )
    return boxes


def predictions_from_result(result: Any) -> list[Detection]:
    """Convert an Ultralytics prediction result to normalized boxes."""
    if result.boxes is None:
        return []
    coordinates = result.boxes.xyxyn.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    confidences = result.boxes.conf.detach().cpu().tolist()
    return [
        Detection(int(class_id), tuple(map(float, xyxy)), float(confidence))
        for xyxy, class_id, confidence in zip(
            coordinates, classes, confidences, strict=True
        )
    ]


def predict_validation(
    checkpoint: Path, image_paths: list[Path], imgsz: int, batch: int = 8
) -> list[list[Detection]]:
    """Run fixed-threshold predictions in deterministic, memory-bounded chunks."""
    model = YOLO(checkpoint)
    predictions: list[list[Detection]] = []
    for start in range(0, len(image_paths), batch):
        chunk = image_paths[start : start + batch]
        results = model.predict(
            source=[str(path) for path in chunk],
            imgsz=imgsz,
            conf=CONFIDENCE_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            batch=batch,
            device=0,
            max_det=300,
            agnostic_nms=False,
            verbose=False,
            stream=False,
        )
        predictions.extend(predictions_from_result(result) for result in results)
        del results
        torch.cuda.empty_cache()
    return predictions


def empty_counts() -> dict[str, dict[str, int]]:
    """Create mutable bin counters."""
    return {name: {"ground_truth_count": 0, "matched_count": 0} for name in SIZE_BINS}


def update_counts(
    counters: dict[str, dict[str, int]],
    ground_truth: list[Detection],
    matches: MatchResult,
    class_id: int | None = None,
) -> None:
    """Accumulate ground-truth and matched counts by relative-area bin."""
    for index, target in enumerate(ground_truth):
        if class_id is not None and target.class_id != class_id:
            continue
        bin_name = area_bin(target.relative_area)
        counters[bin_name]["ground_truth_count"] += 1
        if index in matches.matched_ground_truth_indices:
            counters[bin_name]["matched_count"] += 1


def finalize_counts(counters: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Add explicit recall values to size-bin counts."""
    finalized: dict[str, Any] = {}
    for bin_name, values in counters.items():
        total = values["ground_truth_count"]
        matched = values["matched_count"]
        finalized[bin_name] = {
            **values,
            "recall": None if total == 0 else matched / total,
        }
    return finalized


def candidate_priority(
    class_id: int, relative_area: float, filename: str, object_index: int
) -> tuple[int, int, str, int]:
    """Prioritize requested classes and small objects deterministically."""
    return (
        0 if class_id in FOCUS_CLASS_IDS else 1,
        0 if relative_area < 0.01 else 1,
        filename.casefold(),
        object_index,
    )


def collect_candidates(
    comparisons: list[ImageComparison],
) -> dict[str, list[ExampleCandidate]]:
    """Collect improvement, shared-miss, and introduced-false-positive cases."""
    candidates: dict[str, list[ExampleCandidate]] = {
        "improved": [],
        "both_missed": [],
        "false_positive": [],
    }
    for image_index, comparison in enumerate(comparisons):
        baseline_matched = comparison.baseline_matches.matched_ground_truth_indices
        experiment_matched = comparison.experiment_matches.matched_ground_truth_indices
        for ground_truth_index, target in enumerate(comparison.ground_truth):
            category = None
            if ground_truth_index not in baseline_matched:
                if ground_truth_index in experiment_matched:
                    category = "improved"
                else:
                    category = "both_missed"
            if category is not None:
                candidates[category].append(
                    ExampleCandidate(
                        category=category,
                        image_index=image_index,
                        object_index=ground_truth_index,
                        class_id=target.class_id,
                        relative_area=target.relative_area,
                        priority=candidate_priority(
                            target.class_id,
                            target.relative_area,
                            comparison.path.name,
                            ground_truth_index,
                        ),
                    )
                )

        matched_predictions = comparison.experiment_matches.matched_prediction_indices
        for prediction_index, prediction in enumerate(
            comparison.experiment_predictions
        ):
            if prediction_index in matched_predictions:
                continue
            existed_at_640 = any(
                baseline_prediction.class_id == prediction.class_id
                and box_iou(baseline_prediction.xyxy, prediction.xyxy)
                >= MATCH_IOU_THRESHOLD
                for baseline_prediction in comparison.baseline_predictions
            )
            if existed_at_640:
                continue
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

    for category in candidates:
        candidates[category].sort(key=lambda candidate: candidate.priority)
    return candidates


def draw_panel(
    image: Image.Image,
    ground_truth: list[Detection],
    predictions: list[Detection],
    matches: MatchResult,
    title: str,
    focal_box: tuple[float, float, float, float],
) -> Image.Image:
    """Render one model panel with ground truth, matches, and false positives."""
    panel = ImageOps.contain(image.convert("RGB"), (800, 600))
    canvas = Image.new("RGB", (panel.width, panel.height + 36), "white")
    canvas.paste(panel, (0, 36))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 10), title, fill="black", font=font)

    def pixels(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return (
            round(x1 * panel.width),
            round(y1 * panel.height) + 36,
            round(x2 * panel.width),
            round(y2 * panel.height) + 36,
        )

    for target in ground_truth:
        draw.rectangle(pixels(target.xyxy), outline="gold", width=2)
    for index, prediction in enumerate(predictions):
        color = "limegreen" if index in matches.matched_prediction_indices else "red"
        rectangle = pixels(prediction.xyxy)
        draw.rectangle(rectangle, outline=color, width=2)
        label = f"{EXPECTED_CLASSES[prediction.class_id]} {prediction.confidence:.2f}"
        draw.text((rectangle[0] + 2, rectangle[1] + 2), label, fill=color, font=font)
    draw.rectangle(pixels(focal_box), outline="magenta", width=4)
    return canvas


def render_examples(
    candidates: dict[str, list[ExampleCandidate]],
    comparisons: list[ImageComparison],
    assets_dir: Path,
) -> list[dict[str, Any]]:
    """Save a small deterministic and balanced set of side-by-side examples."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    for existing in assets_dir.glob("example_*.jpg"):
        existing.unlink()

    records: list[dict[str, Any]] = []
    category_labels = {
        "improved": "640 missed; 960 detected",
        "both_missed": "both models missed",
        "false_positive": "960 introduced false positive",
    }
    for category in ("improved", "both_missed", "false_positive"):
        selected: list[ExampleCandidate] = []
        seen_images: set[int] = set()
        seen_classes: set[int] = set()
        for candidate in candidates[category]:
            if (
                candidate.image_index not in seen_images
                and candidate.class_id not in seen_classes
            ):
                selected.append(candidate)
                seen_images.add(candidate.image_index)
                seen_classes.add(candidate.class_id)
            if len(selected) == MAX_EXAMPLES_PER_CATEGORY:
                break
        for candidate in candidates[category]:
            if len(selected) == MAX_EXAMPLES_PER_CATEGORY:
                break
            if candidate.image_index not in seen_images:
                selected.append(candidate)
                seen_images.add(candidate.image_index)
        for ordinal, candidate in enumerate(selected, start=1):
            comparison = comparisons[candidate.image_index]
            if category == "false_positive":
                focal_box = comparison.experiment_predictions[
                    candidate.object_index
                ].xyxy
            else:
                focal_box = comparison.ground_truth[candidate.object_index].xyxy
            with Image.open(comparison.path) as source_image:
                baseline_panel = draw_panel(
                    source_image,
                    comparison.ground_truth,
                    comparison.baseline_predictions,
                    comparison.baseline_matches,
                    "640: gold=GT, green=matched, red=unmatched, magenta=focal",
                    focal_box,
                )
                experiment_panel = draw_panel(
                    source_image,
                    comparison.ground_truth,
                    comparison.experiment_predictions,
                    comparison.experiment_matches,
                    "960: gold=GT, green=matched, red=unmatched, magenta=focal",
                    focal_box,
                )
            combined = Image.new(
                "RGB",
                (
                    baseline_panel.width + experiment_panel.width,
                    max(baseline_panel.height, experiment_panel.height),
                ),
                "white",
            )
            combined.paste(baseline_panel, (0, 0))
            combined.paste(experiment_panel, (baseline_panel.width, 0))
            output_name = f"example_{category}_{ordinal:02d}.jpg"
            combined.save(assets_dir / output_name, quality=86, optimize=True)
            records.append(
                {
                    "category": category,
                    "description": category_labels[category],
                    "file": output_name,
                    "source_image": comparison.path.name,
                    "class_id": candidate.class_id,
                    "class_name": EXPECTED_CLASSES[candidate.class_id],
                    "relative_area": candidate.relative_area,
                }
            )
    return records


def comparison_rows(
    baseline_metrics: dict[str, Any], experiment_metrics: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build exact per-class baseline-versus-experiment rows."""
    baseline_by_class = {
        row["class_name"]: row for row in baseline_metrics["per_class"]
    }
    experiment_by_class = {
        row["class_name"]: row for row in experiment_metrics["per_class"]
    }
    return [
        {
            "class_id": class_id,
            "class_name": class_name,
            "baseline_map50_95": baseline_by_class[class_name]["map50_95"],
            "experiment_map50_95": experiment_by_class[class_name]["map50_95"],
            "map50_95_delta": experiment_by_class[class_name]["map50_95"]
            - baseline_by_class[class_name]["map50_95"],
            "baseline_recall": baseline_by_class[class_name]["recall"],
            "experiment_recall": experiment_by_class[class_name]["recall"],
            "recall_delta": experiment_by_class[class_name]["recall"]
            - baseline_by_class[class_name]["recall"],
        }
        for class_id, class_name in enumerate(EXPECTED_CLASSES)
    ]


def decide_conclusion(comparison: dict[str, Any]) -> dict[str, str]:
    """Apply a documented conservative rule to select conclusion A, B, or C."""
    map_delta = comparison["overall"]["map50_95"]["delta"]
    tiny_delta = comparison["size_recall"]["overall"]["tiny"]["delta"]
    small_delta = comparison["size_recall"]["overall"]["small"]["delta"]
    latency_ratio = comparison["latency"]["ratio"]
    if (
        map_delta >= 0.02
        and max(tiny_delta, small_delta) >= 0.05
        and latency_ratio <= 2.0
    ):
        option = "A"
        text = (
            "Higher resolution materially improves the relevant performance enough "
            "to justify its latency cost."
        )
    elif map_delta > 0 and max(tiny_delta, small_delta) > 0:
        option = "B"
        text = (
            "Higher resolution improves accuracy but the latency/compute tradeoff "
            "is probably not worthwhile."
        )
    else:
        option = "C"
        text = (
            "Higher resolution provides little meaningful improvement, so the "
            "primary bottleneck is likely elsewhere."
        )
    return {
        "option": option,
        "text": text,
        "rule": (
            "A requires mAP50-95 delta >=0.02, tiny or small recall delta >=0.05, "
            "and inference-latency ratio <=2.0. B requires positive mAP50-95 and "
            "positive tiny or small recall but does not meet A. Otherwise C."
        ),
    }


def build_markdown(metrics: dict[str, Any]) -> str:
    """Build the controlled resolution comparison report."""
    comparison = metrics["comparison"]
    overall = comparison["overall"]
    lines = [
        "# YOLO26n 960 controlled resolution experiment",
        "",
        "## Objective and control",
        "",
        "Test whether changing only input resolution from 640 to 960 improves "
        "small-PPE and missing-PPE detection enough to justify its latency cost. "
        "Model, data, classes, epochs, seed, optimizer, schedule, augmentation, "
        "loss settings, and all other controlled training settings are unchanged.",
        "",
        "## 960 training behavior",
        "",
        f"- Actual batch size: {metrics['training']['actual_batch']}",
        f"- Epochs completed: {metrics['training']['final_epoch']}",
        f"- Best epoch: {metrics['training']['best_epoch']}",
        f"- Training duration: {metrics['training']['duration_seconds']:.3f} seconds",
        f"- Early stopping: {metrics['training']['early_stopping_occurred']}",
        "",
        "## Overall validation comparison",
        "",
        "| Metric | 640 baseline | 960 experiment | Absolute delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("map50", "mAP50"),
        ("map50_95", "mAP50-95"),
    ):
        row = overall[key]
        lines.append(
            f"| {label} | {row['baseline']:.6f} | {row['experiment']:.6f} | "
            f"{row['delta']:+.6f} |"
        )
    latency = comparison["latency"]
    lines.append(
        f"| Inference latency (ms/image) | {latency['baseline_ms']:.6f} | "
        f"{latency['experiment_ms']:.6f} | {latency['delta_ms']:+.6f} |"
    )
    lines.extend(
        [
            "",
            "## Per-class comparison",
            "",
            "| Class | 640 AP50-95 | 960 AP50-95 | Delta | 640 recall | "
            "960 recall | Delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comparison["per_class"]:
        lines.append(
            f"| `{row['class_name']}` | {row['baseline_map50_95']:.6f} | "
            f"{row['experiment_map50_95']:.6f} | "
            f"{row['map50_95_delta']:+.6f} | {row['baseline_recall']:.6f} | "
            f"{row['experiment_recall']:.6f} | {row['recall_delta']:+.6f} |"
        )

    focused_classes = (
        "helmet",
        "vest",
        "goggles",
        "no_helmet",
        "no_vest",
        "no_goggle",
        "no_gloves",
    )
    rows_by_class = {row["class_name"]: row for row in comparison["per_class"]}
    lines.extend(["", "### Requested focus classes", ""])
    for class_name in focused_classes:
        row = rows_by_class[class_name]
        lines.append(
            f"- `{class_name}`: AP50-95 {row['baseline_map50_95']:.6f} -> "
            f"{row['experiment_map50_95']:.6f} "
            f"({row['map50_95_delta']:+.6f}); recall "
            f"{row['baseline_recall']:.6f} -> "
            f"{row['experiment_recall']:.6f} ({row['recall_delta']:+.6f})."
        )

    lines.extend(
        [
            "",
            "## Size-stratified fixed-threshold recall",
            "",
            f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}. Matching IoU: "
            f"{MATCH_IOU_THRESHOLD:.2f}. NMS IoU: {NMS_IOU_THRESHOLD:.2f}.",
            "",
            "| Scope | Size | GT count | 640 matched | 640 recall | 960 matched | "
            "960 recall | Delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for scope, bins in comparison["size_recall"].items():
        for bin_name in SIZE_BINS:
            row = bins[bin_name]
            lines.append(
                f"| `{scope}` | {bin_name} | {row['ground_truth_count']} | "
                f"{row['baseline_matched_count']} | {row['baseline_recall']:.6f} | "
                f"{row['experiment_matched_count']} | "
                f"{row['experiment_recall']:.6f} | {row['delta']:+.6f} |"
            )

    gpu = comparison["gpu_memory"]
    timing = comparison["validation_timing_ms_per_image"]
    lines.extend(
        [
            "",
            "## Compute tradeoff",
            "",
            "| Validation timing | 640 (ms/image) | 960 (ms/image) | Delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key in ("preprocess", "inference", "postprocess", "loss"):
        row = timing[key]
        lines.append(
            f"| {key} | {row['baseline']:.6f} | {row['experiment']:.6f} | "
            f"{row['delta']:+.6f} |"
        )
    lines.extend(
        [
            "",
            f"- Inference latency changed by {latency['delta_ms']:+.6f} ms/image "
            f"({latency['ratio']:.3f}x the baseline).",
            f"- Peak allocated GPU memory changed from "
            f"{gpu['baseline_allocated_gib']:.3f} GiB to "
            f"{gpu['experiment_allocated_gib']:.3f} GiB "
            f"({gpu['allocated_delta_gib']:+.3f} GiB).",
            f"- Peak reserved GPU memory changed from "
            f"{gpu['baseline_reserved_gib']:.3f} GiB to "
            f"{gpu['experiment_reserved_gib']:.3f} GiB "
            f"({gpu['reserved_delta_gib']:+.3f} GiB).",
            "",
            "## Representative comparison examples",
            "",
            "The selection is deterministic and includes improvements, shared "
            "misses, and introduced false positives when available. Gold boxes are "
            "ground truth; green predictions matched; red predictions did not; the "
            "magenta box marks the focal object.",
            "",
            f"Candidate counts before deterministic sampling: improvements "
            f"{metrics['failure_analysis_candidate_counts']['improved']}, shared "
            f"misses {metrics['failure_analysis_candidate_counts']['both_missed']}, "
            f"introduced false positives "
            f"{metrics['failure_analysis_candidate_counts']['false_positive']}.",
            "",
        ]
    )
    for example in metrics["failure_examples"]:
        lines.append(
            f"- {example['description']}: `{example['class_name']}`, relative "
            f"area {example['relative_area']:.6f}, source "
            f"`{example['source_image']}`."
        )
        lines.append(
            f"  ![{example['file']}](../../assets/resolution_960/{example['file']})"
        )

    lines.extend(
        [
            "",
            "## Diagnostic method and limitations",
            "",
            "Predictions are processed in descending confidence order. Each can "
            "match only the highest-IoU unmatched ground-truth box of the same "
            "class, and every ground-truth box can be matched at most once. Tiny "
            "means area <0.001; small means 0.001<=area<0.01; large means area "
            ">=0.01.",
            "",
            "This diagnostic is not AP, is sensitive to the documented confidence "
            "and IoU thresholds, uses only one seed and the development validation "
            "split, and does not estimate deployment performance. Sparse bins must "
            "be interpreted from their reported counts rather than recall alone.",
            "",
            "## Conclusion",
            "",
            f"**{metrics['conclusion']['option']}. {metrics['conclusion']['text']}**",
            "",
            f"Decision rule: {metrics['conclusion']['rule']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run fixed-threshold predictions, diagnostics, examples, and report."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for resolution comparison predictions")

    baseline_metrics_path = Path(
        "docs/experiments/baseline/baseline_metrics.json"
    ).resolve()
    experiment_metrics_path = Path(
        "docs/experiments/resolution_960/metrics.json"
    ).resolve()
    baseline_checkpoint = Path("runs/baseline_yolo26n_640/weights/best.pt").resolve()
    experiment_checkpoint = Path(
        "runs/experiment_yolo26n_960/weights/best.pt"
    ).resolve()
    required = (
        baseline_metrics_path,
        experiment_metrics_path,
        baseline_checkpoint,
        experiment_checkpoint,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required comparison artifacts are missing: {missing}")

    runtime_yaml = prepare_runtime_dataset_yaml(
        Path("data/construction-ppe/data.yaml"),
        Path("runs/resolution_960_analysis/dataset_runtime.yaml"),
    )
    dataset = yaml.safe_load(runtime_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(dataset["path"])
    validation_directory = dataset_root / dataset["val"]
    image_paths = sorted(
        (
            path
            for path in validation_directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )
    if len(image_paths) != 200:
        raise RuntimeError(f"Expected 200 validation images, found {len(image_paths)}")

    baseline_predictions = predict_validation(
        baseline_checkpoint, image_paths, imgsz=640
    )
    experiment_predictions = predict_validation(
        experiment_checkpoint, image_paths, imgsz=960
    )
    comparisons: list[ImageComparison] = []
    counters: dict[str, dict[str, dict[str, dict[str, int]]]] = {
        model: {
            "overall": empty_counts(),
            **{
                EXPECTED_CLASSES[class_id]: empty_counts()
                for class_id in FOCUS_CLASS_IDS
            },
        }
        for model in ("baseline", "experiment")
    }
    for index, image_path in enumerate(image_paths):
        label_path = dataset_root / "labels" / "val" / f"{image_path.stem}.txt"
        ground_truth = load_ground_truth(label_path)
        baseline_matches = match_detections(
            ground_truth, baseline_predictions[index], MATCH_IOU_THRESHOLD
        )
        experiment_matches = match_detections(
            ground_truth, experiment_predictions[index], MATCH_IOU_THRESHOLD
        )
        comparisons.append(
            ImageComparison(
                path=image_path,
                ground_truth=ground_truth,
                baseline_predictions=baseline_predictions[index],
                experiment_predictions=experiment_predictions[index],
                baseline_matches=baseline_matches,
                experiment_matches=experiment_matches,
            )
        )
        for model_name, matches in (
            ("baseline", baseline_matches),
            ("experiment", experiment_matches),
        ):
            update_counts(counters[model_name]["overall"], ground_truth, matches)
            for class_id in FOCUS_CLASS_IDS:
                update_counts(
                    counters[model_name][EXPECTED_CLASSES[class_id]],
                    ground_truth,
                    matches,
                    class_id,
                )

    size_recall: dict[str, Any] = {}
    for scope in counters["baseline"]:
        baseline_bins = finalize_counts(counters["baseline"][scope])
        experiment_bins = finalize_counts(counters["experiment"][scope])
        size_recall[scope] = {}
        for bin_name in SIZE_BINS:
            baseline_row = baseline_bins[bin_name]
            experiment_row = experiment_bins[bin_name]
            if (
                baseline_row["ground_truth_count"]
                != experiment_row["ground_truth_count"]
            ):
                raise RuntimeError(
                    "Ground-truth counts differ between model diagnostics"
                )
            baseline_recall = baseline_row["recall"] or 0.0
            experiment_recall = experiment_row["recall"] or 0.0
            size_recall[scope][bin_name] = {
                "ground_truth_count": baseline_row["ground_truth_count"],
                "baseline_matched_count": baseline_row["matched_count"],
                "baseline_recall": baseline_recall,
                "experiment_matched_count": experiment_row["matched_count"],
                "experiment_recall": experiment_recall,
                "delta": experiment_recall - baseline_recall,
            }

    assets_dir = Path("docs/assets/resolution_960").resolve()
    candidates = collect_candidates(comparisons)
    failure_examples = render_examples(candidates, comparisons, assets_dir)
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    experiment_metrics = json.loads(experiment_metrics_path.read_text(encoding="utf-8"))
    overall = {
        key: {
            "baseline": baseline_metrics["overall"][key],
            "experiment": experiment_metrics["overall"][key],
            "delta": experiment_metrics["overall"][key]
            - baseline_metrics["overall"][key],
        }
        for key in ("precision", "recall", "map50", "map50_95")
    }
    baseline_latency = baseline_metrics["validation_speed_ms_per_image"]["inference"]
    experiment_latency = experiment_metrics["validation_speed_ms_per_image"][
        "inference"
    ]
    validation_timing = {
        key: {
            "baseline": baseline_metrics["validation_speed_ms_per_image"][key],
            "experiment": experiment_metrics["validation_speed_ms_per_image"][key],
            "delta": experiment_metrics["validation_speed_ms_per_image"][key]
            - baseline_metrics["validation_speed_ms_per_image"][key],
        }
        for key in ("preprocess", "inference", "postprocess", "loss")
    }
    baseline_training = baseline_metrics["training"]
    experiment_training = experiment_metrics["training"]
    comparison: dict[str, Any] = {
        "overall": overall,
        "per_class": comparison_rows(baseline_metrics, experiment_metrics),
        "size_recall": size_recall,
        "latency": {
            "baseline_ms": baseline_latency,
            "experiment_ms": experiment_latency,
            "delta_ms": experiment_latency - baseline_latency,
            "ratio": experiment_latency / baseline_latency,
        },
        "validation_timing_ms_per_image": validation_timing,
        "gpu_memory": {
            "baseline_allocated_gib": baseline_training[
                "peak_gpu_memory_allocated_gib"
            ],
            "experiment_allocated_gib": experiment_training[
                "peak_gpu_memory_allocated_gib"
            ],
            "allocated_delta_gib": experiment_training["peak_gpu_memory_allocated_gib"]
            - baseline_training["peak_gpu_memory_allocated_gib"],
            "baseline_reserved_gib": baseline_training["peak_gpu_memory_reserved_gib"],
            "experiment_reserved_gib": experiment_training[
                "peak_gpu_memory_reserved_gib"
            ],
            "reserved_delta_gib": experiment_training["peak_gpu_memory_reserved_gib"]
            - baseline_training["peak_gpu_memory_reserved_gib"],
        },
    }
    experiment_metrics["diagnostic_method"] = {
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "matching_iou_threshold": MATCH_IOU_THRESHOLD,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "matching_algorithm": (
            "Predictions sorted by descending confidence; greedy same-class match "
            "to highest-IoU unmatched ground truth; one-to-one at IoU >= 0.50."
        ),
        "size_bins": {
            "tiny": "relative area < 0.001",
            "small": "0.001 <= relative area < 0.01",
            "large": "relative area >= 0.01",
        },
    }
    experiment_metrics["comparison"] = comparison
    experiment_metrics["failure_examples"] = failure_examples
    experiment_metrics["failure_analysis_candidate_counts"] = {
        category: len(values) for category, values in candidates.items()
    }
    experiment_metrics["conclusion"] = decide_conclusion(comparison)
    write_json(experiment_metrics_path, experiment_metrics)
    report_path = experiment_metrics_path.parent / "comparison.md"
    report_path.write_text(build_markdown(experiment_metrics), encoding="utf-8")
    print(json.dumps(experiment_metrics, indent=2, sort_keys=True))
    print(f"Comparison report: {report_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
