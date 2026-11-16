"""Cheap image and detector-output monitoring features."""

from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np

CLASS_NAMES = ("helmet", "vest", "Person", "no_helmet", "no_vest")


def _normalized_histogram(
    values: np.ndarray, bins: int, value_range: tuple[int, int]
) -> list[float]:
    histogram, _ = np.histogram(values, bins=bins, range=value_range)
    total = histogram.sum()
    return (histogram / total if total else histogram.astype(float)).tolist()


def image_features(image: np.ndarray) -> dict[str, object]:
    """Compute resolution, brightness, color, sharpness, and histogram features."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError("Drift input must be a uint8 color image")
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    scale = min(1.0, 256 / max(width, height))
    monitoring_image = (
        cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
        if scale < 1
        else image
    )
    gray = cv2.cvtColor(monitoring_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(monitoring_image, cv2.COLOR_BGR2HSV)
    return {
        "mean_brightness": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "contrast": float(gray.std()),
        "saturation_mean": float(hsv[:, :, 1].mean()),
        "sharpness_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "monitoring_width": monitoring_image.shape[1],
        "monitoring_height": monitoring_image.shape[0],
        "grayscale_histogram": _normalized_histogram(gray, 32, (0, 256)),
        "hue_histogram": _normalized_histogram(hsv[:, :, 0], 18, (0, 180)),
        "saturation_histogram": _normalized_histogram(hsv[:, :, 1], 16, (0, 256)),
        "value_histogram": _normalized_histogram(hsv[:, :, 2], 16, (0, 256)),
    }


def output_features(frames: list[list[dict[str, object]]]) -> dict[str, object]:
    """Aggregate production detections into interpretable monitoring statistics."""
    if not frames:
        raise ValueError("At least one frame is required for output features")
    counts = {name: 0 for name in CLASS_NAMES}
    confidences: dict[str, list[float]] = defaultdict(list)
    zero_person = 0
    negative_frames = 0
    for detections in frames:
        names = []
        for detection in detections:
            name = str(detection.get("class_name", ""))
            confidence = float(detection.get("confidence", math.nan))
            if name not in counts or not math.isfinite(confidence):
                raise ValueError("Malformed detector output record")
            counts[name] += 1
            confidences[name].append(confidence)
            names.append(name)
        zero_person += "Person" not in names
        negative_frames += any(name in {"no_helmet", "no_vest"} for name in names)
    frame_count = len(frames)
    per_class = {}
    for name in CLASS_NAMES:
        ordered = sorted(confidences[name])
        per_class[name] = {
            "frequency_per_frame": counts[name] / frame_count,
            "mean_confidence": float(np.mean(ordered)) if ordered else 0.0,
            "confidence_p10": (
                ordered[round((len(ordered) - 1) * 0.10)] if ordered else 0.0
            ),
            "confidence_p50": (
                ordered[round((len(ordered) - 1) * 0.50)] if ordered else 0.0
            ),
            "confidence_p90": (
                ordered[round((len(ordered) - 1) * 0.90)] if ordered else 0.0
            ),
        }
    ppe_count = sum(counts[name] for name in CLASS_NAMES if name != "Person")
    return {
        "frame_count": frame_count,
        "detections_per_frame": sum(counts.values()) / frame_count,
        "per_class": per_class,
        "person_to_ppe_ratio": counts["Person"] / ppe_count if ppe_count else 0.0,
        "zero_person_frame_fraction": zero_person / frame_count,
        "negative_ppe_frame_fraction": negative_frames / frame_count,
    }
