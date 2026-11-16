"""Deterministic, on-the-fly visual stress perturbations."""

from __future__ import annotations

import cv2
import numpy as np

SEVERITIES = ("mild", "moderate", "severe")
PERTURBATION_PARAMETERS: dict[str, dict[str, float | int]] = {
    "brightness_decrease": {"mild": 0.8, "moderate": 0.6, "severe": 0.4},
    "brightness_increase": {"mild": 1.15, "moderate": 1.30, "severe": 1.50},
    "contrast_reduction": {"mild": 0.8, "moderate": 0.6, "severe": 0.4},
    "gaussian_blur": {"mild": 3, "moderate": 5, "severe": 9},
    "motion_blur": {"mild": 3, "moderate": 5, "severe": 9},
    "gaussian_noise": {"mild": 5.0, "moderate": 15.0, "severe": 30.0},
    "jpeg_compression": {"mild": 70, "moderate": 40, "severe": 20},
    "saturation_reduction": {"mild": 0.8, "moderate": 0.5, "severe": 0.2},
}


def apply_perturbation(
    image: np.ndarray,
    name: str,
    severity: str,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Return one deterministic uint8 BGR perturbation without mutating input."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError("Perturbation input must be a uint8 color image")
    if name not in PERTURBATION_PARAMETERS:
        raise ValueError(f"Unknown perturbation: {name}")
    if severity not in SEVERITIES:
        raise ValueError(f"Unknown severity: {severity}")
    value = PERTURBATION_PARAMETERS[name][severity]
    source = image.copy()
    if name in {"brightness_decrease", "brightness_increase"}:
        return np.clip(source.astype(np.float32) * float(value), 0, 255).astype(
            np.uint8
        )
    if name == "contrast_reduction":
        return np.clip(
            (source.astype(np.float32) - 127.5) * float(value) + 127.5,
            0,
            255,
        ).astype(np.uint8)
    if name == "gaussian_blur":
        kernel = int(value)
        return cv2.GaussianBlur(source, (kernel, kernel), 0)
    if name == "motion_blur":
        kernel_size = int(value)
        kernel = np.eye(kernel_size, dtype=np.float32) / kernel_size
        return cv2.filter2D(source, -1, kernel)
    if name == "gaussian_noise":
        noise = np.random.default_rng(seed).normal(0, float(value), source.shape)
        return np.clip(source.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if name == "jpeg_compression":
        ok, encoded = cv2.imencode(
            ".jpg", source, [cv2.IMWRITE_JPEG_QUALITY, int(value)]
        )
        if not ok:
            raise RuntimeError("OpenCV could not encode JPEG perturbation")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("OpenCV could not decode JPEG perturbation")
        return decoded
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(value), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
