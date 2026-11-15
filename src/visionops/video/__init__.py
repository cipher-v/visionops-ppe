"""Production-style streaming video inference primitives."""

from visionops.video.config import FrozenDetectorConfig, load_frozen_detector_config
from visionops.video.schema import VideoDetection

__all__ = [
    "FrozenDetectorConfig",
    "VideoDetection",
    "load_frozen_detector_config",
]
