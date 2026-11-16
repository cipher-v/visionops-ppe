"""Backend-neutral Ultralytics detector adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from visionops.deployment.config import BackendArtifact, DeploymentConfig
from visionops.video.config import FrozenDetectorConfig


class DetectorBackend:
    """Load one verified artifact while preserving the shared prediction contract."""

    def __init__(
        self,
        name: str,
        frozen: FrozenDetectorConfig,
        deployment_config: Path | None = None,
        repository_root: Path | None = None,
    ) -> None:
        self.name = name
        self.artifact: BackendArtifact | None = None
        if name == "pytorch" and deployment_config is None:
            artifact_path = frozen.weights_path
            artifact_hash = frozen.weights_sha256
        else:
            if deployment_config is None:
                raise ValueError(
                    "Non-PyTorch backends require deployment configuration"
                )
            deployment = DeploymentConfig.from_yaml(
                deployment_config, repository_root=repository_root
            )
            self.artifact = deployment.select(name, frozen.weights_sha256)
            if self.artifact.artifact_path is None:
                raise ValueError(f"Backend '{name}' has no artifact path")
            artifact_path = self.artifact.artifact_path
            artifact_hash = self.artifact.artifact_sha256
        self.artifact_path = artifact_path
        self.artifact_sha256 = artifact_hash
        self.model = YOLO(str(artifact_path))

    def predict(
        self,
        image: np.ndarray,
        *,
        imgsz: int,
        confidence: float,
        iou: float,
        device: str | int,
    ) -> Any:
        return self.model.predict(
            source=image,
            imgsz=imgsz,
            conf=confidence,
            iou=iou,
            device=device,
            verbose=False,
        )[0]

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "artifact": str(self.artifact_path),
            "artifact_sha256": self.artifact_sha256,
            "precision": self.artifact.precision if self.artifact else "fp32",
            "runtime": self.artifact.runtime if self.artifact else "torch",
        }
