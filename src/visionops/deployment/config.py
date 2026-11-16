"""Strict deployment-backend artifact configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from visionops.video.config import sha256_file

SUPPORTED_BACKENDS = frozenset({"pytorch", "onnx", "tensorrt"})


@dataclass(frozen=True)
class BackendArtifact:
    name: str
    available: bool
    artifact_path: Path | None
    artifact_sha256: str | None
    precision: str
    runtime: str | None
    metadata: dict[str, object]

    def verify(self) -> str:
        if not self.available:
            blocker = self.metadata.get("blocker", "runtime unavailable")
            raise RuntimeError(f"Backend '{self.name}' is unavailable: {blocker}")
        if self.artifact_path is None or self.artifact_sha256 is None:
            raise ValueError(f"Backend '{self.name}' has no verifiable artifact")
        if not self.artifact_path.is_file():
            raise FileNotFoundError(
                f"Backend artifact does not exist: {self.artifact_path}"
            )
        actual = sha256_file(self.artifact_path)
        if actual != self.artifact_sha256:
            raise ValueError(
                f"Backend artifact hash mismatch for {self.name}: expected "
                f"{self.artifact_sha256}, found {actual}"
            )
        return actual


@dataclass(frozen=True)
class DeploymentConfig:
    version: int
    source_sha256: str
    input_shape: tuple[int, int, int, int]
    batch_size: int
    backends: dict[str, BackendArtifact]

    @classmethod
    def from_yaml(
        cls, path: Path, repository_root: Path | None = None
    ) -> DeploymentConfig:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Deployment configuration does not exist: {path}")
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Deployment configuration must be a YAML mapping")
        root = repository_root.resolve() if repository_root else path.parents[1]
        try:
            source = values["source"]
            backend_values = values["backends"]
            input_shape = tuple(int(value) for value in source["input_shape"])
            backends = {}
            for name, item in backend_values.items():
                artifact_value = item.get("artifact")
                artifact_path = Path(artifact_value) if artifact_value else None
                if artifact_path is not None and not artifact_path.is_absolute():
                    artifact_path = (root / artifact_path).resolve()
                backends[str(name)] = BackendArtifact(
                    name=str(name),
                    available=bool(item["available"]),
                    artifact_path=artifact_path,
                    artifact_sha256=(
                        str(item["artifact_sha256"]).upper()
                        if item.get("artifact_sha256")
                        else None
                    ),
                    precision=str(item["precision"]),
                    runtime=str(item["runtime"]) if item.get("runtime") else None,
                    metadata=dict(item),
                )
            config = cls(
                version=int(values["version"]),
                source_sha256=str(source["sha256"]).upper(),
                input_shape=input_shape,
                batch_size=int(source["batch_size"]),
                backends=backends,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed deployment configuration: {error}") from error
        config.validate()
        return config

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError("Unsupported deployment configuration version")
        if set(self.backends) != SUPPORTED_BACKENDS:
            raise ValueError("Deployment configuration must define all known backends")
        if self.input_shape != (1, 3, 960, 960) or self.batch_size != 1:
            raise ValueError("Deployment input must remain static batch-1 960x960")
        if len(self.source_sha256) != 64:
            raise ValueError("Source SHA-256 is malformed")

    def select(self, name: str, source_sha256: str) -> BackendArtifact:
        if name not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported detector backend: {name}")
        if source_sha256.upper() != self.source_sha256:
            raise ValueError("Deployment source hash differs from frozen detector")
        artifact = self.backends[name]
        artifact.verify()
        return artifact
