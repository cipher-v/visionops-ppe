from pathlib import Path

import numpy as np
import pytest

from visionops.deployment.config import DeploymentConfig
from visionops.deployment.parity import match_detections, summarize_parity
from visionops.deployment.profiler import StageProfiler
from visionops.video.backend import DetectorBackend
from visionops.video.config import load_frozen_detector_config
from visionops.video.pipeline import PipelineOptions, _structured_detections
from visionops.video.policy import CandidateBox
from visionops.video.schema import VideoDetection
from visionops.video.source import FramePacket

ROOT = Path(__file__).parents[1]


def _box(class_id: int, confidence: float, xyxy=(0.0, 0.0, 10.0, 10.0)):
    return CandidateBox(class_id, confidence, xyxy)


def test_deployment_config_loads_and_verifies_artifacts() -> None:
    config = DeploymentConfig.from_yaml(ROOT / "configs/deployment_backends.yaml", ROOT)
    artifact = config.select("pytorch", config.source_sha256)
    assert artifact.name == "pytorch"
    assert artifact.verify() == config.source_sha256


def test_backend_selection_rejects_unavailable_and_unknown() -> None:
    config = DeploymentConfig.from_yaml(ROOT / "configs/deployment_backends.yaml", ROOT)
    with pytest.raises(RuntimeError, match="unavailable"):
        config.select("tensorrt", config.source_sha256)
    with pytest.raises(ValueError, match="Unsupported"):
        config.select("openvino", config.source_sha256)


def test_backend_hash_mismatch_rejects(tmp_path: Path) -> None:
    source = (ROOT / "configs/deployment_backends.yaml").read_text(encoding="utf-8")
    altered = source.replace(
        "BF99E872A9B57A7C33A4EBF06F0620F9EB8A215AEC29DDE2A90CA1F72F9581F4",
        "A" * 64,
    )
    path = tmp_path / "deployment.yaml"
    path.write_text(altered, encoding="utf-8")
    config = DeploymentConfig.from_yaml(path, ROOT)
    with pytest.raises(ValueError, match="hash"):
        config.select("pytorch", "A" * 64)


def test_deterministic_parity_matcher_and_summary() -> None:
    reference = [_box(0, 0.8), _box(1, 0.7, (20, 20, 30, 30))]
    candidate = [
        _box(1, 0.69, (20, 20, 30, 30)),
        _box(0, 0.81),
        _box(2, 0.9),
    ]
    matches = match_detections(reference, candidate)
    assert [(match.reference_index, match.candidate_index) for match in matches] == [
        (0, 1),
        (1, 0),
    ]
    summary = summarize_parity(reference, candidate, matches)
    assert summary["matched_detections"] == 2
    assert summary["unmatched_candidate_detections"] == 1
    assert summary["mean_matched_iou"] == 1.0


def test_stage_profiler_accounting() -> None:
    profiler = StageProfiler()
    profiler.record("decode", 0.001)
    profiler.record("decode", 0.003)
    summary = profiler.summary()["decode"]
    assert summary["count"] == 2
    assert summary["total_ms"] == pytest.approx(4.0)
    assert summary["p50_ms"] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        profiler.record("", 1.0)


def test_pytorch_default_and_common_backend_contract() -> None:
    frozen = load_frozen_detector_config(
        ROOT / "configs/production_detector.yaml", ROOT
    )
    backend = DetectorBackend("pytorch", frozen)
    assert backend.name == "pytorch"
    assert callable(backend.predict)
    assert backend.metadata()["artifact_sha256"] == frozen.weights_sha256
    options = PipelineOptions(source="video.mp4", output_dir=Path("results/test"))
    assert options.detector_backend == "pytorch"


def test_common_candidate_output_becomes_repository_detection_schema() -> None:
    frozen = load_frozen_detector_config(
        ROOT / "configs/production_detector.yaml", ROOT
    )
    packet = FramePacket(3, 0.12, np.zeros((100, 200, 3), dtype=np.uint8))
    detections = _structured_detections(
        packet,
        [_box(0, 0.8, (10.0, 20.0, 50.0, 80.0))],
        frozen,
        "synthetic",
    )
    assert len(detections) == 1
    assert isinstance(detections[0], VideoDetection)
    assert detections[0].to_dict()["class_name"] == "helmet"


def test_pipeline_rejects_unsupported_or_unconfigured_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        PipelineOptions(
            source="video.mp4",
            output_dir=Path("results/test"),
            detector_backend="unknown",
        )
    with pytest.raises(ValueError, match="deployment configuration"):
        PipelineOptions(
            source="video.mp4",
            output_dir=Path("results/test"),
            detector_backend="onnx",
        )
