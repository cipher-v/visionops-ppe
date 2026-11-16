from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from visionops.api.app import create_app
from visionops.api.runtime import ProductionRuntime, ServiceSettings, VideoJob


class FakeRuntime:
    def __init__(self, settings: ServiceSettings, _metrics: object) -> None:
        self.settings = settings
        self.device = "cpu"
        self.drift_reference = object()
        self.config = SimpleNamespace(
            architecture="YOLO26n detect",
            weights_sha256="ABCDEF1234567890",
            imgsz=960,
            classes={0: "helmet", 1: "vest", 2: "Person", 3: "no_helmet", 4: "no_vest"},
            confidence_thresholds={
                "helmet": 0.32,
                "vest": 0.19,
                "Person": 0.41,
                "no_helmet": 0.27,
                "no_vest": 0.11,
            },
            nms_iou=0.7,
        )
        self.jobs: dict[str, VideoJob] = {}

    def close(self) -> None:
        pass

    def infer_image(self, image: np.ndarray) -> dict[str, object]:
        height, width = image.shape[:2]
        return {
            "width": width,
            "height": height,
            "processing_latency_ms": 1.0,
            "detections": [],
        }

    def validate_zone(self, _content: bytes, _directory: Path) -> Path:
        return _directory / "zone.yaml"

    def submit_video(self, input_path: Path, zone_path: Path | None) -> VideoJob:
        job = VideoJob(
            "job123",
            input_path,
            self.settings.runtime_dir,
            zone_path,
            status="completed",
            progress=1.0,
            result={
                "frames_processed": 2,
                "drift": {"diagnostic_only": True, "accuracy_failure_predictor": False},
            },
            incidents=[{"incident_id": "i1", "track_id": 1, "equipment": "helmet"}],
        )
        self.jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> VideoJob:
        from visionops.api.errors import ServiceError

        if job_id not in self.jobs:
            raise ServiceError(404, "job_not_found", "Video job does not exist")
        return self.jobs[job_id]


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    settings = ServiceSettings(
        tmp_path, tmp_path / "runtime", max_image_bytes=2048, max_video_bytes=100_000
    )
    with TestClient(create_app(FakeRuntime, settings)) as test_client:
        yield test_client


def _image_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((16, 24, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def _video_bytes(path: Path) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    assert writer.isOpened()
    for value in (0, 127):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def test_health_readiness_config_and_metrics(client: TestClient) -> None:
    assert client.get("/health/live").json() == {"status": "ok"}
    ready = client.get("/health/ready").json()
    assert ready["backend"] == "pytorch"
    config = client.get("/v1/config").json()
    assert config["drift"] == {
        "available": True,
        "diagnostic_only": True,
        "accuracy_failure_predictor": False,
    }
    assert "visionops_requests_total" in client.get("/metrics").text


def test_image_success_and_input_errors(client: TestClient) -> None:
    response = client.post(
        "/v1/infer/image", files={"file": ("sample.jpg", _image_bytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json()["width"] == 24
    malformed = client.post(
        "/v1/infer/image", files={"file": ("bad.jpg", b"bad", "image/jpeg")}
    )
    assert malformed.status_code == 422
    assert malformed.json()["error"]["code"] == "invalid_image"
    unsupported = client.post(
        "/v1/infer/image", files={"file": ("bad.txt", b"bad", "text/plain")}
    )
    assert unsupported.status_code == 415
    oversized = client.post(
        "/v1/infer/image", files={"file": ("big.jpg", b"x" * 2049, "image/jpeg")}
    )
    assert oversized.status_code == 413


def test_video_job_lifecycle_result_incidents_and_missing(
    client: TestClient, tmp_path: Path
) -> None:
    content = _video_bytes(tmp_path / "tiny.mp4")
    submitted = client.post(
        "/v1/jobs/video", files={"file": ("tiny.mp4", content, "video/mp4")}
    )
    assert submitted.status_code == 202
    job_id = submitted.json()["job_id"]
    assert client.get(f"/v1/jobs/{job_id}").json()["status"] == "completed"
    result = client.get(f"/v1/jobs/{job_id}/result").json()["summary"]
    assert result["drift"]["diagnostic_only"] is True
    incidents = client.get(f"/v1/jobs/{job_id}/incidents").json()
    assert incidents["incidents"][0]["track_id"] == 1
    missing = client.get("/v1/jobs/missing")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "job_not_found"


def test_runtime_rejects_corrupt_frozen_weight(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    for name in ("production_detector.yaml", "final_model.yaml"):
        shutil.copy2(repository / "configs" / name, tmp_path / "configs" / name)
    weights = tmp_path / "runs/experiment_mixed5_yolo26n_960/weights/best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"corrupt")
    settings = ServiceSettings(tmp_path, tmp_path / "runtime")
    with pytest.raises(ValueError, match="hash mismatch"):
        ProductionRuntime(settings, SimpleNamespace())
