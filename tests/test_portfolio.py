from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from visionops.api.app import create_app
from visionops.api.auth import WebSettings, hash_password
from visionops.api.errors import ServiceError
from visionops.api.runtime import ServiceSettings, VideoJob

pytestmark = pytest.mark.integration


class PortfolioRuntime:
    def __init__(self, settings: ServiceSettings, _metrics: object) -> None:
        self.settings = settings
        self.device = "cpu"
        self.drift_reference = None
        self.config = SimpleNamespace(
            architecture="YOLO26n detect",
            weights_sha256="ABCDEF1234567890",
            imgsz=960,
            classes={0: "helmet", 1: "vest", 2: "Person"},
            confidence_thresholds={"helmet": 0.32, "vest": 0.19, "Person": 0.41},
            nms_iou=0.7,
        )
        self.jobs: dict[str, VideoJob] = {}

    def close(self) -> None:
        pass

    def validate_zone(self, _content: bytes, directory: Path) -> Path:
        return directory / "zones.yaml"

    def submit_video(
        self,
        input_path: Path,
        zone_path: Path | None,
        presentation_path: Path | None = None,
    ) -> VideoJob:
        job_id = uuid.uuid4().hex
        output_dir = self.settings.runtime_dir / "jobs" / job_id
        artifacts = output_dir / "artifacts"
        artifacts.mkdir(parents=True)
        annotated = artifacts / "annotated.mp4"
        annotated.write_bytes(input_path.read_bytes())
        presentation_dir = output_dir / "presentation"
        presentation_dir.mkdir()
        original_presentation = presentation_dir / "original.mp4"
        annotated_presentation = presentation_dir / "annotated.mp4"
        assert presentation_path is not None
        original_presentation.write_bytes(presentation_path.read_bytes())
        annotated_presentation.write_bytes(input_path.read_bytes())
        presentation_path.unlink()
        (artifacts / "incidents.jsonl").write_text(
            json.dumps(
                {
                    "incident_id": "incident-1",
                    "track_id": 4,
                    "equipment": "helmet",
                    "started_at_seconds": 1.5,
                    "duration_seconds": 2.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (artifacts / "run_metadata.json").write_text(
            json.dumps(
                {
                    "source": {"duration_seconds": 0.4},
                    "processing": {"frames_processed": 2},
                    "tracking": {"diagnostics": {"unique_tracks_created": 1}},
                }
            ),
            encoding="utf-8",
        )
        input_path.unlink()
        job = VideoJob(
            job_id,
            output_dir / "removed-input.mp4",
            output_dir,
            zone_path,
            presentation_original_path=original_presentation,
            presentation_annotated_path=annotated_presentation,
            status="completed",
            progress=1.0,
            result={"frames_processed": 2},
            incidents=[{"incident_id": "incident-1"}],
        )
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> VideoJob:
        if job_id not in self.jobs:
            raise ServiceError(404, "job_not_found", "Video job does not exist")
        return self.jobs[job_id]


@pytest.fixture
def portfolio_client(tmp_path: Path) -> TestClient:
    settings = ServiceSettings(
        tmp_path,
        tmp_path / "runtime",
        max_image_bytes=2048,
        max_video_bytes=100_000,
    )
    web_settings = WebSettings(
        tmp_path / "runtime/portfolio.sqlite3",
        "owner",
        hash_password("correct horse", salt=b"0123456789abcdef"),
        "portfolio-test-session-secret-that-is-long-enough",
    )
    with TestClient(
        create_app(PortfolioRuntime, settings, web_settings), follow_redirects=False
    ) as client:
        yield client


def _video_bytes(path: Path) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    assert writer.isOpened()
    for value in (0, 127):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _login(client: TestClient) -> None:
    response = client.post(
        "/admin/login",
        data={"username": "owner", "password": "correct horse"},
    )
    assert response.status_code == 303


def test_login_and_admin_access_control(portfolio_client: TestClient) -> None:
    assert portfolio_client.get("/admin").status_code == 303
    assert portfolio_client.get("/docs").status_code == 404
    assert portfolio_client.get("/openapi.json").status_code == 404
    assert portfolio_client.get("/api/admin/videos").status_code == 401
    assert portfolio_client.post("/api/admin/videos/missing/publish").status_code == 401
    invalid = portfolio_client.post(
        "/admin/login", data={"username": "owner", "password": "wrong"}
    )
    assert invalid.status_code == 401
    valid = portfolio_client.post(
        "/admin/login",
        data={"username": "owner", "password": "correct horse"},
        headers={"x-forwarded-proto": "https"},
    )
    assert valid.status_code == 303
    cookie = valid.headers["set-cookie"]
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Secure" in cookie
    _login(portfolio_client)
    assert portfolio_client.get("/admin").status_code == 200


def test_upload_publish_unpublish_and_public_visibility(
    portfolio_client: TestClient, tmp_path: Path
) -> None:
    content = _video_bytes(tmp_path / "source.mp4")
    home = portfolio_client.get("/")
    assert home.status_code == 200
    assert "Portfolio safety run" not in home.text

    _login(portfolio_client)
    uploaded = portfolio_client.post(
        "/api/admin/videos",
        data={"title": "Portfolio safety run"},
        files={"file": ("../../unsafe-name.mp4", content, "video/mp4")},
    )
    assert uploaded.status_code == 202
    video = uploaded.json()
    assert video["status"] == "completed"
    assert video["source_filename"] == "unsafe-name.mp4"
    assert video["original_url"].startswith("/media/originals/")
    video_id = video["id"]
    admin_page = portfolio_client.get("/admin")
    assert "/static/sync-player.js" in admin_page.text
    assert "data-sync-player" in portfolio_client.get("/static/admin.js").text

    assert "Portfolio safety run" not in portfolio_client.get("/").text
    assert portfolio_client.get(f"/watch/{video_id}").status_code == 404
    assert portfolio_client.get(f"/media/videos/{video_id}").status_code == 200
    assert portfolio_client.get(f"/media/originals/{video_id}").content == content
    for media_url in (video["original_url"], video["video_url"]):
        partial = portfolio_client.get(media_url, headers={"Range": "bytes=0-99"})
        assert partial.status_code == 206
        assert partial.headers["content-range"].startswith("bytes 0-99/")
        assert len(partial.content) == 100

    portfolio_client.post("/admin/logout")
    assert portfolio_client.get(f"/media/videos/{video_id}").status_code == 404
    assert portfolio_client.get(f"/media/originals/{video_id}").status_code == 404
    _login(portfolio_client)

    published = portfolio_client.post(f"/api/admin/videos/{video_id}/publish")
    assert published.status_code == 200
    assert published.json()["published"] is True
    public_home = portfolio_client.get("/")
    assert "Portfolio safety run" in public_home.text
    watch = portfolio_client.get(f"/watch/{video_id}")
    assert watch.status_code == 200
    assert "1 incident" in watch.text
    assert "Original" in watch.text
    assert "VisionOps Output" in watch.text
    assert "data-sync-player" in watch.text
    assert "/static/sync-player.js" in watch.text

    portfolio_client.post("/admin/logout")
    assert portfolio_client.get(f"/media/videos/{video_id}").status_code == 200
    assert portfolio_client.get(f"/media/originals/{video_id}").content == content
    assert portfolio_client.get("/uploads/unsafe-name.mp4").status_code == 404
    assert portfolio_client.get("/runtime/incoming/unsafe-name.mp4").status_code == 404
    assert (
        portfolio_client.post(f"/api/admin/videos/{video_id}/unpublish").status_code
        == 401
    )

    _login(portfolio_client)
    unpublished = portfolio_client.post(f"/api/admin/videos/{video_id}/unpublish")
    assert unpublished.status_code == 200
    assert unpublished.json()["published"] is False
    assert "Portfolio safety run" not in portfolio_client.get("/").text
    portfolio_client.post("/admin/logout")
    assert portfolio_client.get(f"/media/videos/{video_id}").status_code == 404
    assert portfolio_client.get(f"/media/originals/{video_id}").status_code == 404


def test_admin_can_delete_completed_output(
    portfolio_client: TestClient, tmp_path: Path
) -> None:
    _login(portfolio_client)
    uploaded = portfolio_client.post(
        "/api/admin/videos",
        files={
            "file": ("delete.mp4", _video_bytes(tmp_path / "delete.mp4"), "video/mp4")
        },
    )
    video_id = uploaded.json()["id"]
    deleted = portfolio_client.delete(f"/api/admin/videos/{video_id}")
    assert deleted.status_code == 204
    assert portfolio_client.get(f"/api/admin/videos/{video_id}").status_code == 404


def test_admin_frontend_has_distinct_job_states(portfolio_client: TestClient) -> None:
    script = portfolio_client.get("/static/admin.js").text
    assert "Uploading source video..." in script
    assert "Upload complete. Job accepted." in script
    assert "Queued" in script
    assert "Processing with VisionOps..." in script
    assert "Processing complete. Preview is ready." in script
    assert "Processing failed" in script
    assert "window.setInterval(refresh, 1500)" in script
    assert "signature === renderedSignature" in script
    assert 'querySelectorAll("video").forEach((video) => video.load())' in script
    synchronization = portfolio_client.get("/static/sync-player.js").text
    assert "DRIFT_TOLERANCE_SECONDS = 0.25" in synchronization
    assert "if (!ready(other)) return" in synchronization
