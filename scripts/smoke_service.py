"""Run local, real-runtime API smoke requests without external network access."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from visionops.api.app import create_app
from visionops.api.runtime import ServiceSettings

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "service_smoke"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image = np.zeros((128, 192, 3), dtype=np.uint8)
    cv2.rectangle(image, (55, 15), (135, 120), (180, 180, 180), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Could not encode smoke image")
    video_path = OUTPUT / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (192, 128)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create smoke video")
    for offset in range(3):
        writer.write(np.roll(image, offset * 2, axis=1))
    writer.release()

    started = time.perf_counter()
    settings = ServiceSettings(ROOT, OUTPUT / "runtime")
    with TestClient(create_app(settings=settings)) as client:
        startup_seconds = time.perf_counter() - started
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        image_started = time.perf_counter()
        image_response = client.post(
            "/v1/infer/image",
            files={"file": ("smoke.jpg", encoded.tobytes(), "image/jpeg")},
        )
        image_wall_ms = (time.perf_counter() - image_started) * 1000
        submitted = client.post(
            "/v1/jobs/video",
            files={"file": ("tiny.mp4", video_path.read_bytes(), "video/mp4")},
        )
        job_id = submitted.json()["job_id"]
        statuses = []
        for _ in range(120):
            status = client.get(f"/v1/jobs/{job_id}").json()
            statuses.append(status["status"])
            if status["status"] in {"completed", "failed"}:
                break
            time.sleep(0.1)
        result = client.get(f"/v1/jobs/{job_id}/result")
        incidents = client.get(f"/v1/jobs/{job_id}/incidents")
        metrics = client.get("/metrics")
    report = {
        "startup_seconds": startup_seconds,
        "health_live": {"status_code": live.status_code, "body": live.json()},
        "health_ready": {"status_code": ready.status_code, "body": ready.json()},
        "image": {
            "status_code": image_response.status_code,
            "wall_latency_ms": image_wall_ms,
            "body": image_response.json(),
        },
        "video": {
            "submit_status_code": submitted.status_code,
            "statuses_observed": statuses,
            "result_status_code": result.status_code,
            "result": result.json(),
            "incidents_status_code": incidents.status_code,
            "incidents": incidents.json(),
        },
        "metrics_status_code": metrics.status_code,
        "metrics_sample": metrics.text.splitlines()[:20],
    }
    (OUTPUT / "smoke_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
