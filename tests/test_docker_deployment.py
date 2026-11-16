from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_uses_cuda_runtime_and_non_root_service() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime")
    assert "ffmpeg" in dockerfile
    assert "libglib2.0-0" in dockerfile
    assert "USER visionops" in dockerfile
    assert '--workers", "1' in dockerfile
    assert "/health/live" in dockerfile
    assert "COPY runs" not in dockerfile
    assert "COPY ." not in dockerfile


def test_compose_gpu_weight_and_persistent_volume_contract() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    service = compose["services"]["visionops"]
    assert service["gpus"] == "all"
    assert service["ports"] == ["8000:8000"]
    assert service["restart"] == "unless-stopped"
    assert service["environment"]["VISIONOPS_RUNTIME_DIR"] == "/var/lib/visionops"
    assert {
        "ADMIN_USERNAME",
        "ADMIN_PASSWORD_HASH",
        "SESSION_SECRET",
    }.issubset(service["environment"])
    assert "visionops-data:/var/lib/visionops" in service["volumes"]
    weight_mount = next(item for item in service["volumes"] if isinstance(item, dict))
    assert weight_mount["read_only"] is True
    assert weight_mount["bind"]["create_host_path"] is False
    assert weight_mount["source"].endswith("weights/best.pt")
    assert weight_mount["target"].endswith("weights/best.pt")


def test_dockerignore_excludes_runtime_and_secret_material() -> None:
    patterns = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())
    assert {".git", ".venv", ".env", "data", "results", "runs", "*.pt"}.issubset(
        patterns
    )
