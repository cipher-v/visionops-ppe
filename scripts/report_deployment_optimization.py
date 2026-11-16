"""Consolidate detector and full-video deployment measurements."""

from __future__ import annotations

import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/deployment_optimization"
DOCS = ROOT / "docs/experiments/deployment_optimization"
BACKENDS = ("pytorch", "onnx")
CLIPS = ("compliant_multiworker", "helmet_transition", "walking_multiworker")


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_summary(backend: str, clip: str) -> dict[str, object]:
    directory = RESULTS / "video" / backend / clip
    metadata = json.loads((directory / "run_metadata.json").read_text())
    detections = _jsonl(directory / "detections.jsonl")
    tracks = _jsonl(directory / "tracks.jsonl")
    associations = _jsonl(directory / "ppe_associations.jsonl")
    states = _jsonl(directory / "compliance_states.jsonl")
    incidents = _jsonl(directory / "incidents.jsonl")
    class_counts = Counter(item["class_name"] for item in detections)
    temporal_counts = Counter(
        f"{equipment}:{item[equipment]['temporal_state']}"
        for item in states
        for equipment in ("helmet", "vest")
    )
    incident_signature = [
        {
            "event_type": item["event_type"],
            "equipment": item["incident"]["equipment"],
            "started_at_seconds": item["incident"]["started_at_seconds"],
            "confirmed_at_seconds": item["incident"]["confirmed_at_seconds"],
            "cleared_at_seconds": item["incident"]["cleared_at_seconds"],
            "close_reason": item["incident"]["close_reason"],
        }
        for item in incidents
    ]
    return {
        "frames": metadata["processing"]["frames_processed"],
        "resolution": [
            metadata["source"]["original_width"],
            metadata["source"]["original_height"],
        ],
        "effective_fps": metadata["timing"]["effective_pipeline_fps"],
        "detector_latency": metadata["timing"]["detector_latency"],
        "end_to_end_latency": metadata["timing"]["end_to_end_frame_latency"],
        "stage_profile": metadata["timing"]["stage_profile"],
        "detector_class_counts": dict(sorted(class_counts.items())),
        "unique_tracks": len({item["track_id"] for item in tracks}),
        "track_observations": len(tracks),
        "association_records": len(associations),
        "association_states": metadata["ppe_association"]["diagnostics"][
            "frame_level_equipment_states"
        ],
        "temporal_state_counts": dict(sorted(temporal_counts.items())),
        "incidents": incident_signature,
    }


def _aggregate(runs: dict[str, dict[str, object]]) -> dict[str, object]:
    detector_samples = []
    end_to_end_samples = []
    total_frames = 0
    total_wall = 0.0
    for clip, summary in runs.items():
        metadata = json.loads(
            (
                RESULTS / "video" / summary["backend"] / clip / "run_metadata.json"
            ).read_text()
        )
        detector_samples.extend(metadata["timing"]["latency_samples_ms"]["detector"])
        end_to_end_samples.extend(
            metadata["timing"]["latency_samples_ms"]["end_to_end"]
        )
        total_frames += metadata["processing"]["frames_processed"]
        total_wall += metadata["timing"]["processing_wall_clock_seconds"]

    def latency(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "mean_ms": sum(values) / len(values),
            "p50_ms": ordered[round((len(ordered) - 1) * 0.50)],
            "p95_ms": ordered[round((len(ordered) - 1) * 0.95)],
        }

    return {
        "frames": total_frames,
        "effective_fps": total_frames / total_wall,
        "detector_latency": latency(detector_samples),
        "end_to_end_latency": latency(end_to_end_samples),
    }


def main() -> None:
    detector = json.loads(
        (RESULTS / "detector_evaluation.json").read_text(encoding="utf-8")
    )
    video = {}
    aggregates = {}
    for backend in BACKENDS:
        video[backend] = {}
        runs_for_aggregate = {}
        for clip in CLIPS:
            summary = _run_summary(backend, clip)
            video[backend][clip] = summary
            runs_for_aggregate[clip] = {**summary, "backend": backend}
        aggregates[backend] = _aggregate(runs_for_aggregate)
    comparisons = {}
    for clip in CLIPS:
        reference = video["pytorch"][clip]
        candidate = video["onnx"][clip]
        comparisons[clip] = {
            "detector_class_counts_equal": (
                reference["detector_class_counts"] == candidate["detector_class_counts"]
            ),
            "unique_tracks_equal": (
                reference["unique_tracks"] == candidate["unique_tracks"]
            ),
            "association_states_equal": (
                reference["association_states"] == candidate["association_states"]
            ),
            "temporal_states_equal": (
                reference["temporal_state_counts"] == candidate["temporal_state_counts"]
            ),
            "incidents_equal": reference["incidents"] == candidate["incidents"],
            "pytorch_incidents": reference["incidents"],
            "onnx_incidents": candidate["incidents"],
        }
    metrics = {
        "checkpoint": 15,
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "gpu": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
            "nvidia_driver": "546.18",
            "driver_cuda_capability": "12.3",
            "torch_cuda_runtime": "12.1",
            "packages": {
                name: version(name)
                for name in (
                    "torch",
                    "torchvision",
                    "ultralytics",
                    "opencv-python",
                    "onnx",
                    "onnxruntime-gpu",
                )
            },
            "onnxruntime_providers": [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            "onnx_provider_used": "CUDAExecutionProvider",
        },
        "detector_evaluation": detector,
        "video_runs": video,
        "video_aggregate": aggregates,
        "behavioral_comparison": comparisons,
        "tensorrt": {
            "status": "not_built",
            "blocker": (
                "No installed TensorRT runtime, Python bindings, or trtexec. The local "
                "CUDA toolkit is 11.8 while PyTorch uses CUDA 12.1 and the driver "
                "exposes "
                "CUDA 12.3; an additional runtime stack was not forced."
            ),
        },
        "decision": {
            "preferred_backend": "pytorch",
            "onnx_accepted": detector["detection_parity"]["accepted"],
            "reason": (
                "ONNX failed predeclared detection parity and was slower on this GPU; "
                "PyTorch remains the production reference and selected backend."
            ),
        },
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "video_aggregate": aggregates,
                "behavioral_comparison": comparisons,
                "decision": metrics["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
