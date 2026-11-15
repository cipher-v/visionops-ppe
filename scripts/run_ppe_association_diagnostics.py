"""Run deterministic injected-geometry PPE association diagnostics."""

from __future__ import annotations

import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from visionops.association.associator import PPEAssociator
from visionops.association.config import AssociationConfig
from visionops.association.metrics import AssociationDiagnostics
from visionops.tracking.schema import PersonTrack
from visionops.video.schema import VideoDetection

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ppe_association.yaml"
OUTPUT = ROOT / "docs/experiments/ppe_association/diagnostic_results.json"
NAMES = {0: "helmet", 1: "vest", 3: "no_helmet", 4: "no_vest"}


def track(track_id: int, box: tuple[float, float, float, float]) -> PersonTrack:
    return PersonTrack(
        frame_index=0,
        timestamp_seconds=0.0,
        track_id=track_id,
        class_id=2,
        class_name="Person",
        confidence=0.95,
        bbox_xyxy=box,
        bbox_normalized=tuple(value / 500 for value in box),
        source_id="crowded-synthetic",
        track_state="tracked",
    )


def detection(
    class_id: int, box: tuple[float, float, float, float], confidence: float = 0.9
) -> VideoDetection:
    return VideoDetection(
        frame_index=0,
        timestamp_seconds=0.0,
        class_id=class_id,
        class_name=NAMES[class_id],
        confidence=confidence,
        bbox_xyxy=box,
        bbox_normalized=tuple(value / 500 for value in box),
        source_id="crowded-synthetic",
    )


def assigned_owners(records) -> dict[int, int]:
    owners = {}
    for record in records:
        for evidence in (record.helmet, record.vest):
            for match in (
                *evidence.positive_candidates,
                *evidence.negative_candidates,
            ):
                owners[match.detection_index] = record.track_id
    return owners


def crowded_scenario(associator: PPEAssociator):
    tracks = (
        track(1, (20.0, 20.0, 140.0, 420.0)),
        track(2, (100.0, 20.0, 220.0, 420.0)),
        track(3, (180.0, 20.0, 300.0, 420.0)),
    )
    detections = [
        detection(0, (45.0, 40.0, 85.0, 90.0)),
        detection(0, (130.0, 40.0, 170.0, 90.0)),
        detection(0, (215.0, 40.0, 255.0, 90.0)),
        detection(1, (45.0, 160.0, 100.0, 290.0)),
        detection(1, (130.0, 160.0, 190.0, 290.0)),
        detection(1, (220.0, 160.0, 275.0, 290.0)),
        detection(3, (110.0, 50.0, 130.0, 100.0)),
        detection(4, (305.0, 150.0, 330.0, 280.0)),
    ]
    expected = {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 1, 7: None}
    result = associator.associate(tracks, detections)
    owners = assigned_owners(result.records)
    observed = {index: owners.get(index) for index in range(len(detections))}
    correct = sum(observed[index] == owner for index, owner in expected.items())
    return (
        tracks,
        detections,
        result,
        {
            "expected_owner_by_detection": expected,
            "observed_owner_by_detection": observed,
            "correct_assignments": correct,
            "total_expected_decisions": len(expected),
            "synthetic_decision_accuracy": correct / len(expected),
            "assigned": len(result.assigned_detection_indices),
            "unassigned": len(result.unassigned_detection_indices),
            "ambiguous": result.ambiguous_detections,
            "records": [record.to_dict() for record in result.records],
        },
    )


def timing_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "iterations": len(values),
        "mean_ms": statistics.fmean(values) * 1000,
        "p50_ms": statistics.median(values) * 1000,
        "p95_ms": percentile(0.95) * 1000,
    }


def main() -> None:
    associator = PPEAssociator(AssociationConfig.from_yaml(CONFIG))
    tracks, detections, result, crowded = crowded_scenario(associator)
    metrics = AssociationDiagnostics()
    metrics.update(result)
    timings = []
    for _ in range(1000):
        started = time.perf_counter()
        associator.associate(tracks, detections)
        timings.append(time.perf_counter() - started)
    values = {
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "known synthetic geometry; not real-world association performance",
        "configuration": associator.metadata(),
        "crowded_scene": crowded,
        "diagnostics": metrics.summary(),
        "crowded_scene_association_timing": timing_summary(timings),
    }
    OUTPUT.write_text(json.dumps(values, indent=2), encoding="utf-8")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
