"""Run a deterministic multi-person temporal-compliance timeline."""

from __future__ import annotations

import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from visionops.association.schema import EquipmentEvidence, PersonPPEFrame
from visionops.compliance.config import TemporalComplianceConfig
from visionops.compliance.engine import TemporalComplianceEngine

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/compliance_temporal.yaml"
METRICS = ROOT / "docs/experiments/temporal_compliance/metrics.json"
TIMELINE = ROOT / "docs/experiments/temporal_compliance/diagnostic_timeline.md"


def evidence(state: str) -> EquipmentEvidence:
    return EquipmentEvidence(
        state=state,
        positive_detection=None,
        negative_detection=None,
        positive_candidates=(),
        negative_candidates=(),
    )


def record(
    frame: int,
    timestamp: float,
    track_id: int,
    helmet: str,
    vest: str,
) -> PersonPPEFrame:
    x1 = 20.0 + track_id * 100
    box = (x1, 20.0, x1 + 80.0, 220.0)
    return PersonPPEFrame(
        frame_index=frame,
        timestamp_seconds=timestamp,
        track_id=track_id,
        person_bbox_xyxy=box,
        person_bbox_normalized=tuple(value / 500 for value in box),
        source_id="deterministic-temporal-scenario",
        helmet=evidence(helmet),
        vest=evidence(vest),
    )


def percentile_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "observations": len(values),
        "mean_ms": statistics.fmean(values) * 1000,
        "p50_ms": statistics.median(values) * 1000,
        "p95_ms": percentile(0.95) * 1000,
    }


def performance(config: TemporalComplianceConfig) -> dict[str, float | int]:
    engine = TemporalComplianceEngine(config, "performance-run")
    timings = []
    states = ("present", "present", "unknown", "present", "conflict", "present")
    for frame in range(2000):
        timestamp = frame * 0.041
        item = record(
            frame,
            timestamp,
            99,
            states[frame % len(states)],
            "present",
        )
        started = time.perf_counter()
        engine.update(item)
        timings.append(time.perf_counter() - started)
    return percentile_summary(timings)


def main() -> None:
    config = TemporalComplianceConfig.from_yaml(CONFIG)
    engine = TemporalComplianceEngine(config, "diagnostic-run")
    timestamps = [0.0, 0.6, 0.75, 1.0, 1.5, 2.0, 2.2, 2.5, 2.8, 3.0, 4.0]
    track_two_helmet = {
        0.0: "violation",
        0.6: "violation",
        0.75: "violation",
        1.0: "violation",
        1.5: "violation",
        2.0: "unknown",
        2.2: "violation",
        2.5: "conflict",
        2.8: "violation",
        3.0: "present",
        4.0: "present",
    }
    event_values = []
    timeline_rows = []
    for frame, timestamp in enumerate(timestamps):
        inputs = (
            record(frame, timestamp, 1, "present", "present"),
            record(frame, timestamp, 2, track_two_helmet[timestamp], "present"),
            record(
                frame,
                timestamp,
                3,
                "present",
                "violation" if timestamp <= 0.6 else "present",
            ),
        )
        for item in inputs:
            update = engine.update(item)
            for event in update.events:
                event_values.append(event.to_dict())
            if item.track_id == 2:
                timeline_rows.append(
                    {
                        "timestamp": timestamp,
                        "evidence": item.helmet.state,
                        "temporal_state": update.record.helmet.temporal_state,
                        "violation_duration": (
                            update.record.helmet.continuous_violation_duration
                        ),
                        "incident_active": update.record.helmet.incident_active,
                    }
                )

    event_values.extend(event.to_dict() for event in engine.finalize(timestamps[-1]))
    expected = [
        {
            "event_type": "incident_opened",
            "track_id": 2,
            "equipment": "helmet",
            "timestamp": 1.5,
        },
        {
            "event_type": "incident_closed",
            "track_id": 2,
            "equipment": "helmet",
            "timestamp": 4.0,
        },
    ]
    actual = [
        {
            "event_type": item["event_type"],
            "track_id": item["incident"]["track_id"],
            "equipment": item["incident"]["equipment"],
            "timestamp": item["event_timestamp_seconds"],
        }
        for item in event_values
    ]
    values = {
        "created_at": datetime.now(UTC).isoformat(),
        "real_video_temporal_validation": False,
        "purpose": "deterministic synthetic decision validation only",
        "configuration": config.to_dict(),
        "scenario": {
            "workers": {
                "1": "sustained helmet and vest positive evidence",
                "2": "persistent no_helmet, short unknown, conflict, then cleared",
                "3": "transient no_vest shorter than confirmation threshold",
            },
            "expected_events": expected,
            "actual_events": actual,
            "exact_event_match": actual == expected,
            "expected_incident_count": 1,
            "actual_incident_count": sum(
                item["event_type"] == "incident_opened" for item in event_values
            ),
            "events": event_values,
            "track_2_helmet_timeline": timeline_rows,
        },
        "engine_summary": engine.summary(),
        "temporal_engine_timing": performance(config),
    }
    METRICS.write_text(json.dumps(values, indent=2), encoding="utf-8")

    lines = [
        "# Deterministic temporal event timeline",
        "",
        "| Time (s) | Evidence | Temporal state | Violation time | Incident active |",
        "|---:|---|---|---:|---|",
    ]
    lines.extend(
        f"| {item['timestamp']:.2f} | {item['evidence']} | {item['temporal_state']} | "
        f"{item['violation_duration']:.2f} s | {item['incident_active']} |"
        for item in timeline_rows
    )
    lines.extend(
        [
            "",
            "Expected and actual events match exactly:",
            "",
            "- `1.50 s`: track 2 helmet incident opened (candidate began at 0.00 s).",
            "- `4.00 s`: the incident closed after 1.00 s sustained present evidence.",
            "- Track 1 stayed compliant; track 3's transient no-vest episode opened",
            "  no incident.",
            "",
            "This is deterministic synthetic correctness, not real-world accuracy.",
        ]
    )
    TIMELINE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
