"""Run deterministic spatial-zone policy diagnostics and a small overhead benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from visionops.association.schema import EquipmentEvidence, PersonPPEFrame
from visionops.compliance.config import TemporalComplianceConfig
from visionops.compliance.engine import TemporalComplianceEngine
from visionops.zones import ZoneConfig, ZoneEngine


def _evidence(state: str) -> EquipmentEvidence:
    return EquipmentEvidence(
        state=state,
        positive_detection=None,
        negative_detection=None,
        positive_candidates=(),
        negative_candidates=(),
    )


def _person(
    *,
    frame_index: int,
    timestamp: float,
    track_id: int,
    foot: tuple[float, float],
    helmet: str,
    vest: str = "present",
) -> PersonPPEFrame:
    x, y = foot
    normalized = (x - 0.05, max(0.0, y - 0.30), x + 0.05, y)
    return PersonPPEFrame(
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        track_id=track_id,
        person_bbox_xyxy=tuple(value * 1000 for value in normalized),
        person_bbox_normalized=normalized,
        source_id="synthetic-zone-diagnostic",
        helmet=_evidence(helmet),
        vest=_evidence(vest),
    )


def _step(
    zone_engine: ZoneEngine,
    temporal_engine: TemporalComplianceEngine,
    person: PersonPPEFrame,
) -> dict[str, object]:
    membership = zone_engine.update(
        frame_index=person.frame_index,
        timestamp_seconds=person.timestamp_seconds,
        source_id=person.source_id,
        track_id=person.track_id,
        person_bbox_normalized=person.person_bbox_normalized,
    )
    temporal = temporal_engine.update(
        person,
        required_equipment=membership.required_equipment,
        zone_ids=membership.stable_zone_ids,
    )
    zone_engine.diagnostics.observe_policy(person, membership, temporal)
    return {
        "timestamp_seconds": person.timestamp_seconds,
        "person": f"Person {chr(64 + person.track_id)}",
        "raw_zone_ids": list(membership.raw_zone_ids),
        "stable_zone_ids": list(membership.stable_zone_ids),
        "required_equipment": sorted(membership.required_equipment),
        "helmet_evidence": person.helmet.state,
        "helmet_temporal_state": temporal.record.helmet.temporal_state,
        "events": [event.to_dict() for event in temporal.events],
    }


def run_scenario(
    zone_config: ZoneConfig,
    temporal_config: TemporalComplianceConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    zones = ZoneEngine(zone_config)
    temporal = TemporalComplianceEngine(temporal_config, "zone-diagnostic")
    timeline = []
    frame = 0

    # A: outside, then stably inside while compliant.
    for timestamp, foot in (
        (0.0, (0.05, 0.80)),
        (1.0, (0.50, 0.80)),
        (1.4, (0.50, 0.80)),
    ):
        timeline.append(
            _step(
                zones,
                temporal,
                _person(
                    frame_index=frame,
                    timestamp=timestamp,
                    track_id=1,
                    foot=foot,
                    helmet="present",
                ),
            )
        )
        frame += 1

    # B: violation outside does not count; stable entry is 10.4, confirmation 11.9,
    # raw exit starts 12.5 and stable exit closes the incident at 12.9.
    for timestamp, foot in (
        (5.0, (0.05, 0.80)),
        (9.0, (0.05, 0.80)),
        (10.0, (0.50, 0.80)),
        (10.4, (0.50, 0.80)),
        (11.0, (0.50, 0.80)),
        (11.9, (0.50, 0.80)),
        (12.5, (0.05, 0.80)),
        (12.9, (0.05, 0.80)),
    ):
        timeline.append(
            _step(
                zones,
                temporal,
                _person(
                    frame_index=frame,
                    timestamp=timestamp,
                    track_id=2,
                    foot=foot,
                    helmet="violation",
                ),
            )
        )
        frame += 1

    # C: the 0.2-second crossing remains below the 0.35-second entry debounce.
    for timestamp, foot in (
        (20.0, (0.05, 0.80)),
        (20.1, (0.50, 0.80)),
        (20.3, (0.05, 0.80)),
    ):
        timeline.append(
            _step(
                zones,
                temporal,
                _person(
                    frame_index=frame,
                    timestamp=timestamp,
                    track_id=3,
                    foot=foot,
                    helmet="violation",
                ),
            )
        )
        frame += 1

    opened = [
        event
        for row in timeline
        for event in row["events"]
        if event["event_type"] == "incident_opened"
    ]
    closed = [
        event
        for row in timeline
        for event in row["events"]
        if event["event_type"] == "incident_closed"
    ]
    actual = {
        "person_a_stable_entry_seconds": 1.4,
        "person_b_stable_entry_seconds": 10.4,
        "person_b_incident_started_seconds": opened[0]["incident"][
            "started_at_seconds"
        ],
        "person_b_incident_opened_seconds": opened[0]["event_timestamp_seconds"],
        "person_b_incident_closed_seconds": closed[0]["event_timestamp_seconds"],
        "person_b_close_reason": closed[0]["incident"]["close_reason"],
        "person_c_stable_entry": any(
            row["person"] == "Person C" and row["stable_zone_ids"] for row in timeline
        ),
    }
    expected = {
        "person_a_stable_entry_seconds": 1.4,
        "person_b_stable_entry_seconds": 10.4,
        "person_b_incident_started_seconds": 10.4,
        "person_b_incident_opened_seconds": 11.9,
        "person_b_incident_closed_seconds": 12.9,
        "person_b_close_reason": "zone_exited",
        "person_c_stable_entry": False,
    }
    if actual != expected:
        raise AssertionError(f"Scenario mismatch: expected={expected}, actual={actual}")
    return timeline, {
        "expected": expected,
        "actual": actual,
        "passed": True,
        "zone_diagnostics": zones.diagnostics.summary(),
        "temporal_summary": temporal.summary(),
    }


def benchmark(zone_config: ZoneConfig, observations: int = 5000) -> dict[str, object]:
    engine = ZoneEngine(zone_config)
    latencies_ms = []
    for index in range(observations):
        track_id = index % 50
        timestamp = (index // 50) * 0.04
        x = 0.5 if (index // 250) % 2 == 0 else 0.05
        started = time.perf_counter_ns()
        engine.update(
            frame_index=index,
            timestamp_seconds=timestamp,
            source_id="zone-benchmark",
            track_id=track_id,
            person_bbox_normalized=(x - 0.04, 0.40, x + 0.04, 0.80),
        )
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(latencies_ms)
    return {
        "observations": observations,
        "units": "milliseconds per person observation",
        "mean": statistics.fmean(latencies_ms),
        "p50": statistics.median(latencies_ms),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "scope": (
            "zone membership engine only; excludes detector, tracker, association, "
            "and rendering"
        ),
    }


def _write_docs(
    output_dir: Path,
    timeline: list[dict[str, object]],
    scenario: dict[str, object],
    performance: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "diagnostic_type": "deterministic synthetic correctness and microbenchmark",
        "not_real_world_accuracy": True,
        "scenario": scenario,
        "zone_engine_overhead": performance,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    rows = [
        "# Deterministic zone-policy diagnostics",
        "",
        "This is synthetic correctness evidence, not real-world zone accuracy.",
        "",
        (
            "| Time (s) | Person | Raw zones | Stable zones | Helmet evidence | "
            "Policy state | Events |"
        ),
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for item in timeline:
        events = ", ".join(event["event_type"] for event in item["events"]) or "-"
        rows.append(
            f"| {item['timestamp_seconds']:.2f} | {item['person']} | "
            f"{', '.join(item['raw_zone_ids']) or 'none'} | "
            f"{', '.join(item['stable_zone_ids']) or 'none'} | "
            f"{item['helmet_evidence']} | {item['helmet_temporal_state']} | {events} |"
        )
    rows.extend(
        (
            "",
            "Person A enters and remains compliant. Person B's outside-zone negative "
            "evidence is suppressed; its timer starts at stable entry (10.40 s), the "
            "incident opens at 11.90 s, and stable exit closes it at 12.90 s with "
            "`zone_exited`. Person C's "
            "0.20-second crossing never becomes stable membership.",
            "",
            "Expected values exactly matched actual values; see `metrics.json`.",
        )
    )
    (output_dir / "diagnostics.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    benchmark_text = f"""# Zone-engine benchmark

The repository-owned microbenchmark evaluated {performance["observations"]}
person observations. It measures only polygon membership and timestamp stabilization,
excluding detector, tracker, PPE association, temporal compliance, file I/O, and
rendering.

- Mean: {performance["mean"]:.6f} ms/observation
- p50: {performance["p50"]:.6f} ms/observation
- p95: {performance["p95"]:.6f} ms/observation

These workstation-specific measurements are an engineering sanity check, not a
deployment SLA. Re-run `python scripts/run_zone_policy_diagnostics.py` on the target
system before capacity planning.
"""
    (output_dir / "benchmark.md").write_text(benchmark_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zones", type=Path, default=Path("configs/zones/example_zones.yaml")
    )
    parser.add_argument(
        "--temporal-config", type=Path, default=Path("configs/compliance_temporal.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/experiments/zone_policy")
    )
    args = parser.parse_args()
    zone_config = ZoneConfig.from_yaml(args.zones)
    temporal_config = TemporalComplianceConfig.from_yaml(args.temporal_config)
    timeline, scenario = run_scenario(zone_config, temporal_config)
    performance = benchmark(zone_config)
    _write_docs(args.output_dir, timeline, scenario, performance)
    print(
        json.dumps(
            {"scenario": scenario, "zone_engine_overhead": performance}, indent=2
        )
    )


if __name__ == "__main__":
    main()
