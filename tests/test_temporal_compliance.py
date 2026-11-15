from pathlib import Path

import pytest

from visionops.association.schema import EquipmentEvidence, PersonPPEFrame
from visionops.compliance.config import TemporalComplianceConfig
from visionops.compliance.engine import TemporalComplianceEngine
from visionops.video.pipeline import PipelineOptions, build_temporal_metadata

ROOT = Path(__file__).parents[1]


def _evidence(state: str) -> EquipmentEvidence:
    return EquipmentEvidence(
        state=state,
        positive_detection=None,
        negative_detection=None,
        positive_candidates=(),
        negative_candidates=(),
    )


def _record(
    frame: int,
    timestamp: float,
    track_id: int = 1,
    helmet: str = "unknown",
    vest: str = "unknown",
) -> PersonPPEFrame:
    return PersonPPEFrame(
        frame_index=frame,
        timestamp_seconds=timestamp,
        track_id=track_id,
        person_bbox_xyxy=(10.0, 10.0, 100.0, 200.0),
        person_bbox_normalized=(0.01, 0.01, 0.1, 0.2),
        source_id="synthetic",
        helmet=_evidence(helmet),
        vest=_evidence(vest),
    )


@pytest.fixture
def config() -> TemporalComplianceConfig:
    return TemporalComplianceConfig.from_yaml(ROOT / "configs/compliance_temporal.yaml")


@pytest.fixture
def engine(config: TemporalComplianceConfig) -> TemporalComplianceEngine:
    return TemporalComplianceEngine(config, "test-run")


def _opened(events) -> list[dict[str, object]]:
    return [
        event.to_dict() for event in events if event.event_type == "incident_opened"
    ]


def _closed(events) -> list[dict[str, object]]:
    return [
        event.to_dict() for event in events if event.event_type == "incident_closed"
    ]


def test_single_and_short_violation_are_suspected_only(
    engine: TemporalComplianceEngine,
) -> None:
    first = engine.update(_record(0, 0.0, helmet="violation"))
    second = engine.update(_record(1, 1.49, helmet="violation"))
    assert first.record.helmet.temporal_state == "suspected_violation"
    assert second.record.helmet.temporal_state == "suspected_violation"
    assert not first.events and not second.events


def test_confirmation_opens_once_and_sustained_violation_deduplicates(
    engine: TemporalComplianceEngine,
) -> None:
    engine.update(_record(0, 0.0, helmet="violation"))
    opened = engine.update(_record(1, 1.5, helmet="violation"))
    continued = engine.update(_record(2, 2.0, helmet="violation"))
    assert len(_opened(opened.events)) == 1
    assert not continued.events
    assert opened.record.helmet.incident_active is True
    assert continued.record.helmet.active_incident_id == (
        opened.record.helmet.active_incident_id
    )


def test_short_unknown_pauses_but_does_not_accumulate_violation(
    engine: TemporalComplianceEngine,
) -> None:
    engine.update(_record(0, 0.0, helmet="violation"))
    before_gap = engine.update(_record(1, 0.8, helmet="violation"))
    gap = engine.update(_record(2, 1.0, helmet="unknown"))
    resumed = engine.update(_record(3, 1.2, helmet="violation"))
    confirmed = engine.update(_record(4, 1.9, helmet="violation"))
    assert before_gap.record.helmet.continuous_violation_duration == pytest.approx(0.8)
    assert gap.record.helmet.temporal_state == "suspected_violation"
    assert resumed.record.helmet.continuous_violation_duration == pytest.approx(0.8)
    assert len(_opened(confirmed.events)) == 1


def test_long_unknown_invalidates_candidate_timer(
    engine: TemporalComplianceEngine,
) -> None:
    engine.update(_record(0, 0.0, helmet="violation"))
    engine.update(_record(1, 0.8, helmet="violation"))
    engine.update(_record(2, 1.0, helmet="unknown"))
    long_gap = engine.update(_record(3, 2.0, helmet="unknown"))
    resumed = engine.update(_record(4, 2.1, helmet="violation"))
    later = engine.update(_record(5, 3.0, helmet="violation"))
    assert long_gap.record.helmet.temporal_state == "unknown"
    assert resumed.record.helmet.continuous_violation_duration == 0.0
    assert later.record.helmet.continuous_violation_duration == pytest.approx(0.9)
    assert not later.events


def test_active_incident_requires_sustained_present_to_close(
    engine: TemporalComplianceEngine,
) -> None:
    engine.update(_record(0, 0.0, helmet="violation"))
    engine.update(_record(1, 1.5, helmet="violation"))
    brief = engine.update(_record(2, 1.6, helmet="present"))
    still_brief = engine.update(_record(3, 2.0, helmet="present"))
    cleared = engine.update(_record(4, 2.6, helmet="present"))
    assert brief.record.helmet.temporal_state == "confirmed_violation"
    assert still_brief.record.helmet.incident_active is True
    assert len(_closed(cleared.events)) == 1
    assert cleared.record.helmet.temporal_state == "compliant"
    assert cleared.events[0].incident["close_reason"] == "sustained_present"


def test_conflict_never_forces_compliance_or_violation(
    engine: TemporalComplianceEngine,
) -> None:
    conflict = engine.update(_record(0, 0.0, helmet="conflict"))
    conflict_later = engine.update(_record(1, 0.6, helmet="conflict"))
    assert conflict.record.helmet.temporal_state == "conflict"
    assert conflict_later.record.helmet.temporal_state == "conflict"
    assert not conflict.events and not conflict_later.events


def test_two_people_and_two_equipment_are_independent(
    engine: TemporalComplianceEngine,
) -> None:
    engine.update(_record(0, 0.0, 1, helmet="violation", vest="violation"))
    engine.update(_record(0, 0.0, 2, helmet="present", vest="present"))
    person_one = engine.update(_record(1, 1.5, 1, helmet="violation", vest="violation"))
    person_two = engine.update(_record(1, 1.5, 2, helmet="present", vest="present"))
    assert len(_opened(person_one.events)) == 2
    assert person_one.record.helmet.active_incident_id != (
        person_one.record.vest.active_incident_id
    )
    assert person_two.record.helmet.temporal_state == "compliant"
    assert person_two.record.vest.temporal_state == "compliant"


def test_track_expiration_closes_active_incident(
    engine: TemporalComplianceEngine,
) -> None:
    engine.update(_record(0, 0.0, helmet="violation"))
    engine.update(_record(1, 1.5, helmet="violation"))
    events = engine.expire_track(1, 2.0)
    assert len(_closed(events)) == 1
    assert events[0].incident["close_reason"] == "track_expired"
    assert 1 not in engine.known_track_ids


def test_cooldown_suppresses_then_allows_new_incident(
    engine: TemporalComplianceEngine,
) -> None:
    engine.update(_record(0, 0.0, helmet="violation"))
    first = engine.update(_record(1, 1.5, helmet="violation"))
    engine.update(_record(2, 1.6, helmet="present"))
    closed = engine.update(_record(3, 2.6, helmet="present"))
    engine.update(_record(4, 2.7, helmet="violation"))
    suppressed = engine.update(_record(5, 4.2, helmet="violation"))
    reopened = engine.update(_record(6, 7.7, helmet="violation"))
    assert len(_opened(first.events)) == 1
    assert len(_closed(closed.events)) == 1
    assert not suppressed.events
    assert suppressed.record.helmet.temporal_state == "confirmed_violation"
    assert len(_opened(reopened.events)) == 1
    assert reopened.record.helmet.active_incident_id != (
        first.record.helmet.active_incident_id
    )


def test_irregular_timestamps_use_elapsed_time_and_invalid_times_reject(
    engine: TemporalComplianceEngine,
) -> None:
    times = [0.0, 0.033, 0.071, 0.119, 0.400, 0.433]
    last = None
    for frame, timestamp in enumerate(times):
        last = engine.update(_record(frame, timestamp, helmet="violation"))
    assert last is not None
    assert last.record.helmet.continuous_violation_duration == pytest.approx(0.433)
    assert not last.events
    confirmed = engine.update(_record(6, 1.51, helmet="violation"))
    assert len(_opened(confirmed.events)) == 1
    with pytest.raises(ValueError, match="increase strictly"):
        engine.update(_record(7, 1.51, helmet="violation"))

    other = TemporalComplianceEngine(engine.config, "other-run")
    with pytest.raises(ValueError, match="finite"):
        other.update(_record(0, float("nan"), helmet="violation"))


def test_temporal_and_incident_serialization(engine: TemporalComplianceEngine) -> None:
    engine.update(_record(0, 0.0, helmet="violation"))
    result = engine.update(_record(1, 1.5, helmet="violation"))
    values = result.record.to_dict()
    event = result.events[0].to_dict()
    assert values["helmet"]["temporal_state"] == "confirmed_violation"
    assert event["event_type"] == "incident_opened"
    assert event["incident"]["status"] == "active"
    assert event["incident"]["incident_id"].startswith("test-run-track-1-helmet")
    assert "zone" not in event["incident"]


def test_long_unknown_does_not_close_active_incident(
    engine: TemporalComplianceEngine,
) -> None:
    engine.update(_record(0, 0.0, helmet="violation"))
    engine.update(_record(1, 1.5, helmet="violation"))
    engine.update(_record(2, 1.6, helmet="unknown"))
    unknown = engine.update(_record(3, 2.5, helmet="unknown"))
    assert unknown.record.helmet.temporal_state == "unknown"
    assert unknown.record.helmet.incident_active is True
    assert not unknown.events


def test_finalize_closes_active_incident_with_source_reason(
    engine: TemporalComplianceEngine,
) -> None:
    engine.update(_record(0, 0.0, helmet="violation"))
    engine.update(_record(1, 1.5, helmet="violation"))
    events = engine.finalize(2.0)
    assert len(_closed(events)) == 1
    assert events[0].incident["close_reason"] == "source_ended"


def test_cli_dependencies_metadata_and_backward_compatibility() -> None:
    with pytest.raises(ValueError, match="requires tracking"):
        PipelineOptions(
            source="video.mp4",
            output_dir=Path("results/test"),
            enable_temporal_compliance=True,
            temporal_compliance_config=ROOT / "configs/compliance_temporal.yaml",
        )
    with pytest.raises(ValueError, match="requires PPE association"):
        PipelineOptions(
            source="video.mp4",
            output_dir=Path("results/test"),
            enable_tracking=True,
            tracker_config=ROOT / "configs/bytetrack_person.yaml",
            enable_temporal_compliance=True,
            temporal_compliance_config=ROOT / "configs/compliance_temporal.yaml",
        )
    detector_only = PipelineOptions(source="video.mp4", output_dir=Path("results/test"))
    assert detector_only.enable_temporal_compliance is False
    assert build_temporal_metadata(None, []) == {"enabled": False}
