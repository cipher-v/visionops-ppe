from pathlib import Path

import pytest

from visionops.association.schema import EquipmentEvidence, PersonPPEFrame
from visionops.compliance.config import TemporalComplianceConfig
from visionops.compliance.engine import TemporalComplianceEngine
from visionops.video.pipeline import PipelineOptions
from visionops.zones.config import MembershipConfig, ZoneConfig, ZoneDefinition
from visionops.zones.engine import ZoneEngine
from visionops.zones.geometry import bottom_center, point_in_polygon

ROOT = Path(__file__).parents[1]


def _zone(
    zone_id: str = "work",
    polygon=((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)),
    required=frozenset({"helmet", "vest"}),
) -> ZoneDefinition:
    return ZoneDefinition(zone_id, zone_id, tuple(polygon), required, True)


def _config(
    *zones: ZoneDefinition,
    enter: float = 0.35,
    exit: float = 0.35,
) -> ZoneConfig:
    return ZoneConfig(
        1,
        MembershipConfig("bottom_center", enter, exit),
        tuple(zones or (_zone(),)),
    )


def _membership(
    engine: ZoneEngine,
    frame: int,
    timestamp: float,
    track_id: int,
    anchor: tuple[float, float],
):
    x, y = anchor
    return engine.update(
        frame_index=frame,
        timestamp_seconds=timestamp,
        source_id="synthetic",
        track_id=track_id,
        person_bbox_normalized=(x - 0.05, y - 0.2, x + 0.05, y),
    )


def _evidence(state: str) -> EquipmentEvidence:
    return EquipmentEvidence(state, None, None, (), ())


def _association(
    frame: int,
    timestamp: float,
    track_id: int,
    anchor: tuple[float, float],
    helmet: str = "unknown",
    vest: str = "unknown",
) -> PersonPPEFrame:
    x, y = anchor
    normalized = (x - 0.05, y - 0.2, x + 0.05, y)
    return PersonPPEFrame(
        frame,
        timestamp,
        track_id,
        tuple(value * 1000 for value in normalized),
        normalized,
        "synthetic",
        _evidence(helmet),
        _evidence(vest),
    )


def _policy_update(
    zones: ZoneEngine,
    temporal: TemporalComplianceEngine,
    record: PersonPPEFrame,
):
    membership = zones.update(
        frame_index=record.frame_index,
        timestamp_seconds=record.timestamp_seconds,
        source_id=record.source_id,
        track_id=record.track_id,
        person_bbox_normalized=record.person_bbox_normalized,
    )
    result = temporal.update(
        record,
        required_equipment=membership.required_equipment,
        zone_ids=membership.stable_zone_ids,
    )
    zones.diagnostics.observe_policy(record, membership, result)
    return membership, result


@pytest.fixture
def temporal() -> TemporalComplianceEngine:
    config = TemporalComplianceConfig.from_yaml(
        ROOT / "configs/compliance_temporal.yaml"
    )
    return TemporalComplianceEngine(config, "zone-test")


def test_anchor_and_boundary_inclusive_point_in_polygon() -> None:
    polygon = _zone().polygon
    assert bottom_center((0.1, 0.1, 0.3, 0.7)) == pytest.approx((0.2, 0.7))
    assert point_in_polygon((0.5, 0.5), polygon)
    assert point_in_polygon((0.2, 0.5), polygon)
    assert not point_in_polygon((0.1, 0.5), polygon)


def test_bbox_overlap_does_not_override_outside_anchor() -> None:
    engine = ZoneEngine(_config(enter=0))
    record = engine.update(
        frame_index=0,
        timestamp_seconds=0,
        source_id="synthetic",
        track_id=1,
        person_bbox_normalized=(0.3, 0.3, 0.7, 0.9),
    )
    assert record.anchor_normalized == pytest.approx((0.5, 0.9))
    assert record.stable_zone_ids == ()


def test_entry_and_exit_are_timestamp_debounced() -> None:
    engine = ZoneEngine(_config())
    assert not _membership(engine, 0, 0.0, 1, (0.1, 0.5)).stable_zone_ids
    assert not _membership(engine, 1, 1.0, 1, (0.5, 0.5)).stable_zone_ids
    assert not _membership(engine, 2, 1.2, 1, (0.1, 0.5)).stable_zone_ids
    assert not _membership(engine, 3, 2.0, 1, (0.5, 0.5)).stable_zone_ids
    entered = _membership(engine, 4, 2.35, 1, (0.5, 0.5))
    assert entered.stable_zone_ids == ("work",)
    jitter = _membership(engine, 5, 2.5, 1, (0.1, 0.5))
    assert jitter.stable_zone_ids == ("work",)
    _membership(engine, 6, 2.6, 1, (0.5, 0.5))
    _membership(engine, 7, 3.0, 1, (0.1, 0.5))
    exited = _membership(engine, 8, 3.35, 1, (0.1, 0.5))
    assert exited.stable_zone_ids == ()


def test_overlapping_zones_union_requirements() -> None:
    helmet = _zone("helmet", required=frozenset({"helmet"}))
    vest = _zone(
        "vest",
        polygon=((0.4, 0.3), (0.9, 0.3), (0.9, 0.9), (0.4, 0.9)),
        required=frozenset({"vest"}),
    )
    engine = ZoneEngine(_config(helmet, vest, enter=0, exit=0))
    record = _membership(engine, 0, 0, 1, (0.5, 0.5))
    assert record.stable_zone_ids == ("helmet", "vest")
    assert record.required_equipment == frozenset({"helmet", "vest"})


@pytest.mark.parametrize(
    "zone,match",
    [
        (_zone(polygon=((0.1, 0.1), (0.2, 0.2))), "three vertices"),
        (
            _zone(polygon=((0.1, 0.1), (1.2, 0.1), (0.2, 0.2))),
            "normalized",
        ),
        (
            _zone(polygon=((0.1, 0.1), (0.5, 0.5), (0.9, 0.9))),
            "zero area",
        ),
        (
            _zone(polygon=((0.1, 0.1), (0.9, 0.9), (0.1, 0.9), (0.9, 0.1))),
            "self-intersects",
        ),
        (_zone(required=frozenset({"boots"})), "unknown PPE"),
    ],
)
def test_invalid_zone_definitions_reject(zone: ZoneDefinition, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _config(zone).validate()


def test_duplicate_ids_and_example_config() -> None:
    with pytest.raises(ValueError, match="unique"):
        _config(_zone("same"), _zone("same")).validate()
    loaded = ZoneConfig.from_yaml(ROOT / "configs/zones/example_zones.yaml")
    assert loaded.membership.anchor == "bottom_center"
    assert loaded.enabled_zones[0].required_equipment == frozenset({"helmet", "vest"})


def test_malformed_yaml_types_reject_cleanly(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text(
        """version: 1
membership:
  anchor: bottom_center
  enter_confirm_seconds: invalid
  exit_confirm_seconds: 0.35
zones: []
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Malformed zone configuration"):
        ZoneConfig.from_yaml(malformed)


def test_outside_violation_is_suppressed_and_timer_starts_at_entry(
    temporal: TemporalComplianceEngine,
) -> None:
    zones = ZoneEngine(_config(enter=0.35, exit=0.35))
    _, outside = _policy_update(
        zones, temporal, _association(0, 0, 1, (0.1, 0.5), helmet="violation")
    )
    _policy_update(
        zones, temporal, _association(1, 5, 1, (0.1, 0.5), helmet="violation")
    )
    _policy_update(
        zones, temporal, _association(2, 10, 1, (0.5, 0.5), helmet="violation")
    )
    membership, entered = _policy_update(
        zones, temporal, _association(3, 10.4, 1, (0.5, 0.5), helmet="violation")
    )
    assert outside.record.helmet.temporal_state == "not_required"
    assert membership.required_equipment == frozenset({"helmet", "vest"})
    assert entered.record.helmet.continuous_violation_duration == 0
    assert not entered.events
    _, confirmed = _policy_update(
        zones, temporal, _association(4, 11.9, 1, (0.5, 0.5), helmet="violation")
    )
    assert [event.event_type for event in confirmed.events] == ["incident_opened"]
    assert confirmed.events[0].incident["started_at_seconds"] == pytest.approx(10.4)
    assert zones.diagnostics.violations_outside_required_suppressed == 3


def test_zone_exit_closes_incident_after_debounce(
    temporal: TemporalComplianceEngine,
) -> None:
    zones = ZoneEngine(_config(enter=0, exit=0.35))
    _policy_update(
        zones, temporal, _association(0, 0, 1, (0.5, 0.5), helmet="violation")
    )
    _, opened = _policy_update(
        zones, temporal, _association(1, 1.5, 1, (0.5, 0.5), helmet="violation")
    )
    assert opened.events[0].event_type == "incident_opened"
    _, jitter = _policy_update(
        zones, temporal, _association(2, 2.0, 1, (0.1, 0.5), helmet="violation")
    )
    assert not jitter.events
    _, closed = _policy_update(
        zones, temporal, _association(3, 2.35, 1, (0.1, 0.5), helmet="violation")
    )
    assert closed.events[0].incident["close_reason"] == "zone_exited"
    assert closed.record.helmet.temporal_state == "not_required"


def test_required_zone_transition_retains_incident(
    temporal: TemporalComplianceEngine,
) -> None:
    zone_a = _zone(
        "a",
        polygon=((0, 0.2), (0.6, 0.2), (0.6, 0.8), (0, 0.8)),
        required=frozenset({"helmet"}),
    )
    zone_b = _zone(
        "b",
        polygon=((0.4, 0.2), (1, 0.2), (1, 0.8), (0.4, 0.8)),
        required=frozenset({"helmet"}),
    )
    zones = ZoneEngine(_config(zone_a, zone_b, enter=0, exit=0))
    _policy_update(
        zones, temporal, _association(0, 0, 1, (0.3, 0.5), helmet="violation")
    )
    _, opened = _policy_update(
        zones, temporal, _association(1, 1.5, 1, (0.5, 0.5), helmet="violation")
    )
    incident_id = opened.record.helmet.active_incident_id
    membership, continued = _policy_update(
        zones, temporal, _association(2, 2.0, 1, (0.7, 0.5), helmet="violation")
    )
    assert membership.stable_zone_ids == ("b",)
    assert continued.record.helmet.active_incident_id == incident_id
    assert not continued.events


def test_helmet_only_zone_makes_vest_not_required(
    temporal: TemporalComplianceEngine,
) -> None:
    zones = ZoneEngine(_config(_zone(required=frozenset({"helmet"})), enter=0, exit=0))
    _, result = _policy_update(
        zones,
        temporal,
        _association(
            0,
            0,
            1,
            (0.5, 0.5),
            helmet="present",
            vest="violation",
        ),
    )
    assert result.record.helmet.temporal_state == "compliant"
    assert result.record.vest.temporal_state == "not_required"


def test_people_have_independent_zone_policy_and_serialization() -> None:
    helmet = _zone(
        "helmet",
        polygon=((0, 0), (0.49, 0), (0.49, 1), (0, 1)),
        required=frozenset({"helmet"}),
    )
    vest = _zone(
        "vest",
        polygon=((0.51, 0), (1, 0), (1, 1), (0.51, 1)),
        required=frozenset({"vest"}),
    )
    zones = ZoneEngine(_config(helmet, vest, enter=0, exit=0))
    left = _membership(zones, 0, 0, 1, (0.25, 0.5))
    right = _membership(zones, 0, 0, 2, (0.75, 0.5))
    assert left.required_equipment == frozenset({"helmet"})
    assert right.required_equipment == frozenset({"vest"})
    assert left.to_dict()["anchor_normalized"] == [0.25, 0.5]
    assert left.to_dict()["required_equipment"] == ["helmet"]


def test_zone_cli_dependencies_and_disabled_backward_compatibility() -> None:
    zone_path = ROOT / "configs/zones/example_zones.yaml"
    with pytest.raises(ValueError, match="tracking"):
        PipelineOptions(
            source="video.mp4",
            output_dir=Path("results/test"),
            zones_config=zone_path,
        )
    with pytest.raises(ValueError, match="PPE association"):
        PipelineOptions(
            source="video.mp4",
            output_dir=Path("results/test"),
            enable_tracking=True,
            tracker_config=ROOT / "configs/production_tracker.yaml",
            zones_config=zone_path,
        )
    with pytest.raises(ValueError, match="temporal compliance"):
        PipelineOptions(
            source="video.mp4",
            output_dir=Path("results/test"),
            enable_tracking=True,
            tracker_config=ROOT / "configs/production_tracker.yaml",
            enable_ppe_association=True,
            ppe_association_config=ROOT / "configs/ppe_association.yaml",
            zones_config=zone_path,
        )
    disabled = PipelineOptions(
        source="video.mp4",
        output_dir=Path("results/test"),
    )
    assert disabled.zones_config is None
