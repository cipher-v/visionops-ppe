from pathlib import Path

import pytest

from visionops.association.associator import PPEAssociator
from visionops.association.config import AssociationConfig
from visionops.association.geometry import (
    anatomical_region,
    intersection_over_area,
)
from visionops.association.metrics import AssociationDiagnostics
from visionops.tracking.schema import PersonTrack
from visionops.video.pipeline import (
    PipelineOptions,
    build_association_metadata,
)
from visionops.video.schema import VideoDetection

ROOT = Path(__file__).parents[1]
NAMES = {0: "helmet", 1: "vest", 2: "Person", 3: "no_helmet", 4: "no_vest"}


@pytest.fixture
def config() -> AssociationConfig:
    return AssociationConfig.from_yaml(ROOT / "configs/ppe_association.yaml")


@pytest.fixture
def associator(config: AssociationConfig) -> PPEAssociator:
    return PPEAssociator(config)


def _track(track_id: int, box=(100.0, 50.0, 300.0, 450.0)) -> PersonTrack:
    return PersonTrack(
        frame_index=0,
        timestamp_seconds=0.0,
        track_id=track_id,
        class_id=2,
        class_name="Person",
        confidence=0.95,
        bbox_xyxy=box,
        bbox_normalized=tuple(value / 500 for value in box),
        source_id="synthetic",
        track_state="tracked",
    )


def _detection(
    class_id: int,
    box: tuple[float, float, float, float],
    confidence: float = 0.9,
) -> VideoDetection:
    return VideoDetection(
        frame_index=0,
        timestamp_seconds=0.0,
        class_id=class_id,
        class_name=NAMES[class_id],
        confidence=confidence,
        bbox_xyxy=box,
        bbox_normalized=tuple(value / 500 for value in box),
        source_id="synthetic",
    )


def test_head_and_torso_region_generation(config: AssociationConfig) -> None:
    person = (100.0, 50.0, 300.0, 450.0)
    assert anatomical_region(person, config.head) == (125.0, 50.0, 275.0, 190.0)
    assert anatomical_region(person, config.torso) == (115.0, 130.0, 285.0, 338.0)


def test_intersection_over_area_uses_ppe_as_subject() -> None:
    ppe = (0.0, 0.0, 20.0, 20.0)
    assert intersection_over_area(ppe, (0.0, 0.0, 10.0, 20.0)) == 0.5
    assert intersection_over_area(ppe, (30.0, 30.0, 40.0, 40.0)) == 0.0


def test_head_helmet_and_torso_vest_assign_with_documented_score(
    associator: PPEAssociator,
) -> None:
    detections = [
        _detection(0, (170.0, 70.0, 230.0, 120.0)),
        _detection(1, (150.0, 180.0, 250.0, 300.0)),
    ]
    result = associator.associate((_track(1),), detections)
    record = result.records[0]
    assert record.helmet.state == "present"
    assert record.vest.state == "present"
    match = record.helmet.positive_detection
    assert match is not None
    expected = (
        0.30 * float(match.center_inside_region)
        + 0.20 * match.person_ioa
        + 0.30 * match.region_ioa
        + 0.20 * max(0.0, 1.0 - match.normalized_center_distance)
    )
    assert match.association_score == pytest.approx(expected)


def test_outside_person_is_gated_and_unknown(
    associator: PPEAssociator,
) -> None:
    result = associator.associate(
        (_track(1),), [_detection(0, (350.0, 20.0, 390.0, 60.0))]
    )
    assert result.records[0].helmet.state == "unknown"
    assert result.unassigned_detection_indices == {0}


def test_one_ambiguous_detection_has_one_deterministic_owner(
    associator: PPEAssociator,
) -> None:
    tracks = (_track(9), _track(3))
    result = associator.associate(tracks, [_detection(0, (170.0, 70.0, 230.0, 120.0))])
    owners = [
        record.track_id
        for record in result.records
        if record.helmet.positive_detection is not None
    ]
    assert owners == [3]
    match = next(
        record.helmet.positive_detection
        for record in result.records
        if record.helmet.positive_detection is not None
    )
    assert match.ambiguous is True
    assert match.candidate_person_count == 2
    assert match.alternative_track_ids == (9,)
    assert result.ambiguous_detections == 1
    assert len(result.assigned_detection_indices) == 1


def test_two_people_receive_distinct_ppe(associator: PPEAssociator) -> None:
    tracks = (
        _track(1, (0.0, 20.0, 120.0, 420.0)),
        _track(2, (180.0, 20.0, 300.0, 420.0)),
    )
    detections = [
        _detection(0, (35.0, 40.0, 85.0, 90.0)),
        _detection(1, (205.0, 150.0, 275.0, 280.0)),
    ]
    result = associator.associate(tracks, detections)
    by_id = {record.track_id: record for record in result.records}
    assert by_id[1].helmet.state == "present"
    assert by_id[1].vest.state == "unknown"
    assert by_id[2].helmet.state == "unknown"
    assert by_id[2].vest.state == "present"


def test_positive_and_negative_evidence_conflict(
    associator: PPEAssociator,
) -> None:
    result = associator.associate(
        (_track(1),),
        [
            _detection(0, (170.0, 70.0, 230.0, 120.0)),
            _detection(3, (175.0, 75.0, 225.0, 125.0)),
        ],
    )
    helmet = result.records[0].helmet
    assert helmet.state == "conflict"
    assert helmet.positive_detection is not None
    assert helmet.negative_detection is not None


def test_no_ppe_means_unknown_not_violation(associator: PPEAssociator) -> None:
    record = associator.associate((_track(1),), []).records[0]
    assert record.helmet.state == "unknown"
    assert record.vest.state == "unknown"


def test_malformed_zero_area_ppe_is_rejected(
    associator: PPEAssociator,
) -> None:
    result = associator.associate(
        (_track(1),), [_detection(0, (180.0, 90.0, 180.0, 110.0))]
    )
    assert result.malformed_ppe_detections == 1
    assert result.unassigned_detection_indices == {0}
    assert result.records[0].helmet.state == "unknown"


def test_same_class_competition_retains_diagnostics_and_primary(
    associator: PPEAssociator,
) -> None:
    result = associator.associate(
        (_track(1),),
        [
            _detection(0, (170.0, 70.0, 230.0, 120.0), 0.80),
            _detection(0, (175.0, 75.0, 225.0, 115.0), 0.95),
        ],
    )
    evidence = result.records[0].helmet
    assert len(evidence.positive_candidates) == 2
    assert sum(item.primary_for_class for item in evidence.positive_candidates) == 1
    assert evidence.positive_detection == evidence.positive_candidates[0]


def test_serialization_and_metrics(associator: PPEAssociator) -> None:
    result = associator.associate(
        (_track(1),), [_detection(3, (170.0, 70.0, 230.0, 120.0))]
    )
    values = result.records[0].to_dict()
    assert values["helmet"]["state"] == "violation"
    assert values["helmet"]["negative_detection"]["ppe_class_name"] == "no_helmet"
    assert "incident_id" not in values
    diagnostics = AssociationDiagnostics()
    diagnostics.update(result)
    summary = diagnostics.summary()
    assert summary["total_person_observations"] == 1
    assert summary["ppe_detections_assigned"] == 1
    assert summary["frame_level_equipment_states"]["vest"]["unknown"] == 1


def test_cli_dependency_and_backward_compatibility() -> None:
    with pytest.raises(ValueError, match="requires tracking"):
        PipelineOptions(
            source="video.mp4",
            output_dir=Path("results/test"),
            enable_ppe_association=True,
            ppe_association_config=ROOT / "configs/ppe_association.yaml",
        )
    detector_only = PipelineOptions(source="video.mp4", output_dir=Path("results/test"))
    assert detector_only.enable_tracking is False
    assert detector_only.enable_ppe_association is False
    assert build_association_metadata(None, None, []) == {"enabled": False}
