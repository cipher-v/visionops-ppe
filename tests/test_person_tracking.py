from pathlib import Path

import pytest

from visionops.tracking.metrics import TrackingDiagnostics
from visionops.tracking.schema import PersonTrack
from visionops.tracking.tracker import PersonByteTracker, TrackerSettings
from visionops.video.pipeline import PipelineOptions, build_tracking_metadata
from visionops.video.schema import VideoDetection

ROOT = Path(__file__).parents[1]
SHAPE = (200, 320)


def _detection(
    frame_index: int,
    x1: float,
    y1: float = 30,
    width: float = 40,
    height: float = 100,
    class_id: int = 2,
    confidence: float = 0.9,
) -> VideoDetection:
    x2, y2 = x1 + width, y1 + height
    names = {0: "helmet", 1: "vest", 2: "Person", 3: "no_helmet", 4: "no_vest"}
    return VideoDetection(
        frame_index=frame_index,
        timestamp_seconds=frame_index / 10,
        class_id=class_id,
        class_name=names[class_id],
        confidence=confidence,
        bbox_xyxy=(x1, y1, x2, y2),
        bbox_normalized=(x1 / 320, y1 / 200, x2 / 320, y2 / 200),
        source_id="synthetic",
    )


def _tracker(buffer: int = 3) -> PersonByteTracker:
    return PersonByteTracker(TrackerSettings(track_buffer=buffer))


def _update(
    tracker: PersonByteTracker,
    frame_index: int,
    detections: list[VideoDetection],
):
    return tracker.update(frame_index, frame_index / 10, detections, SHAPE, "synthetic")


def test_tracker_initialization_and_configuration() -> None:
    settings = TrackerSettings.from_yaml(ROOT / "configs/bytetrack_person.yaml")
    tracker = PersonByteTracker(settings)
    metadata = tracker.metadata()
    assert metadata["name"] == "Ultralytics ByteTrack"
    assert metadata["appearance_reid"] is False
    assert metadata["configuration"]["track_buffer"] == 30
    assert metadata["person_confidence_threshold"] == 0.41


def test_person_only_and_frozen_threshold_filtering() -> None:
    tracker = _tracker()
    result = _update(
        tracker,
        0,
        [
            _detection(0, 10, class_id=0),
            _detection(0, 60, confidence=0.4099),
            _detection(0, 110, confidence=0.41),
        ],
    )
    assert len(result.tracks) == 1
    assert result.tracks[0].class_name == "Person"
    assert result.rejected_person_detections == 1


def test_smooth_trajectory_retains_one_track_id() -> None:
    tracker = _tracker()
    ids = []
    for frame_index in range(12):
        result = _update(
            tracker, frame_index, [_detection(frame_index, 20 + frame_index * 2)]
        )
        ids.extend(track.track_id for track in result.tracks)
    assert len(ids) == 12
    assert len(set(ids)) == 1


def test_two_people_keep_distinct_ids() -> None:
    tracker = _tracker()
    observed = []
    for frame_index in range(10):
        result = _update(
            tracker,
            frame_index,
            [
                _detection(frame_index, 20 + frame_index),
                _detection(frame_index, 220 - frame_index),
            ],
        )
        observed.append({track.track_id for track in result.tracks})
    assert all(len(ids) == 2 for ids in observed)
    assert len(set.union(*observed)) == 2


def test_short_disappearance_recovers_same_track() -> None:
    tracker = _tracker(buffer=3)
    initial_id = None
    for frame_index in range(3):
        result = _update(tracker, frame_index, [_detection(frame_index, 30)])
        initial_id = result.tracks[0].track_id
    lost = _update(tracker, 3, [])
    _update(tracker, 4, [])
    recovered = _update(tracker, 5, [_detection(5, 32)])
    assert initial_id is not None
    assert lost.events.lost_ids == (initial_id,)
    assert recovered.tracks[0].track_id == initial_id
    assert recovered.tracks[0].track_state == "recovered"


def test_expired_track_is_not_reused_on_reappearance() -> None:
    tracker = _tracker(buffer=2)
    first = _update(tracker, 0, [_detection(0, 30)]).tracks[0].track_id
    _update(tracker, 1, [_detection(1, 31)])
    for frame_index in range(2, 6):
        _update(tracker, frame_index, [])
    _update(tracker, 6, [_detection(6, 31)])
    reappeared = _update(tracker, 7, [_detection(7, 32)])
    assert reappeared.tracks
    assert reappeared.tracks[0].track_id != first


def test_empty_and_invalid_detections_do_not_create_tracks() -> None:
    tracker = _tracker()
    assert not _update(tracker, 0, []).tracks
    zero_area = _detection(1, 20, width=0)
    result = _update(tracker, 1, [zero_area])
    assert not result.tracks
    assert result.rejected_person_detections == 1
    with pytest.raises(ValueError, match="increase strictly"):
        _update(tracker, 1, [])


def test_person_track_serialization() -> None:
    track = PersonTrack(
        frame_index=4,
        timestamp_seconds=0.4,
        track_id=7,
        class_id=2,
        class_name="Person",
        confidence=0.94,
        bbox_xyxy=(10.0, 20.0, 50.0, 120.0),
        bbox_normalized=(0.1, 0.1, 0.5, 0.6),
        source_id="camera-1",
        track_state="tracked",
    )
    values = track.to_dict()
    assert values["track_id"] == 7
    assert values["bbox_xyxy"] == [10.0, 20.0, 50.0, 120.0]
    assert "helmet" not in values
    assert "compliance" not in values


def test_tracking_disabled_backward_compatibility_and_metadata() -> None:
    options = PipelineOptions(source="video.mp4", output_dir=Path("results/test"))
    assert options.enable_tracking is False
    assert options.render is False
    assert build_tracking_metadata(None, None, 30.0, []) == {"enabled": False}

    tracker = _tracker()
    diagnostics = TrackingDiagnostics()
    result = _update(tracker, 0, [_detection(0, 20)])
    diagnostics.update(result)
    metadata = build_tracking_metadata(tracker, diagnostics, 10.0, [0.001])
    assert metadata["enabled"] is True
    assert metadata["diagnostics"]["unique_tracks_created"] == 1
    assert metadata["diagnostics"]["track_observations"] == 1
