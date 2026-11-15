"""Run deterministic, ground-truth-aware adapter diagnostics without YOLO inference."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from visionops.tracking.metrics import TrackingDiagnostics
from visionops.tracking.tracker import PersonByteTracker, TrackerSettings
from visionops.video.schema import VideoDetection

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/experiments/person_tracking/diagnostic_results.json"
WIDTH, HEIGHT, FPS = 320, 200, 10.0


def detection(frame: int, x: float, width: float = 40) -> VideoDetection:
    bbox = (x, 30.0, x + width, 130.0)
    return VideoDetection(
        frame_index=frame,
        timestamp_seconds=frame / FPS,
        class_id=2,
        class_name="Person",
        confidence=0.90,
        bbox_xyxy=bbox,
        bbox_normalized=(
            bbox[0] / WIDTH,
            bbox[1] / HEIGHT,
            bbox[2] / WIDTH,
            bbox[3] / HEIGHT,
        ),
        source_id="deterministic-diagnostic",
    )


def update(
    tracker: PersonByteTracker,
    diagnostics: TrackingDiagnostics,
    frame: int,
    detections: list[VideoDetection],
):
    result = tracker.update(
        frame,
        frame / FPS,
        detections,
        (HEIGHT, WIDTH),
        "deterministic-diagnostic",
    )
    diagnostics.update(result)
    return result


def smooth_scenario() -> dict[str, object]:
    tracker, metrics = PersonByteTracker(TrackerSettings()), TrackingDiagnostics()
    ids = []
    for frame in range(20):
        result = update(tracker, metrics, frame, [detection(frame, 20 + frame * 2)])
        ids.extend(track.track_id for track in result.tracks)
    return {
        "observed_ids": sorted(set(ids)),
        "one_id_preserved": len(set(ids)) == 1,
        "metrics": metrics.summary(FPS, tracker.live_track_ids),
    }


def two_person_scenario() -> dict[str, object]:
    tracker, metrics = PersonByteTracker(TrackerSettings()), TrackingDiagnostics()
    frames = []
    for frame in range(20):
        result = update(
            tracker,
            metrics,
            frame,
            [detection(frame, 20 + frame), detection(frame, 230 - frame)],
        )
        frames.append(sorted(track.track_id for track in result.tracks))
    unique = sorted({track_id for ids in frames for track_id in ids})
    return {
        "observed_ids": unique,
        "two_ids_preserved": len(unique) == 2 and all(len(ids) == 2 for ids in frames),
        "metrics": metrics.summary(FPS, tracker.live_track_ids),
    }


def disappearance_scenarios() -> dict[str, object]:
    short_tracker = PersonByteTracker(TrackerSettings(track_buffer=3))
    short_metrics = TrackingDiagnostics()
    initial = None
    for frame in range(3):
        result = update(short_tracker, short_metrics, frame, [detection(frame, 40)])
        initial = result.tracks[0].track_id
    update(short_tracker, short_metrics, 3, [])
    update(short_tracker, short_metrics, 4, [])
    recovered = update(short_tracker, short_metrics, 5, [detection(5, 42)])

    long_tracker = PersonByteTracker(TrackerSettings(track_buffer=2))
    long_metrics = TrackingDiagnostics()
    old_id = (
        update(long_tracker, long_metrics, 0, [detection(0, 40)]).tracks[0].track_id
    )
    update(long_tracker, long_metrics, 1, [detection(1, 41)])
    for frame in range(2, 6):
        update(long_tracker, long_metrics, frame, [])
    update(long_tracker, long_metrics, 6, [detection(6, 41)])
    new_track = update(long_tracker, long_metrics, 7, [detection(7, 42)])
    new_id = new_track.tracks[0].track_id
    return {
        "short_gap": {
            "initial_id": initial,
            "reappeared_id": recovered.tracks[0].track_id,
            "state": recovered.tracks[0].track_state,
            "same_id_recovered": recovered.tracks[0].track_id == initial,
            "metrics": short_metrics.summary(FPS, short_tracker.live_track_ids),
        },
        "beyond_buffer": {
            "old_id": old_id,
            "new_id": new_id,
            "new_id_created": new_id != old_id,
            "metrics": long_metrics.summary(FPS, long_tracker.live_track_ids),
        },
    }


def no_detection_scenario() -> dict[str, object]:
    tracker, metrics = PersonByteTracker(TrackerSettings()), TrackingDiagnostics()
    for frame in range(10):
        update(tracker, metrics, frame, [])
    return metrics.summary(FPS, tracker.live_track_ids)


def crossing_occlusion_scenario() -> dict[str, object]:
    tracker, metrics = PersonByteTracker(TrackerSettings()), TrackingDiagnostics()
    observations = []
    trajectory_a = {}
    trajectory_b = {}
    last_frame = 24
    for frame in range(last_frame + 1):
        x_a = 20 + frame * 8
        x_b = 220 - frame * 8
        trajectory_a[frame] = x_a + 20
        trajectory_b[frame] = x_b + 20
        if frame == 12:
            boxes = []
        elif frame in {11, 13}:
            boxes = [detection(frame, (x_a + x_b) / 2)]
        else:
            boxes = [detection(frame, x_a), detection(frame, x_b)]
        result = update(tracker, metrics, frame, boxes)
        observations.append(
            {
                "frame": frame,
                "tracks": [
                    {
                        "track_id": track.track_id,
                        "center_x": (track.bbox_xyxy[0] + track.bbox_xyxy[2]) / 2,
                        "state": track.track_state,
                    }
                    for track in result.tracks
                ],
                "events": {
                    "lost": list(result.events.lost_ids),
                    "recovered": list(result.events.recovered_ids),
                    "expired": list(result.events.expired_ids),
                },
            }
        )

    def nearest_id(frame: int, expected_center: float) -> int | None:
        tracks = observations[frame]["tracks"]
        if not tracks:
            return None
        return min(tracks, key=lambda item: abs(item["center_x"] - expected_center))[
            "track_id"
        ]

    initial_a = nearest_id(0, trajectory_a[0])
    initial_b = nearest_id(0, trajectory_b[0])
    final_a = nearest_id(last_frame, trajectory_a[last_frame])
    final_b = nearest_id(last_frame, trajectory_b[last_frame])
    return {
        "initial_trajectory_ids": {"a": initial_a, "b": initial_b},
        "final_trajectory_ids": {"a": final_a, "b": final_b},
        "ids_preserved": initial_a == final_a and initial_b == final_b,
        "ids_switched": initial_a == final_b and initial_b == final_a,
        "observations": observations,
        "metrics": metrics.summary(FPS, tracker.live_track_ids),
    }


def main() -> None:
    values = {
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "deterministic adapter diagnostics; not formal MOT evaluation",
        "tracker": PersonByteTracker(TrackerSettings()).metadata(),
        "smooth_motion": smooth_scenario(),
        "two_independent": two_person_scenario(),
        "disappearance": disappearance_scenarios(),
        "no_detections": no_detection_scenario(),
        "crossing_with_occlusion": crossing_occlusion_scenario(),
    }
    OUTPUT.write_text(json.dumps(values, indent=2), encoding="utf-8")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
