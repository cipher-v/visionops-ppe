"""Ground-truth-free diagnostics for emitted Person tracks."""

from __future__ import annotations

import statistics
from collections import defaultdict

from visionops.tracking.schema import TrackerFrameResult


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


class TrackingDiagnostics:
    """Accumulate observable track lifecycles without claiming MOT accuracy."""

    def __init__(self) -> None:
        self._frames_by_track: dict[int, list[int]] = defaultdict(list)
        self._timestamps_by_track: dict[int, list[float]] = defaultdict(list)
        self.frames_with_active_tracks = 0
        self.maximum_simultaneous_active_tracks = 0
        self.lost_transitions = 0
        self.recovered_transitions = 0
        self.expired_tracks = 0
        self.rejected_person_detections = 0

    def update(self, result: TrackerFrameResult) -> None:
        if result.tracks:
            self.frames_with_active_tracks += 1
        self.maximum_simultaneous_active_tracks = max(
            self.maximum_simultaneous_active_tracks, len(result.tracks)
        )
        self.rejected_person_detections += result.rejected_person_detections
        self.lost_transitions += len(result.events.lost_ids)
        self.recovered_transitions += len(result.events.recovered_ids)
        self.expired_tracks += len(result.events.expired_ids)
        for track in result.tracks:
            self._frames_by_track[track.track_id].append(track.frame_index)
            self._timestamps_by_track[track.track_id].append(track.timestamp_seconds)

    def summary(
        self, reported_fps: float | None, live_track_ids: tuple[int, ...]
    ) -> dict[str, object]:
        lengths = [len(frames) for frames in self._frames_by_track.values()]
        durations = []
        for track_id, timestamps in self._timestamps_by_track.items():
            frame_duration = 1 / reported_fps if reported_fps else 0.0
            durations.append(timestamps[-1] - timestamps[0] + frame_duration)
        return {
            "unique_tracks_created": len(lengths),
            "track_observations": sum(lengths),
            "mean_track_length_frames": statistics.fmean(lengths) if lengths else 0.0,
            "median_track_length_frames": statistics.median(lengths)
            if lengths
            else 0.0,
            "p95_track_length_frames": _percentile(lengths, 0.95) if lengths else 0.0,
            "singleton_tracks": sum(length == 1 for length in lengths),
            "tracks_lasting_at_least_one_second": sum(
                duration >= 1.0 for duration in durations
            ),
            "frames_with_active_tracks": self.frames_with_active_tracks,
            "maximum_simultaneous_active_tracks": (
                self.maximum_simultaneous_active_tracks
            ),
            "lost_transitions": self.lost_transitions,
            "recovered_transitions": self.recovered_transitions,
            "expired_tracks": self.expired_tracks,
            "rejected_person_detections": self.rejected_person_detections,
            "live_track_ids_at_end": list(live_track_ids),
        }
