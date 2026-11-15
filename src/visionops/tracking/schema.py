"""Structured person-track records and per-frame tracker results."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PersonTrack:
    """One active Person-track observation local to a single source run."""

    frame_index: int
    timestamp_seconds: float
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_normalized: tuple[float, float, float, float]
    source_id: str
    track_state: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation without compliance fields."""
        values = asdict(self)
        values["bbox_xyxy"] = list(self.bbox_xyxy)
        values["bbox_normalized"] = list(self.bbox_normalized)
        return values


@dataclass(frozen=True)
class TrackerEvents:
    """Lifecycle transitions reliably derived from ByteTrack state."""

    created_ids: tuple[int, ...] = ()
    lost_ids: tuple[int, ...] = ()
    recovered_ids: tuple[int, ...] = ()
    expired_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TrackerFrameResult:
    """Active observations and lifecycle changes for one processed frame."""

    tracks: tuple[PersonTrack, ...]
    matched_person_indices: frozenset[int]
    rejected_person_detections: int
    events: TrackerEvents
