"""Temporal compliance, incident, and event schemas."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from visionops.association.schema import EquipmentState

TemporalState = Literal[
    "unknown",
    "compliant",
    "suspected_violation",
    "confirmed_violation",
    "conflict",
]


@dataclass
class Incident:
    incident_id: str
    source_id: str
    track_id: int
    equipment: str
    type: str
    started_at_seconds: float
    confirmed_at_seconds: float
    cleared_at_seconds: float | None
    duration_seconds: float | None
    status: str
    close_reason: str | None
    evidence_summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class IncidentEvent:
    event_type: str
    event_timestamp_seconds: float
    incident: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "event_type": self.event_type,
            "event_timestamp_seconds": self.event_timestamp_seconds,
            "incident": self.incident,
        }


@dataclass(frozen=True)
class EquipmentTemporalSnapshot:
    equipment: str
    evidence_state: EquipmentState
    temporal_state: TemporalState
    first_observed_timestamp: float
    last_evidence_timestamp: float | None
    continuous_violation_duration: float
    continuous_present_duration: float
    unknown_duration: float
    conflict_duration: float
    last_stable_state: str
    incident_active: bool
    active_incident_id: str | None
    incident_duration_seconds: float | None
    cooldown_until_seconds: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalComplianceRecord:
    frame_index: int
    timestamp_seconds: float
    source_id: str
    track_id: int
    person_bbox_xyxy: tuple[float, float, float, float]
    helmet: EquipmentTemporalSnapshot
    vest: EquipmentTemporalSnapshot

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "source_id": self.source_id,
            "track_id": self.track_id,
            "person_bbox_xyxy": list(self.person_bbox_xyxy),
            "helmet": self.helmet.to_dict(),
            "vest": self.vest.to_dict(),
        }


@dataclass(frozen=True)
class TemporalUpdateResult:
    record: TemporalComplianceRecord
    events: tuple[IncidentEvent, ...]
