"""Deterministic timestamp-driven PPE temporal state machine."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from visionops.association.schema import EquipmentState, PersonPPEFrame
from visionops.compliance.config import EquipmentTiming, TemporalComplianceConfig
from visionops.compliance.schema import (
    EquipmentTemporalSnapshot,
    Incident,
    IncidentEvent,
    TemporalComplianceRecord,
    TemporalState,
    TemporalUpdateResult,
)


@dataclass
class _EquipmentRuntime:
    equipment: str
    timing: EquipmentTiming
    first_observed_timestamp: float
    last_update_timestamp: float
    last_evidence_timestamp: float | None = None
    last_evidence_state: EquipmentState = "unknown"
    temporal_state: TemporalState = "unknown"
    last_stable_state: str = "unknown"
    violation_duration: float = 0.0
    present_duration: float = 0.0
    unknown_started_at: float | None = None
    conflict_started_at: float | None = None
    violation_started_at: float | None = None
    present_started_at: float | None = None
    paused_evidence_state: str | None = None
    active_incident: Incident | None = None
    cooldown_until: float | None = None
    incident_sequence: int = 0
    cooldown_suppressed: bool = False
    evidence_counts: Counter[str] = field(default_factory=Counter)
    violation_zone_ids: tuple[str, ...] = ()
    current_zone_ids: tuple[str, ...] = ()


@dataclass
class _TrackRuntime:
    source_id: str
    track_id: int
    helmet: _EquipmentRuntime
    vest: _EquipmentRuntime
    last_timestamp: float | None = None
    last_frame_index: int | None = None
    person_bbox_xyxy: tuple[float, float, float, float] | None = None


class TemporalComplianceEngine:
    """Maintain independent run-local helmet and vest state per anonymous track."""

    def __init__(self, config: TemporalComplianceConfig, run_id: str) -> None:
        config.validate()
        if not run_id.strip():
            raise ValueError("Temporal engine run ID cannot be empty")
        self.config = config
        self.run_id = run_id.strip()
        self._tracks: dict[int, _TrackRuntime] = {}
        self._opened_incidents = 0
        self._closed_incidents = 0
        self._suppressed_confirmations = 0
        self._close_reasons: Counter[str] = Counter()
        self._temporal_state_observations: Counter[str] = Counter()

    @property
    def known_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def metadata(self) -> dict[str, object]:
        return {
            "name": "VisionOps timestamp-driven temporal compliance",
            "run_id": self.run_id,
            "incident_id_semantics": "run-local track/equipment sequence",
            "configuration": self.config.to_dict(),
        }

    def summary(self) -> dict[str, object]:
        active = sum(
            runtime.active_incident is not None
            for track in self._tracks.values()
            for runtime in (track.helmet, track.vest)
        )
        return {
            "known_tracks_at_end": len(self._tracks),
            "incidents_opened": self._opened_incidents,
            "incidents_closed": self._closed_incidents,
            "incidents_active": active,
            "cooldown_confirmations_suppressed": self._suppressed_confirmations,
            "close_reasons": dict(sorted(self._close_reasons.items())),
            "temporal_state_observations": dict(
                sorted(self._temporal_state_observations.items())
            ),
        }

    def _new_equipment(self, equipment: str, timestamp: float) -> _EquipmentRuntime:
        return _EquipmentRuntime(
            equipment=equipment,
            timing=self.config.timing_for(equipment),
            first_observed_timestamp=timestamp,
            last_update_timestamp=timestamp,
        )

    def _track(self, record: PersonPPEFrame) -> _TrackRuntime:
        runtime = self._tracks.get(record.track_id)
        if runtime is None:
            runtime = _TrackRuntime(
                source_id=record.source_id,
                track_id=record.track_id,
                helmet=self._new_equipment("helmet", record.timestamp_seconds),
                vest=self._new_equipment("vest", record.timestamp_seconds),
            )
            self._tracks[record.track_id] = runtime
        elif runtime.source_id != record.source_id:
            raise ValueError("A run-local track ID cannot change source")
        return runtime

    @staticmethod
    def _validate_timestamp(timestamp: float) -> None:
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Timestamp must be finite and non-negative")

    def _validate_track_time(self, track: _TrackRuntime, timestamp: float) -> None:
        self._validate_timestamp(timestamp)
        if track.last_timestamp is not None and timestamp <= track.last_timestamp:
            raise ValueError("Per-track timestamps must increase strictly")

    @staticmethod
    def _evidence_summary(runtime: _EquipmentRuntime) -> dict[str, object]:
        return {
            "observation_counts": dict(sorted(runtime.evidence_counts.items())),
            "continuous_violation_duration": runtime.violation_duration,
            "continuous_present_duration": runtime.present_duration,
            "last_stable_state": runtime.last_stable_state,
        }

    def _incident_event(
        self, event_type: str, timestamp: float, incident: Incident
    ) -> IncidentEvent:
        return IncidentEvent(event_type, timestamp, incident.to_dict())

    def _open_incident(
        self,
        track: _TrackRuntime,
        runtime: _EquipmentRuntime,
        timestamp: float,
        zone_policy_enabled: bool,
    ) -> IncidentEvent:
        runtime.incident_sequence += 1
        incident = Incident(
            incident_id=(
                f"{self.run_id}-track-{track.track_id}-{runtime.equipment}-"
                f"{runtime.incident_sequence:04d}"
            ),
            source_id=track.source_id,
            track_id=track.track_id,
            equipment=runtime.equipment,
            type="ppe_violation",
            started_at_seconds=(
                runtime.violation_started_at
                if runtime.violation_started_at is not None
                else timestamp
            ),
            confirmed_at_seconds=timestamp,
            cleared_at_seconds=None,
            duration_seconds=None,
            status="active",
            close_reason=None,
            evidence_summary=self._evidence_summary(runtime),
            zone_ids_at_start=(
                runtime.violation_zone_ids if zone_policy_enabled else None
            ),
            zone_ids_at_confirmation=(
                runtime.current_zone_ids if zone_policy_enabled else None
            ),
            last_zone_ids=(runtime.current_zone_ids if zone_policy_enabled else None),
            zone_policy_required=True if zone_policy_enabled else None,
        )
        runtime.active_incident = incident
        runtime.cooldown_suppressed = False
        runtime.temporal_state = "confirmed_violation"
        runtime.last_stable_state = "confirmed_violation"
        self._opened_incidents += 1
        return self._incident_event("incident_opened", timestamp, incident)

    def _close_incident(
        self,
        runtime: _EquipmentRuntime,
        timestamp: float,
        reason: str,
    ) -> IncidentEvent | None:
        incident = runtime.active_incident
        if incident is None:
            return None
        incident.cleared_at_seconds = timestamp
        incident.duration_seconds = max(0.0, timestamp - incident.started_at_seconds)
        incident.status = "closed"
        incident.close_reason = reason
        if incident.zone_policy_required:
            incident.last_zone_ids = runtime.current_zone_ids
        incident.evidence_summary = self._evidence_summary(runtime)
        runtime.active_incident = None
        runtime.cooldown_until = timestamp + self.config.incident_cooldown_seconds
        runtime.cooldown_suppressed = False
        self._closed_incidents += 1
        self._close_reasons[reason] += 1
        return self._incident_event("incident_closed", timestamp, incident)

    @staticmethod
    def _gap_is_short(
        runtime: _EquipmentRuntime,
        timestamp: float,
        evidence: EquipmentState,
        unknown_grace: float,
        conflict_grace: float,
    ) -> bool:
        if runtime.paused_evidence_state != evidence:
            return False
        if runtime.last_evidence_state == "unknown":
            return (
                runtime.unknown_started_at is not None
                and timestamp - runtime.unknown_started_at <= unknown_grace
            )
        if runtime.last_evidence_state == "conflict":
            return (
                runtime.conflict_started_at is not None
                and timestamp - runtime.conflict_started_at <= conflict_grace
            )
        return False

    def _update_violation(
        self,
        track: _TrackRuntime,
        runtime: _EquipmentRuntime,
        timestamp: float,
        zone_policy_enabled: bool,
    ) -> list[IncidentEvent]:
        elapsed = timestamp - runtime.last_update_timestamp
        if runtime.last_evidence_state == "violation":
            runtime.violation_duration += elapsed
        elif not self._gap_is_short(
            runtime,
            timestamp,
            "violation",
            self.config.unknown_grace_seconds,
            self.config.conflict_grace_seconds,
        ):
            runtime.violation_duration = 0.0
            runtime.violation_started_at = timestamp
            runtime.violation_zone_ids = runtime.current_zone_ids
        runtime.present_duration = 0.0
        runtime.present_started_at = None
        runtime.unknown_started_at = None
        runtime.conflict_started_at = None
        runtime.paused_evidence_state = None

        if runtime.active_incident is not None:
            if runtime.active_incident.zone_policy_required:
                runtime.active_incident.last_zone_ids = runtime.current_zone_ids
            runtime.temporal_state = "confirmed_violation"
            runtime.last_stable_state = "confirmed_violation"
            return []

        if runtime.violation_duration < runtime.timing.violation_confirm_seconds:
            runtime.temporal_state = "suspected_violation"
            return []

        runtime.temporal_state = "confirmed_violation"
        runtime.last_stable_state = "confirmed_violation"
        if runtime.cooldown_until is not None and timestamp < runtime.cooldown_until:
            if not runtime.cooldown_suppressed:
                self._suppressed_confirmations += 1
                runtime.cooldown_suppressed = True
            return []
        return [
            self._open_incident(
                track,
                runtime,
                timestamp,
                zone_policy_enabled,
            )
        ]

    def _update_present(
        self,
        runtime: _EquipmentRuntime,
        timestamp: float,
    ) -> list[IncidentEvent]:
        elapsed = timestamp - runtime.last_update_timestamp
        if runtime.last_evidence_state == "present":
            runtime.present_duration += elapsed
        elif not self._gap_is_short(
            runtime,
            timestamp,
            "present",
            self.config.unknown_grace_seconds,
            self.config.conflict_grace_seconds,
        ):
            runtime.present_duration = 0.0
            runtime.present_started_at = timestamp
        runtime.violation_duration = 0.0
        runtime.violation_started_at = None
        runtime.unknown_started_at = None
        runtime.conflict_started_at = None
        runtime.paused_evidence_state = None
        runtime.cooldown_suppressed = False

        if runtime.active_incident is not None:
            if runtime.present_duration < runtime.timing.clear_confirm_seconds:
                runtime.temporal_state = "confirmed_violation"
                return []
            runtime.temporal_state = "compliant"
            runtime.last_stable_state = "compliant"
            event = self._close_incident(runtime, timestamp, "sustained_present")
            return [event] if event is not None else []

        runtime.temporal_state = "compliant"
        runtime.last_stable_state = "compliant"
        return []

    def _update_unknown(
        self, runtime: _EquipmentRuntime, timestamp: float
    ) -> list[IncidentEvent]:
        if runtime.last_evidence_state != "unknown":
            runtime.unknown_started_at = timestamp
            if runtime.last_evidence_state in {"present", "violation"}:
                runtime.paused_evidence_state = runtime.last_evidence_state
        runtime.conflict_started_at = None
        unknown_duration = (
            timestamp - runtime.unknown_started_at
            if runtime.unknown_started_at is not None
            else 0.0
        )
        if unknown_duration > self.config.unknown_grace_seconds:
            runtime.temporal_state = "unknown"
            runtime.violation_duration = 0.0
            runtime.present_duration = 0.0
            runtime.violation_started_at = None
            runtime.present_started_at = None
            runtime.paused_evidence_state = None
        return []

    def _update_conflict(
        self, runtime: _EquipmentRuntime, timestamp: float
    ) -> list[IncidentEvent]:
        if runtime.last_evidence_state != "conflict":
            runtime.conflict_started_at = timestamp
            if runtime.last_evidence_state in {"present", "violation"}:
                runtime.paused_evidence_state = runtime.last_evidence_state
        runtime.unknown_started_at = None
        conflict_duration = (
            timestamp - runtime.conflict_started_at
            if runtime.conflict_started_at is not None
            else 0.0
        )
        runtime.temporal_state = "conflict"
        if conflict_duration > self.config.conflict_grace_seconds:
            runtime.violation_duration = 0.0
            runtime.present_duration = 0.0
            runtime.violation_started_at = None
            runtime.present_started_at = None
            runtime.paused_evidence_state = None
        return []

    def _update_equipment(
        self,
        track: _TrackRuntime,
        runtime: _EquipmentRuntime,
        evidence: EquipmentState,
        timestamp: float,
        *,
        required: bool = True,
        zone_ids: tuple[str, ...] = (),
        zone_policy_enabled: bool = False,
    ) -> tuple[EquipmentTemporalSnapshot, list[IncidentEvent]]:
        runtime.evidence_counts[evidence] += 1
        runtime.current_zone_ids = zone_ids
        if not required:
            event = self._close_incident(runtime, timestamp, "zone_exited")
            events = [event] if event is not None else []
            runtime.temporal_state = "not_required"
            runtime.violation_duration = 0.0
            runtime.present_duration = 0.0
            runtime.violation_started_at = None
            runtime.present_started_at = None
            runtime.unknown_started_at = None
            runtime.conflict_started_at = None
            runtime.paused_evidence_state = None
            runtime.violation_zone_ids = ()
            runtime.last_stable_state = "not_required"
        elif evidence == "violation":
            events = self._update_violation(
                track,
                runtime,
                timestamp,
                zone_policy_enabled,
            )
        elif evidence == "present":
            events = self._update_present(runtime, timestamp)
        elif evidence == "conflict":
            events = self._update_conflict(runtime, timestamp)
        else:
            events = self._update_unknown(runtime, timestamp)

        if evidence != "unknown":
            runtime.last_evidence_timestamp = timestamp
        runtime.last_update_timestamp = timestamp
        runtime.last_evidence_state = evidence if required else "unknown"
        unknown_duration = (
            timestamp - runtime.unknown_started_at
            if runtime.unknown_started_at is not None
            else 0.0
        )
        conflict_duration = (
            timestamp - runtime.conflict_started_at
            if runtime.conflict_started_at is not None
            else 0.0
        )
        self._temporal_state_observations[
            f"{runtime.equipment}:{runtime.temporal_state}"
        ] += 1
        snapshot = EquipmentTemporalSnapshot(
            equipment=runtime.equipment,
            evidence_state=evidence,
            temporal_state=runtime.temporal_state,
            first_observed_timestamp=runtime.first_observed_timestamp,
            last_evidence_timestamp=runtime.last_evidence_timestamp,
            continuous_violation_duration=runtime.violation_duration,
            continuous_present_duration=runtime.present_duration,
            unknown_duration=unknown_duration,
            conflict_duration=conflict_duration,
            last_stable_state=runtime.last_stable_state,
            incident_active=runtime.active_incident is not None,
            active_incident_id=(
                runtime.active_incident.incident_id
                if runtime.active_incident is not None
                else None
            ),
            incident_duration_seconds=(
                timestamp - runtime.active_incident.started_at_seconds
                if runtime.active_incident is not None
                else None
            ),
            cooldown_until_seconds=runtime.cooldown_until,
        )
        return snapshot, events

    def update(
        self,
        record: PersonPPEFrame,
        *,
        required_equipment: frozenset[str] | None = None,
        zone_ids: tuple[str, ...] = (),
    ) -> TemporalUpdateResult:
        self._validate_timestamp(record.timestamp_seconds)
        track = self._track(record)
        self._validate_track_time(track, record.timestamp_seconds)
        if record.frame_index < 0:
            raise ValueError("Frame index cannot be negative")
        if (
            track.last_frame_index is not None
            and record.frame_index <= track.last_frame_index
        ):
            raise ValueError("Per-track frame indices must increase strictly")
        zone_policy_enabled = required_equipment is not None
        required = (
            frozenset({"helmet", "vest"})
            if required_equipment is None
            else required_equipment
        )
        helmet, helmet_events = self._update_equipment(
            track,
            track.helmet,
            record.helmet.state,
            record.timestamp_seconds,
            required="helmet" in required,
            zone_ids=zone_ids,
            zone_policy_enabled=zone_policy_enabled,
        )
        vest, vest_events = self._update_equipment(
            track,
            track.vest,
            record.vest.state,
            record.timestamp_seconds,
            required="vest" in required,
            zone_ids=zone_ids,
            zone_policy_enabled=zone_policy_enabled,
        )
        track.last_timestamp = record.timestamp_seconds
        track.last_frame_index = record.frame_index
        track.person_bbox_xyxy = record.person_bbox_xyxy
        return TemporalUpdateResult(
            record=TemporalComplianceRecord(
                frame_index=record.frame_index,
                timestamp_seconds=record.timestamp_seconds,
                source_id=record.source_id,
                track_id=record.track_id,
                person_bbox_xyxy=record.person_bbox_xyxy,
                helmet=helmet,
                vest=vest,
            ),
            events=tuple((*helmet_events, *vest_events)),
        )

    def mark_missing(
        self,
        track_id: int,
        timestamp: float,
        *,
        required_equipment: frozenset[str] | None = None,
        zone_ids: tuple[str, ...] = (),
    ) -> tuple[IncidentEvent, ...]:
        track = self._tracks.get(track_id)
        if track is None:
            return ()
        self._validate_track_time(track, timestamp)
        zone_policy_enabled = required_equipment is not None
        required = (
            frozenset({"helmet", "vest"})
            if required_equipment is None
            else required_equipment
        )
        _, helmet_events = self._update_equipment(
            track,
            track.helmet,
            "unknown",
            timestamp,
            required="helmet" in required,
            zone_ids=zone_ids,
            zone_policy_enabled=zone_policy_enabled,
        )
        _, vest_events = self._update_equipment(
            track,
            track.vest,
            "unknown",
            timestamp,
            required="vest" in required,
            zone_ids=zone_ids,
            zone_policy_enabled=zone_policy_enabled,
        )
        track.last_timestamp = timestamp
        return tuple((*helmet_events, *vest_events))

    def expire_track(
        self, track_id: int, timestamp: float
    ) -> tuple[IncidentEvent, ...]:
        self._validate_timestamp(timestamp)
        track = self._tracks.get(track_id)
        if track is None:
            return ()
        if track.last_timestamp is not None and timestamp < track.last_timestamp:
            raise ValueError("Track expiration cannot precede its last observation")
        self._tracks.pop(track_id)
        events = []
        for runtime in (track.helmet, track.vest):
            event = self._close_incident(runtime, timestamp, "track_expired")
            if event is not None:
                events.append(event)
        return tuple(events)

    def finalize(self, timestamp: float) -> tuple[IncidentEvent, ...]:
        self._validate_timestamp(timestamp)
        events = []
        for track in self._tracks.values():
            close_at = max(timestamp, track.last_timestamp or timestamp)
            for runtime in (track.helmet, track.vest):
                event = self._close_incident(runtime, close_at, "source_ended")
                if event is not None:
                    events.append(event)
        return tuple(events)
