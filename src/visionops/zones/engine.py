"""Timestamp-stabilized per-track zone membership and diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from visionops.association.schema import PersonPPEFrame
from visionops.compliance.schema import TemporalUpdateResult
from visionops.zones.config import ZoneConfig
from visionops.zones.geometry import bottom_center, point_in_polygon
from visionops.zones.schema import ZoneMembershipRecord


@dataclass
class _ZoneState:
    stable_inside: bool = False
    enter_started_at: float | None = None
    exit_started_at: float | None = None
    last_raw_inside: bool = False


@dataclass
class _TrackState:
    zones: dict[str, _ZoneState] = field(default_factory=dict)
    last_timestamp: float | None = None
    stable_zone_ids: tuple[str, ...] = ()


class ZoneDiagnostics:
    def __init__(self) -> None:
        self.person_observations_evaluated = 0
        self.raw_zone_entries = 0
        self.raw_zone_exits = 0
        self.stable_zone_entries = 0
        self.stable_zone_exits = 0
        self.observations_inside: Counter[str] = Counter()
        self.required_observations: Counter[str] = Counter()
        self.violations_outside_required_suppressed = 0
        self.in_zone_suspected_violations = 0
        self.in_zone_confirmed_incidents = 0
        self.incidents_closed_by_zone_exit = 0

    def observe_membership(self, record: ZoneMembershipRecord) -> None:
        self.person_observations_evaluated += 1
        self.observations_inside.update(record.stable_zone_ids)
        self.required_observations.update(record.required_equipment)

    def observe_policy(
        self,
        association: PersonPPEFrame,
        membership: ZoneMembershipRecord,
        temporal: TemporalUpdateResult,
    ) -> None:
        for equipment in ("helmet", "vest"):
            evidence = getattr(association, equipment).state
            snapshot = getattr(temporal.record, equipment)
            required = equipment in membership.required_equipment
            if evidence == "violation" and not required:
                self.violations_outside_required_suppressed += 1
            if required and snapshot.temporal_state == "suspected_violation":
                self.in_zone_suspected_violations += 1
        for event in temporal.events:
            if event.event_type == "incident_opened":
                self.in_zone_confirmed_incidents += 1
            if event.incident.get("close_reason") == "zone_exited":
                self.incidents_closed_by_zone_exit += 1

    def summary(self) -> dict[str, object]:
        return {
            "person_observations_evaluated": self.person_observations_evaluated,
            "raw_zone_entries": self.raw_zone_entries,
            "raw_zone_exits": self.raw_zone_exits,
            "stable_zone_entries": self.stable_zone_entries,
            "stable_zone_exits": self.stable_zone_exits,
            "observations_inside_each_zone": dict(
                sorted(self.observations_inside.items())
            ),
            "equipment_required_observations": dict(
                sorted(self.required_observations.items())
            ),
            "violations_outside_required_zones_suppressed": (
                self.violations_outside_required_suppressed
            ),
            "in_zone_suspected_violations": self.in_zone_suspected_violations,
            "in_zone_confirmed_incidents": self.in_zone_confirmed_incidents,
            "incidents_closed_by_zone_exit": self.incidents_closed_by_zone_exit,
        }


class ZoneEngine:
    """Determine raw and stabilized zone policy for anonymous Person tracks."""

    def __init__(self, config: ZoneConfig) -> None:
        config.validate()
        self.config = config
        self.diagnostics = ZoneDiagnostics()
        self._tracks: dict[int, _TrackState] = {}

    def update(
        self,
        *,
        frame_index: int,
        timestamp_seconds: float,
        source_id: str,
        track_id: int,
        person_bbox_normalized: tuple[float, float, float, float],
    ) -> ZoneMembershipRecord:
        if frame_index < 0:
            raise ValueError("Frame index cannot be negative")
        if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise ValueError("Zone timestamp must be finite and non-negative")
        state = self._tracks.setdefault(track_id, _TrackState())
        if (
            state.last_timestamp is not None
            and timestamp_seconds <= state.last_timestamp
        ):
            raise ValueError("Per-track zone timestamps must increase strictly")
        anchor = bottom_center(person_bbox_normalized)
        raw_ids = []
        stable_ids = []
        for zone in self.config.enabled_zones:
            raw_inside = point_in_polygon(anchor, zone.polygon)
            zone_state = state.zones.setdefault(zone.id, _ZoneState())
            if raw_inside != zone_state.last_raw_inside:
                if raw_inside:
                    self.diagnostics.raw_zone_entries += 1
                else:
                    self.diagnostics.raw_zone_exits += 1
            if raw_inside:
                raw_ids.append(zone.id)
                zone_state.exit_started_at = None
                if not zone_state.stable_inside:
                    if zone_state.enter_started_at is None:
                        zone_state.enter_started_at = timestamp_seconds
                    if (
                        timestamp_seconds - zone_state.enter_started_at
                        >= self.config.membership.enter_confirm_seconds
                    ):
                        zone_state.stable_inside = True
                        self.diagnostics.stable_zone_entries += 1
            else:
                zone_state.enter_started_at = None
                if zone_state.stable_inside:
                    if zone_state.exit_started_at is None:
                        zone_state.exit_started_at = timestamp_seconds
                    if (
                        timestamp_seconds - zone_state.exit_started_at
                        >= self.config.membership.exit_confirm_seconds
                    ):
                        zone_state.stable_inside = False
                        self.diagnostics.stable_zone_exits += 1
            zone_state.last_raw_inside = raw_inside
            if zone_state.stable_inside:
                stable_ids.append(zone.id)
        required = frozenset(
            equipment
            for zone in self.config.enabled_zones
            if zone.id in stable_ids
            for equipment in zone.required_equipment
        )
        state.last_timestamp = timestamp_seconds
        state.stable_zone_ids = tuple(sorted(stable_ids))
        record = ZoneMembershipRecord(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            source_id=source_id,
            track_id=track_id,
            anchor_normalized=anchor,
            raw_zone_ids=tuple(sorted(raw_ids)),
            stable_zone_ids=state.stable_zone_ids,
            required_equipment=required,
        )
        self.diagnostics.observe_membership(record)
        return record

    def stable_policy(self, track_id: int) -> tuple[tuple[str, ...], frozenset[str]]:
        state = self._tracks.get(track_id)
        zone_ids = state.stable_zone_ids if state is not None else ()
        required = frozenset(
            equipment
            for zone in self.config.enabled_zones
            if zone.id in zone_ids
            for equipment in zone.required_equipment
        )
        return zone_ids, required

    def expire_track(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)
