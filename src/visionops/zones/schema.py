"""Structured spatial zone membership records."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ZoneMembershipRecord:
    frame_index: int
    timestamp_seconds: float
    source_id: str
    track_id: int
    anchor_normalized: tuple[float, float]
    raw_zone_ids: tuple[str, ...]
    stable_zone_ids: tuple[str, ...]
    required_equipment: frozenset[str]

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["anchor_normalized"] = list(self.anchor_normalized)
        values["raw_zone_ids"] = list(self.raw_zone_ids)
        values["stable_zone_ids"] = list(self.stable_zone_ids)
        values["required_equipment"] = sorted(self.required_equipment)
        return values
