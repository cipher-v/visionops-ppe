"""Strict configuration contract for normalized safety zones."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from visionops.zones.geometry import polygon_area, polygon_self_intersects

SUPPORTED_EQUIPMENT = frozenset({"helmet", "vest"})


@dataclass(frozen=True)
class MembershipConfig:
    anchor: str
    enter_confirm_seconds: float
    exit_confirm_seconds: float

    def validate(self) -> None:
        if self.anchor != "bottom_center":
            raise ValueError("Only the bottom_center zone anchor is supported")
        for name, value in (
            ("enter_confirm_seconds", self.enter_confirm_seconds),
            ("exit_confirm_seconds", self.exit_confirm_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ZoneDefinition:
    id: str
    name: str
    polygon: tuple[tuple[float, float], ...]
    required_equipment: frozenset[str]
    enabled: bool

    def validate(self) -> None:
        if not self.id.strip() or not self.name.strip():
            raise ValueError("Zone ID and name cannot be empty")
        if len(self.polygon) < 3:
            raise ValueError(f"Zone '{self.id}' requires at least three vertices")
        for point in self.polygon:
            if len(point) != 2 or any(not math.isfinite(value) for value in point):
                raise ValueError(f"Zone '{self.id}' contains a malformed vertex")
            if any(value < 0 or value > 1 for value in point):
                raise ValueError(f"Zone '{self.id}' coordinates must be normalized")
        if polygon_self_intersects(self.polygon):
            raise ValueError(f"Zone '{self.id}' polygon self-intersects")
        if polygon_area(self.polygon) <= 1e-12:
            raise ValueError(f"Zone '{self.id}' polygon has zero area")
        unknown = self.required_equipment - SUPPORTED_EQUIPMENT
        if unknown:
            raise ValueError(f"Zone '{self.id}' has unknown PPE: {sorted(unknown)}")


@dataclass(frozen=True)
class ZoneConfig:
    version: int
    membership: MembershipConfig
    zones: tuple[ZoneDefinition, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> ZoneConfig:
        if not path.is_file():
            raise FileNotFoundError(f"Zone configuration does not exist: {path}")
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Zone configuration must be a YAML mapping")
        try:
            membership_values = values["membership"]
            zone_values = values["zones"]
            if not isinstance(membership_values, dict) or not isinstance(
                zone_values, list
            ):
                raise TypeError("membership must be a mapping and zones a list")
            membership = MembershipConfig(**membership_values)
            zones = tuple(
                ZoneDefinition(
                    id=item["id"],
                    name=item["name"],
                    polygon=tuple(tuple(point) for point in item["polygon"]),
                    required_equipment=frozenset(item["required_equipment"]),
                    enabled=item.get("enabled", True),
                )
                for item in zone_values
            )
            config = cls(version=values["version"], membership=membership, zones=zones)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed zone configuration: {error}") from error
        try:
            config.validate()
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed zone configuration: {error}") from error
        return config

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError("Unsupported zone configuration version")
        self.membership.validate()
        if not self.zones:
            raise ValueError("At least one zone must be configured")
        ids = [zone.id for zone in self.zones]
        if len(ids) != len(set(ids)):
            raise ValueError("Zone IDs must be unique")
        for zone in self.zones:
            zone.validate()

    @property
    def enabled_zones(self) -> tuple[ZoneDefinition, ...]:
        return tuple(zone for zone in self.zones if zone.enabled)

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["zones"] = [
            {
                **asdict(zone),
                "polygon": [list(point) for point in zone.polygon],
                "required_equipment": sorted(zone.required_equipment),
            }
            for zone in self.zones
        ]
        return values
