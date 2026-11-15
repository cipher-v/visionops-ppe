"""Configuration contract for timestamp-driven compliance state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EquipmentTiming:
    violation_confirm_seconds: float
    clear_confirm_seconds: float

    def validate(self, name: str) -> None:
        if self.violation_confirm_seconds <= 0:
            raise ValueError(f"{name} violation confirmation must be positive")
        if self.clear_confirm_seconds <= 0:
            raise ValueError(f"{name} clear confirmation must be positive")


@dataclass(frozen=True)
class TemporalComplianceConfig:
    version: int
    helmet: EquipmentTiming
    vest: EquipmentTiming
    unknown_grace_seconds: float
    conflict_grace_seconds: float
    incident_cooldown_seconds: float
    require_strictly_increasing_per_track: bool
    decisions_use_elapsed_seconds: bool

    @classmethod
    def from_yaml(cls, path: Path) -> TemporalComplianceConfig:
        if not path.is_file():
            raise FileNotFoundError(
                f"Temporal compliance config does not exist: {path}"
            )
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Temporal compliance config must be a YAML mapping")
        try:
            policy = values["timestamp_policy"]
            config = cls(
                version=int(values["version"]),
                helmet=EquipmentTiming(**values["helmet"]),
                vest=EquipmentTiming(**values["vest"]),
                unknown_grace_seconds=float(values["unknown_grace_seconds"]),
                conflict_grace_seconds=float(values["conflict_grace_seconds"]),
                incident_cooldown_seconds=float(values["incident_cooldown_seconds"]),
                require_strictly_increasing_per_track=bool(
                    policy["require_strictly_increasing_per_track"]
                ),
                decisions_use_elapsed_seconds=bool(
                    policy["decisions_use_elapsed_seconds"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Malformed temporal compliance config: {error}"
            ) from error
        config.validate()
        return config

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError(f"Unsupported temporal config version: {self.version}")
        self.helmet.validate("helmet")
        self.vest.validate("vest")
        if self.unknown_grace_seconds < 0 or self.conflict_grace_seconds < 0:
            raise ValueError("Grace durations cannot be negative")
        if self.incident_cooldown_seconds < 0:
            raise ValueError("Incident cooldown cannot be negative")
        if not self.require_strictly_increasing_per_track:
            raise ValueError("Per-track timestamps must be strictly increasing")
        if not self.decisions_use_elapsed_seconds:
            raise ValueError("Temporal decisions must use elapsed seconds")

    def timing_for(self, equipment: str) -> EquipmentTiming:
        if equipment == "helmet":
            return self.helmet
        if equipment == "vest":
            return self.vest
        raise ValueError(f"Unsupported equipment: {equipment}")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "helmet": asdict(self.helmet),
            "vest": asdict(self.vest),
            "unknown_grace_seconds": self.unknown_grace_seconds,
            "conflict_grace_seconds": self.conflict_grace_seconds,
            "incident_cooldown_seconds": self.incident_cooldown_seconds,
            "timestamp_policy": {
                "require_strictly_increasing_per_track": (
                    self.require_strictly_increasing_per_track
                ),
                "decisions_use_elapsed_seconds": self.decisions_use_elapsed_seconds,
            },
        }
