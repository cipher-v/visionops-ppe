"""Configuration contract for deterministic Person-to-PPE association."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RegionFractions:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def validate(self, name: str) -> None:
        values = (self.x_min, self.x_max, self.y_min, self.y_max)
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError(f"{name} region fractions must be normalized")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError(f"{name} region must have positive area")


@dataclass(frozen=True)
class AssociationGates:
    require_center_inside_person: bool
    min_person_ioa: float
    min_region_ioa: float
    max_normalized_center_distance: float
    min_association_score: float


@dataclass(frozen=True)
class ScoreWeights:
    center_inside_region: float
    person_ioa: float
    region_ioa: float
    center_proximity: float

    def total(self) -> float:
        return sum(asdict(self).values())


@dataclass(frozen=True)
class AssociationConfig:
    """All important geometry, gate, score, and tie-break settings."""

    version: int
    head: RegionFractions
    torso: RegionFractions
    gates: AssociationGates
    weights: ScoreWeights
    person_tie_breaking: tuple[str, ...]
    primary_tie_breaking: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> AssociationConfig:
        if not path.is_file():
            raise FileNotFoundError(f"PPE association config does not exist: {path}")
        values = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("PPE association config must be a YAML mapping")
        try:
            config = cls(
                version=int(values["version"]),
                head=RegionFractions(**values["regions"]["head"]),
                torso=RegionFractions(**values["regions"]["torso"]),
                gates=AssociationGates(**values["gates"]),
                weights=ScoreWeights(**values["score_weights"]),
                person_tie_breaking=tuple(values["tie_breaking"]["person_assignment"]),
                primary_tie_breaking=tuple(
                    values["tie_breaking"]["primary_same_class"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed PPE association config: {error}") from error
        config.validate()
        return config

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError(
                f"Unsupported PPE association config version: {self.version}"
            )
        self.head.validate("head")
        self.torso.validate("torso")
        gate_values = (
            self.gates.min_person_ioa,
            self.gates.min_region_ioa,
            self.gates.min_association_score,
        )
        if any(not 0 <= value <= 1 for value in gate_values):
            raise ValueError("Containment and score gates must be normalized")
        if self.gates.max_normalized_center_distance <= 0:
            raise ValueError("Center-distance limit must be positive")
        if any(value < 0 for value in asdict(self.weights).values()):
            raise ValueError("Score weights cannot be negative")
        if abs(self.weights.total() - 1.0) > 1e-9:
            raise ValueError("Association score weights must sum to one")
        expected_person = (
            "association_score_desc",
            "person_ioa_desc",
            "center_distance_asc",
            "track_id_asc",
        )
        expected_primary = (
            "association_score_desc",
            "confidence_desc",
            "detection_index_asc",
        )
        if self.person_tie_breaking != expected_person:
            raise ValueError("Unsupported Person-assignment tie-breaking policy")
        if self.primary_tie_breaking != expected_primary:
            raise ValueError("Unsupported primary-evidence tie-breaking policy")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "regions": {
                "head": asdict(self.head),
                "torso": asdict(self.torso),
            },
            "gates": asdict(self.gates),
            "score_weights": asdict(self.weights),
            "tie_breaking": {
                "person_assignment": list(self.person_tie_breaking),
                "primary_same_class": list(self.primary_tie_breaking),
            },
        }
