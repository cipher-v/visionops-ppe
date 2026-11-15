"""Frame-level PPE evidence schemas with explicit unknown semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

EquipmentState = Literal["present", "violation", "conflict", "unknown"]


@dataclass(frozen=True)
class PPEAssociation:
    detection_index: int
    ppe_class_id: int
    ppe_class_name: str
    ppe_confidence: float
    ppe_bbox_xyxy: tuple[float, float, float, float]
    association_score: float
    center_inside_region: bool
    person_ioa: float
    region_ioa: float
    normalized_center_distance: float
    candidate_person_count: int
    ambiguous: bool
    alternative_track_ids: tuple[int, ...]
    primary_for_class: bool

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["ppe_bbox_xyxy"] = list(self.ppe_bbox_xyxy)
        values["alternative_track_ids"] = list(self.alternative_track_ids)
        return values


@dataclass(frozen=True)
class EquipmentEvidence:
    state: EquipmentState
    positive_detection: PPEAssociation | None
    negative_detection: PPEAssociation | None
    positive_candidates: tuple[PPEAssociation, ...]
    negative_candidates: tuple[PPEAssociation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "positive_detection": (
                self.positive_detection.to_dict()
                if self.positive_detection is not None
                else None
            ),
            "negative_detection": (
                self.negative_detection.to_dict()
                if self.negative_detection is not None
                else None
            ),
            "positive_candidates": [
                item.to_dict() for item in self.positive_candidates
            ],
            "negative_candidates": [
                item.to_dict() for item in self.negative_candidates
            ],
        }


@dataclass(frozen=True)
class PersonPPEFrame:
    frame_index: int
    timestamp_seconds: float
    track_id: int
    person_bbox_xyxy: tuple[float, float, float, float]
    person_bbox_normalized: tuple[float, float, float, float]
    source_id: str
    helmet: EquipmentEvidence
    vest: EquipmentEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "track_id": self.track_id,
            "person_bbox_xyxy": list(self.person_bbox_xyxy),
            "person_bbox_normalized": list(self.person_bbox_normalized),
            "source_id": self.source_id,
            "helmet": self.helmet.to_dict(),
            "vest": self.vest.to_dict(),
        }


@dataclass(frozen=True)
class AssociationFrameResult:
    records: tuple[PersonPPEFrame, ...]
    assigned_detection_indices: frozenset[int]
    unassigned_detection_indices: frozenset[int]
    ppe_detections_considered: int
    malformed_ppe_detections: int
    ambiguous_detections: int
