"""Aggregate observable frame-level PPE association diagnostics."""

from __future__ import annotations

import statistics
from collections import Counter

from visionops.association.schema import AssociationFrameResult, PersonPPEFrame


def _matches(record: PersonPPEFrame):
    for equipment in (record.helmet, record.vest):
        yield from equipment.positive_candidates
        yield from equipment.negative_candidates


class AssociationDiagnostics:
    def __init__(self) -> None:
        self.total_person_observations = 0
        self.ppe_detections_considered = 0
        self.ppe_detections_assigned = 0
        self.ppe_detections_unassigned = 0
        self.malformed_ppe_detections = 0
        self.ambiguous_candidate_count = 0
        self.scores: list[float] = []
        self.per_class_assignment_counts: Counter[str] = Counter()
        self.helmet_states: Counter[str] = Counter()
        self.vest_states: Counter[str] = Counter()

    def update(self, result: AssociationFrameResult) -> None:
        self.total_person_observations += len(result.records)
        self.ppe_detections_considered += result.ppe_detections_considered
        self.ppe_detections_assigned += len(result.assigned_detection_indices)
        self.ppe_detections_unassigned += len(result.unassigned_detection_indices)
        self.malformed_ppe_detections += result.malformed_ppe_detections
        self.ambiguous_candidate_count += result.ambiguous_detections
        for record in result.records:
            self.helmet_states[record.helmet.state] += 1
            self.vest_states[record.vest.state] += 1
            for match in _matches(record):
                self.scores.append(match.association_score)
                self.per_class_assignment_counts[match.ppe_class_name] += 1

    def summary(self) -> dict[str, object]:
        states = ("present", "violation", "conflict", "unknown")
        return {
            "total_person_observations": self.total_person_observations,
            "ppe_detections_considered": self.ppe_detections_considered,
            "ppe_detections_assigned": self.ppe_detections_assigned,
            "ppe_detections_unassigned": self.ppe_detections_unassigned,
            "malformed_ppe_detections": self.malformed_ppe_detections,
            "ambiguous_candidate_count": self.ambiguous_candidate_count,
            "mean_association_score": (
                statistics.fmean(self.scores) if self.scores else None
            ),
            "per_class_assignment_counts": dict(
                sorted(self.per_class_assignment_counts.items())
            ),
            "frame_level_equipment_states": {
                "helmet": {state: self.helmet_states[state] for state in states},
                "vest": {state: self.vest_states[state] for state in states},
            },
        }
