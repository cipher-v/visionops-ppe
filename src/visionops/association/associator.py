"""Deterministic geometry-aware assignment of PPE detections to Person tracks."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from visionops.association.config import AssociationConfig
from visionops.association.geometry import (
    GeometricSignals,
    anatomical_region,
    geometric_signals,
    valid_box,
)
from visionops.association.schema import (
    AssociationFrameResult,
    EquipmentEvidence,
    EquipmentState,
    PersonPPEFrame,
    PPEAssociation,
)
from visionops.tracking.schema import PersonTrack
from visionops.video.schema import VideoDetection

PPE_CLASS_NAMES = {0: "helmet", 1: "vest", 3: "no_helmet", 4: "no_vest"}
HEAD_CLASS_IDS = {0, 3}
TORSO_CLASS_IDS = {1, 4}


@dataclass(frozen=True)
class _Candidate:
    detection_index: int
    detection: VideoDetection
    track: PersonTrack
    signals: GeometricSignals
    score: float
    alternative_track_ids: tuple[int, ...] = ()
    candidate_person_count: int = 1


def equipment_state(has_positive: bool, has_negative: bool) -> EquipmentState:
    """Map independent positive/negative evidence to an explicit frame state."""
    if has_positive and has_negative:
        return "conflict"
    if has_positive:
        return "present"
    if has_negative:
        return "violation"
    return "unknown"


class PPEAssociator:
    """Associate each valid PPE detection with at most one active Person track."""

    def __init__(self, config: AssociationConfig) -> None:
        config.validate()
        self.config = config

    def metadata(self) -> dict[str, object]:
        return {
            "name": "VisionOps deterministic geometry association",
            "learned_association": False,
            "configuration": self.config.to_dict(),
            "supported_ppe_classes": PPE_CLASS_NAMES,
            "missing_detection_semantics": "unknown",
        }

    def _score(self, signals: GeometricSignals) -> float:
        weights = self.config.weights
        return (
            weights.center_inside_region * float(signals.center_inside_region)
            + weights.person_ioa * signals.person_ioa
            + weights.region_ioa * signals.region_ioa
            + weights.center_proximity * signals.center_proximity
        )

    def _candidate(
        self,
        detection_index: int,
        detection: VideoDetection,
        track: PersonTrack,
    ) -> _Candidate | None:
        fractions = (
            self.config.head
            if detection.class_id in HEAD_CLASS_IDS
            else self.config.torso
        )
        region = anatomical_region(track.bbox_xyxy, fractions)
        signals = geometric_signals(detection.bbox_xyxy, track.bbox_xyxy, region)
        gates = self.config.gates
        if gates.require_center_inside_person and not signals.center_inside_person:
            return None
        if signals.person_ioa < gates.min_person_ioa:
            return None
        if signals.region_ioa < gates.min_region_ioa:
            return None
        if signals.normalized_center_distance > gates.max_normalized_center_distance:
            return None
        score = self._score(signals)
        if score < gates.min_association_score:
            return None
        return _Candidate(detection_index, detection, track, signals, score)

    @staticmethod
    def _person_sort_key(candidate: _Candidate) -> tuple[float, float, float, int]:
        return (
            -candidate.score,
            -candidate.signals.person_ioa,
            candidate.signals.normalized_center_distance,
            candidate.track.track_id,
        )

    @staticmethod
    def _primary_sort_key(candidate: _Candidate) -> tuple[float, float, int]:
        return (
            -candidate.score,
            -candidate.detection.confidence,
            candidate.detection_index,
        )

    @staticmethod
    def _association(candidate: _Candidate, primary: bool) -> PPEAssociation:
        return PPEAssociation(
            detection_index=candidate.detection_index,
            ppe_class_id=candidate.detection.class_id,
            ppe_class_name=candidate.detection.class_name,
            ppe_confidence=candidate.detection.confidence,
            ppe_bbox_xyxy=candidate.detection.bbox_xyxy,
            association_score=candidate.score,
            center_inside_region=candidate.signals.center_inside_region,
            person_ioa=candidate.signals.person_ioa,
            region_ioa=candidate.signals.region_ioa,
            normalized_center_distance=(candidate.signals.normalized_center_distance),
            candidate_person_count=candidate.candidate_person_count,
            ambiguous=candidate.candidate_person_count > 1,
            alternative_track_ids=candidate.alternative_track_ids,
            primary_for_class=primary,
        )

    def _evidence(
        self,
        by_class: dict[int, list[_Candidate]],
        positive_class: int,
        negative_class: int,
    ) -> EquipmentEvidence:
        positive_raw = sorted(
            by_class.get(positive_class, []), key=self._primary_sort_key
        )
        negative_raw = sorted(
            by_class.get(negative_class, []), key=self._primary_sort_key
        )
        positive = tuple(
            self._association(candidate, index == 0)
            for index, candidate in enumerate(positive_raw)
        )
        negative = tuple(
            self._association(candidate, index == 0)
            for index, candidate in enumerate(negative_raw)
        )
        return EquipmentEvidence(
            state=equipment_state(bool(positive), bool(negative)),
            positive_detection=positive[0] if positive else None,
            negative_detection=negative[0] if negative else None,
            positive_candidates=positive,
            negative_candidates=negative,
        )

    def associate(
        self,
        tracks: tuple[PersonTrack, ...],
        detections: list[VideoDetection],
    ) -> AssociationFrameResult:
        """Associate one frame and emit evidence for every active Person track."""
        track_ids = [track.track_id for track in tracks]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("Active Person track IDs must be unique within a frame")
        for track in tracks:
            if track.class_id != 2 or track.class_name != "Person":
                raise ValueError("PPE association accepts Person tracks only")
            if not valid_box(track.bbox_xyxy):
                raise ValueError("Active Person track has a malformed or zero-area box")
        active_frame = tracks[0].frame_index if tracks else None
        if active_frame is not None and any(
            detection.frame_index != active_frame for detection in detections
        ):
            raise ValueError("Detections and active Person tracks must share a frame")

        ppe_items = [
            (index, detection)
            for index, detection in enumerate(detections)
            if detection.class_id in PPE_CLASS_NAMES
        ]
        assigned_by_track: dict[int, dict[int, list[_Candidate]]] = defaultdict(
            lambda: defaultdict(list)
        )
        assigned_indices: set[int] = set()
        unassigned_indices: set[int] = set()
        malformed = 0
        ambiguous = 0

        for detection_index, detection in ppe_items:
            if (
                detection.class_name != PPE_CLASS_NAMES[detection.class_id]
                or not math.isfinite(detection.confidence)
                or not valid_box(detection.bbox_xyxy)
            ):
                malformed += 1
                unassigned_indices.add(detection_index)
                continue
            candidates = [
                candidate
                for track in tracks
                if (candidate := self._candidate(detection_index, detection, track))
                is not None
            ]
            if not candidates:
                unassigned_indices.add(detection_index)
                continue
            ordered = sorted(candidates, key=self._person_sort_key)
            chosen = ordered[0]
            alternatives = tuple(candidate.track.track_id for candidate in ordered[1:])
            if alternatives:
                ambiguous += 1
            chosen = _Candidate(
                detection_index=chosen.detection_index,
                detection=chosen.detection,
                track=chosen.track,
                signals=chosen.signals,
                score=chosen.score,
                alternative_track_ids=alternatives,
                candidate_person_count=len(ordered),
            )
            assigned_by_track[chosen.track.track_id][detection.class_id].append(chosen)
            assigned_indices.add(detection_index)

        records = []
        for track in sorted(tracks, key=lambda item: item.track_id):
            by_class = assigned_by_track[track.track_id]
            records.append(
                PersonPPEFrame(
                    frame_index=track.frame_index,
                    timestamp_seconds=track.timestamp_seconds,
                    track_id=track.track_id,
                    person_bbox_xyxy=track.bbox_xyxy,
                    person_bbox_normalized=track.bbox_normalized,
                    source_id=track.source_id,
                    helmet=self._evidence(by_class, 0, 3),
                    vest=self._evidence(by_class, 1, 4),
                )
            )
        return AssociationFrameResult(
            records=tuple(records),
            assigned_detection_indices=frozenset(assigned_indices),
            unassigned_detection_indices=frozenset(unassigned_indices),
            ppe_detections_considered=len(ppe_items),
            malformed_ppe_detections=malformed,
            ambiguous_detections=ambiguous,
        )
