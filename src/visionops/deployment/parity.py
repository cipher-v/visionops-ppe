"""Deterministic one-to-one detection parity matching."""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass

from visionops.video.policy import CandidateBox


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class DetectionMatch:
    reference_index: int
    candidate_index: int
    class_id: int
    iou: float
    confidence_delta: float


def match_detections(
    reference: list[CandidateBox],
    candidate: list[CandidateBox],
    minimum_iou: float = 0.90,
) -> tuple[DetectionMatch, ...]:
    possible = []
    for reference_index, reference_box in enumerate(reference):
        for candidate_index, candidate_box in enumerate(candidate):
            if reference_box.class_id != candidate_box.class_id:
                continue
            overlap = box_iou(reference_box.bbox_xyxy, candidate_box.bbox_xyxy)
            if overlap >= minimum_iou:
                possible.append((overlap, reference_index, candidate_index))
    possible.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_reference: set[int] = set()
    used_candidate: set[int] = set()
    matches = []
    for overlap, reference_index, candidate_index in possible:
        if reference_index in used_reference or candidate_index in used_candidate:
            continue
        used_reference.add(reference_index)
        used_candidate.add(candidate_index)
        matches.append(
            DetectionMatch(
                reference_index=reference_index,
                candidate_index=candidate_index,
                class_id=reference[reference_index].class_id,
                iou=overlap,
                confidence_delta=abs(
                    reference[reference_index].confidence
                    - candidate[candidate_index].confidence
                ),
            )
        )
    return tuple(matches)


def summarize_parity(
    reference: list[CandidateBox],
    candidate: list[CandidateBox],
    matches: tuple[DetectionMatch, ...],
) -> dict[str, object]:
    ious = sorted(match.iou for match in matches)
    deltas = sorted(match.confidence_delta for match in matches)

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        return values[round((len(values) - 1) * fraction)]

    per_class: dict[int, dict[str, int]] = defaultdict(
        lambda: {"reference": 0, "candidate": 0, "matched": 0}
    )
    for box in reference:
        per_class[box.class_id]["reference"] += 1
    for box in candidate:
        per_class[box.class_id]["candidate"] += 1
    for match in matches:
        per_class[match.class_id]["matched"] += 1
    denominator = max(len(reference), len(candidate), 1)
    return {
        "reference_detection_count": len(reference),
        "candidate_detection_count": len(candidate),
        "matched_detections": len(matches),
        "unmatched_reference_detections": len(reference) - len(matches),
        "unmatched_candidate_detections": len(candidate) - len(matches),
        "match_rate": len(matches) / denominator,
        "mean_matched_iou": statistics.fmean(ious) if ious else None,
        "p5_matched_iou": percentile(ious, 0.05),
        "mean_absolute_confidence_delta": (
            statistics.fmean(deltas) if deltas else None
        ),
        "p95_absolute_confidence_delta": percentile(deltas, 0.95),
        "per_class": {key: per_class[key] for key in sorted(per_class)},
    }
