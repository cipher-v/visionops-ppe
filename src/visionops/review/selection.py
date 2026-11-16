"""Transparent priority, temporal/visual deduplication, and queue budgets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from visionops.review.schema import ReviewCandidate

DEFAULT_WEIGHTS = {
    "low_confidence": 18.0,
    "association_ambiguous": 28.0,
    "association_near_gate": 18.0,
    "high_confidence_unassigned": 26.0,
    "frame_conflict": 35.0,
    "repeated_unknown": 20.0,
    "suspected_violation": 24.0,
    "confirmed_incident": 40.0,
    "incident_closed": 28.0,
    "track_lost": 20.0,
    "track_recovered": 24.0,
    "short_track": 16.0,
    "drift_warning": 4.0,
    "drift_high": 8.0,
}


@dataclass(frozen=True)
class ReviewConfig:
    reason_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS)
    )
    temporal_gap_seconds: float = 2.0
    visual_hamming_threshold: int = 4
    max_candidates_per_video: int = 40
    max_candidates_per_minute: int = 12
    max_per_type: int = 12


def priority_score(
    reason_codes: list[str],
    weights: dict[str, float],
    confidence_margin: float | None = None,
) -> float:
    score = sum(weights.get(reason, 0.0) for reason in set(reason_codes))
    if confidence_margin is not None:
        score += max(0.0, 12.0 * (1.0 - min(confidence_margin / 0.10, 1.0)))
    return min(100.0, round(score, 4))


def perceptual_hash(image: np.ndarray) -> str:
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("Visual deduplication requires a color image")
    gray = cv2.cvtColor(cv2.resize(image, (9, 8)), cv2.COLOR_BGR2GRAY).astype(float)
    bits = gray[:, 1:] > gray[:, :-1]
    return f"{int(''.join('1' if bit else '0' for bit in bits.flat), 2):016x}"


def hash_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def temporal_deduplicate(
    candidates: list[ReviewCandidate], gap_seconds: float
) -> list[ReviewCandidate]:
    if gap_seconds < 0:
        raise ValueError("Temporal gap cannot be negative")
    selected: list[ReviewCandidate] = []
    groups: dict[tuple[object, ...], list[ReviewCandidate]] = {}
    for item in candidates:
        key = (item.source_id, item.candidate_type, item.track_id, item.equipment)
        groups.setdefault(key, []).append(item)
    for items in groups.values():
        items.sort(
            key=lambda item: (item.timestamp_seconds, -item.priority, item.candidate_id)
        )
        bucket: list[ReviewCandidate] = []
        bucket_start = items[0].timestamp_seconds
        for item in items:
            if item.timestamp_seconds - bucket_start >= gap_seconds and bucket:
                selected.append(
                    max(
                        bucket,
                        key=lambda value: (
                            value.priority,
                            -value.timestamp_seconds,
                            value.candidate_id,
                        ),
                    )
                )
                bucket, bucket_start = [], item.timestamp_seconds
            bucket.append(item)
        if bucket:
            selected.append(
                max(
                    bucket,
                    key=lambda value: (
                        value.priority,
                        -value.timestamp_seconds,
                        value.candidate_id,
                    ),
                )
            )
    return sorted(
        selected,
        key=lambda item: (-item.priority, item.timestamp_seconds, item.candidate_id),
    )


def visual_deduplicate(
    candidates: list[ReviewCandidate], threshold: int
) -> list[ReviewCandidate]:
    retained: list[ReviewCandidate] = []
    for item in sorted(
        candidates,
        key=lambda value: (
            -value.priority,
            value.timestamp_seconds,
            value.candidate_id,
        ),
    ):
        redundant = any(
            item.visual_hash
            and prior.visual_hash
            and item.source_id == prior.source_id
            and hash_distance(item.visual_hash, prior.visual_hash) <= threshold
            for prior in retained
        )
        if not redundant:
            retained.append(item)
    return retained


def apply_budget(
    candidates: list[ReviewCandidate], config: ReviewConfig
) -> list[ReviewCandidate]:
    selected: list[ReviewCandidate] = []
    type_counts: Counter[str] = Counter()
    minute_counts: Counter[tuple[str, int]] = Counter()
    source_counts: Counter[str] = Counter()
    for item in sorted(
        candidates,
        key=lambda value: (
            -value.priority,
            value.timestamp_seconds,
            value.candidate_id,
        ),
    ):
        minute = int(item.timestamp_seconds // 60)
        if (
            source_counts[item.source_id] >= config.max_candidates_per_video
            or type_counts[item.candidate_type] >= config.max_per_type
            or minute_counts[(item.source_id, minute)]
            >= config.max_candidates_per_minute
        ):
            continue
        selected.append(item)
        source_counts[item.source_id] += 1
        type_counts[item.candidate_type] += 1
        minute_counts[(item.source_id, minute)] += 1
    return selected


def select_candidates(
    candidates: list[ReviewCandidate], config: ReviewConfig
) -> tuple[list[ReviewCandidate], dict[str, int]]:
    temporal = temporal_deduplicate(candidates, config.temporal_gap_seconds)
    visual = visual_deduplicate(temporal, config.visual_hamming_threshold)
    final = apply_budget(visual, config)
    return final, {
        "raw": len(candidates),
        "post_temporal": len(temporal),
        "post_visual": len(visual),
        "selected": len(final),
    }
