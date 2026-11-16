from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scripts.export_review_candidates import export_reviewed

from visionops.review.schema import ReviewCandidate
from visionops.review.selection import (
    ReviewConfig,
    hash_distance,
    perceptual_hash,
    priority_score,
    select_candidates,
    temporal_deduplicate,
    visual_deduplicate,
)


def candidate(
    identifier: str,
    timestamp: float,
    kind: str,
    reason: str,
    priority: float,
    **values: object,
) -> ReviewCandidate:
    return ReviewCandidate(
        identifier,
        "clip",
        round(timestamp * 10),
        timestamp,
        kind,
        priority,
        [reason],
        **values,
    )


def test_schema_keeps_human_fields_blank_and_validates_updates() -> None:
    item = candidate("a", 0, "detector_uncertainty", "low_confidence", 20)
    assert item.human_verdict is None and item.review_status == "pending"
    item.apply_review("reviewed", "incorrect_detection", "helmet", "checked")
    assert item.human_review_label == "helmet"
    with pytest.raises(ValueError):
        item.apply_review("pending", "correct")


def test_required_sources_have_transparent_priorities() -> None:
    reasons = [
        "low_confidence",
        "association_ambiguous",
        "frame_conflict",
        "repeated_unknown",
        "confirmed_incident",
        "track_recovered",
        "drift_high",
    ]
    scores = [
        priority_score([reason], ReviewConfig().reason_weights) for reason in reasons
    ]
    assert scores == [18, 28, 35, 20, 40, 24, 8]
    assert scores[-1] < scores[2]


def test_temporal_dedup_retains_highest_priority() -> None:
    items = [
        candidate(
            "a", 1.0, "conflict", "frame_conflict", 30, track_id=1, equipment="helmet"
        ),
        candidate(
            "b", 1.5, "conflict", "frame_conflict", 35, track_id=1, equipment="helmet"
        ),
    ]
    assert [item.candidate_id for item in temporal_deduplicate(items, 2.0)] == ["b"]


def test_visual_hash_and_deduplication() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    first = candidate("a", 0, "unknown", "repeated_unknown", 20)
    second = candidate("b", 3, "unknown", "repeated_unknown", 19)
    first.visual_hash = second.visual_hash = perceptual_hash(image)
    assert hash_distance(first.visual_hash, second.visual_hash) == 0
    assert visual_deduplicate([second, first], 4) == [first]


def test_diagnostic_order_dedup_budget_and_drift_semantics() -> None:
    items = [
        candidate("low", 0, "detector_uncertainty", "low_confidence", 18),
        candidate("amb", 1, "association_uncertainty", "association_ambiguous", 28),
        candidate("conflict", 2, "frame_conflict", "frame_conflict", 35),
        candidate(
            "unknown1", 3, "temporal_uncertainty", "repeated_unknown", 20, track_id=1
        ),
        candidate(
            "unknown2", 3.2, "temporal_uncertainty", "repeated_unknown", 21, track_id=1
        ),
        candidate(
            "incident",
            5,
            "incident_context",
            "confirmed_incident",
            40,
            incident_id="i1",
        ),
        candidate("track", 7, "tracking_event", "track_recovered", 24),
        candidate("drift", 9, "drift_telemetry", "drift_high", 8, drift_level="high"),
    ]
    selected, stages = select_candidates(
        items,
        ReviewConfig(
            max_candidates_per_video=6, max_candidates_per_minute=6, max_per_type=3
        ),
    )
    assert stages == {"raw": 8, "post_temporal": 7, "post_visual": 7, "selected": 6}
    assert selected[0].candidate_id == "incident"
    drift = next(item for item in items if item.candidate_id == "drift")
    assert drift.human_verdict is None and "model_error" not in drift.reason_codes


def test_manual_export_only_reviewed_candidates(tmp_path: Path) -> None:
    queue, output = tmp_path / "queue", tmp_path / "export"
    (queue / "images").mkdir(parents=True)
    cv2.imwrite(str(queue / "images/a.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
    reviewed = candidate("a", 0, "conflict", "frame_conflict", 35)
    reviewed.artifact_reference = "images/a.jpg"
    reviewed.apply_review("accepted", "correct")
    pending = candidate("b", 1, "conflict", "frame_conflict", 35)
    (queue / "candidates.jsonl").write_text(
        json.dumps(reviewed.to_dict()) + "\n" + json.dumps(pending.to_dict()) + "\n",
        encoding="utf-8",
    )
    assert export_reviewed(queue, output) == 1
    assert (output / "a.jpg").is_file()
