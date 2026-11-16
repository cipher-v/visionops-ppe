"""Review queue schema and strictly human-owned verdict fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Literal

ReviewStatus = Literal["pending", "reviewed", "accepted", "rejected", "needs_followup"]
HumanVerdict = Literal[
    "correct",
    "incorrect_detection",
    "incorrect_association",
    "tracking_error",
    "false_violation",
    "missed_violation",
    "uncertain",
    "other",
]


@dataclass
class ReviewCandidate:
    candidate_id: str
    source_id: str
    frame_index: int
    timestamp_seconds: float
    candidate_type: str
    priority: float
    reason_codes: list[str]
    track_id: int | None = None
    equipment: str | None = None
    confidences: dict[str, float] = field(default_factory=dict)
    association_score: float | None = None
    temporal_state: str | None = None
    incident_id: str | None = None
    drift_level: str | None = None
    artifact_reference: str | None = None
    visual_hash: str | None = None
    review_status: ReviewStatus = "pending"
    human_verdict: HumanVerdict | None = None
    human_review_label: str | None = None
    review_notes: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def __post_init__(self) -> None:
        if (
            self.frame_index < 0
            or self.timestamp_seconds < 0
            or not 0 <= self.priority <= 100
        ):
            raise ValueError("Candidate frame, timestamp, or priority is invalid")
        if not self.reason_codes:
            raise ValueError("Candidate requires at least one reason")
        if self.review_status == "pending" and any(
            (self.human_verdict, self.human_review_label, self.review_notes)
        ):
            raise ValueError("Pending candidates cannot contain human review fields")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def apply_review(
        self,
        status: ReviewStatus,
        verdict: HumanVerdict,
        label: str | None = None,
        notes: str | None = None,
    ) -> None:
        if status == "pending":
            raise ValueError("A human review update cannot remain pending")
        self.review_status = status
        self.human_verdict = verdict
        self.human_review_label = label
        self.review_notes = notes
