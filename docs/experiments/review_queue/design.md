# Review queue design

Checkpoint 18 turns existing detector, association, temporal, tracking, incident and drift
telemetry into a bounded human-review queue. It neither labels training data nor starts
training. A candidate records a deterministic ID, run-local source/frame/time/track,
type, reasons, priority, equipment, confidence and association/temporal/incident/drift
context, artifact reference, lifecycle status, and separate human-owned fields.

Allowed states are `pending`, `reviewed`, `accepted`, `rejected`, and
`needs_followup`. Verdicts are operational review judgments, not training annotations.
Pending candidates reject populated human fields. Track IDs remain anonymous and run-local.

Generated frames and manifests live under ignored `results/review_queue/<run_id>/`.
No review API was added: the deliberate JSONL/CSV workflow avoids creating a partially
secured annotation service. Review updates are made offline and exported manually.

