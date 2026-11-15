# Frame-level Person-to-PPE association design

## Scope and evidence semantics

Association runs only when both tracking and `--enable-ppe-association` are enabled. It
uses active Person tracks and the already-filtered frame detections. It does not modify
YOLO, ByteTrack, or their frozen thresholds. It creates frame-level evidence, not temporal
compliance, an alert, or an incident.

Helmet and `no_helmet` are independent evidence channels, as are vest and `no_vest`.
Missing detection always means `unknown`; it never implies presence or violation.

| Positive | Negative | State |
|---|---|---|
| yes | no | `present` |
| no | yes | `violation` |
| yes | yes | `conflict` |
| no | no | `unknown` |

## Anatomical proxy regions

All fractions in `configs/ppe_association.yaml` are relative to the current Person track
box:

- Head: x `[0.125, 0.875]`, y `[0.00, 0.35]` — central 75% of width and top 35% of height.
- Torso: x `[0.075, 0.925]`, y `[0.20, 0.72]` — central 85% of width and 20–72% of height.

Helmet classes use the head region; vest classes use the torso region. These are practical
geometric priors, not anatomical truths, and require real-camera validation.

## Geometry and score

For PPE box `B`, Person box `P`, and class-appropriate proxy region `R`:

- `C` is 1 when the PPE center is inside `R`, otherwise 0.
- `IoA_P = area(B intersection P) / area(B)`.
- `IoA_R = area(B intersection R) / area(B)`.
- `d` is PPE-center distance from the region center divided by the region half-diagonal.
- `proximity = max(0, 1 - d)`.

The configured score is:

```text
score = 0.30*C + 0.20*IoA_P + 0.30*IoA_R + 0.20*proximity
```

PPE-box IoA is used instead of full-box IoU because a valid PPE box is expected to occupy
only a small part of a Person box.

## Gates

A candidate Person/PPE pair is rejected before assignment when:

- either box is malformed or has zero area;
- the PPE center is outside the Person box;
- Person containment is below `0.60`;
- proxy-region containment is below `0.10`;
- normalized center distance exceeds `1.25`; or
- final association score is below `0.50`.

Only active tracks supplied for the current processed frame are candidates.

## Deterministic assignment and ambiguity

Each PPE detection is evaluated against all active Person tracks. Valid candidates sort by
score descending, Person IoA descending, center distance ascending, and track ID ascending.
Only the first receives the detection, so one PPE box cannot be assigned to multiple
people. Alternative track IDs and candidate count remain in the winning diagnostic record.

Multiple same-class detections may remain associated with one Person for diagnostics. The
primary item sorts by score descending, detection confidence descending, and original
detection index ascending. Nothing ambiguous is silently discarded.

## Output schema

Each `ppe_associations.jsonl` line represents one active Person observation and includes
frame index, timestamp, track ID, Person boxes, source ID, and separate helmet/vest
evidence. Evidence includes state, primary positive/negative detections, all candidates,
association score, IoA values, center distance, ambiguity, and alternatives. It contains no
dwell time, incident, severity, zone, alert, or compliance persistence.

## Pipeline and visualization

The detector still runs once. Person detections feed ByteTrack, and the resulting active
tracks plus raw PPE detections feed the repository associator. Detector-only and tracking-
only modes remain available. Association mode writes detections, tracks, associations,
metadata, and annotated video. Raw PPE labels remain visible; Person labels add concise
`H:` and `V:` evidence states. `unknown` is displayed literally and never labeled safe.

```powershell
.\.venv\Scripts\python.exe scripts\run_video_inference.py `
  --source path\to\video.mp4 `
  --enable-tracking `
  --enable-ppe-association `
  --output-dir results\video_inference\ppe_demo
```
