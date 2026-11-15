# Timestamp-driven temporal PPE compliance

## Scope

The temporal engine consumes explicit frame-level `present`, `violation`, `conflict`, and
`unknown` states for helmet and vest. It does not infer violation from missing positive
PPE, and it does not turn conflict into either compliance or violation. State is independent
for each run-local Person track and equipment type.

All decisions use source timestamps in seconds. Frame count never determines confirmation,
clearing, grace, or cooldown because source FPS, stride, and dropped frames can vary.

## Initial engineering configuration

`configs/compliance_temporal.yaml` defines:

| Setting | Helmet | Vest |
|---|---:|---:|
| Violation confirmation | 1.5 s | 1.5 s |
| Positive-evidence clearing | 1.0 s | 1.0 s |

Global values are unknown grace `0.75 s`, conflict grace `0.50 s`, and post-close
incident cooldown `5.0 s`. These are initial engineering defaults, not calibrated safety
standards. They must not be tuned further without real PPE video.

## State machine

Temporal states are `unknown`, `compliant`, `suspected_violation`,
`confirmed_violation`, and `conflict`.

- `present`: begins or reinforces positive evidence. With no active incident, state becomes
  compliant. An active incident remains confirmed until positive evidence accumulates the
  configured clear duration.
- `violation`: begins suspected violation. Only repeated explicit violation evidence whose
  accumulated elapsed time reaches the confirmation duration opens an incident. An active
  incident remains the same incident.
- `unknown`: pauses present/violation accumulation. Within 0.75 seconds it preserves the
  prior temporal state. Beyond grace it resets candidate timers and transitions to unknown.
  It never opens or clears an incident.
- `conflict`: immediately emits temporal conflict. Within 0.50 seconds it pauses prior
  evidence; beyond grace it invalidates candidate timers. It never forces compliance,
  violation, opening, or clearing.

Short unknown/conflict gaps do not add time to either timer. For example, violation from
0.0–0.8 s, unknown from 1.0–1.2 s, then violation resumes with 0.8 s accumulated—not
1.2 s. A long gap resets an unconfirmed candidate. An already active incident remains open
through unknown/conflict until sustained present evidence, track expiration, or run end.

## Incident lifecycle

Suspected state is the candidate stage and emits no incident event. Confirmation creates
one `incident_opened` event and one run-local ID:

```text
<run-id>-track-<track-id>-<equipment>-<sequence>
```

Continued violation updates internal state without duplicate open events. Sustained present
evidence closes it once with `sustained_present`. Track expiration closes it with
`track_expired`; run termination closes any remaining active incident with `source_ended`.
The close event records final duration and evidence counts.

After close, the same source/track/equipment key enters a five-second cooldown. Internal
violation state continues to be recorded, but no duplicate incident opens during cooldown.
If explicit violation persists after cooldown, a new sequence ID may open. Different tracks
and helmet/vest are never deduplicated together.

## Track loss and timestamps

Buffered ByteTrack loss is supplied as unknown evidence on each processed timestamp. A
recovered ID resumes its state according to unknown grace. Expiration closes active
incidents and discards that anonymous track runtime. A later new track ID is never merged.

Timestamps must be finite, non-negative, and strictly increasing per track. Duplicate or
decreasing timestamps fail clearly rather than creating zero/negative durations.

## Outputs and visualization

Temporal mode adds `compliance_states.jsonl` and event-oriented `incidents.jsonl` to the
existing detections, tracks, associations, metadata, and annotated video. Visualization
uses `OK`, `UNKNOWN`, `CONFLICT`, `SUSPECT`, and `VIOLATION`; unknown is never labeled
safe. There are no zones, severity levels, alerts, or worker identities.

```powershell
.\.venv\Scripts\python.exe scripts\run_video_inference.py `
  --source path\to\video.mp4 `
  --enable-tracking `
  --enable-ppe-association `
  --enable-temporal-compliance `
  --output-dir results\video_inference\temporal_demo
```
