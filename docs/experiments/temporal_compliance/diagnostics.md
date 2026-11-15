# Deterministic temporal diagnostics

No real PPE video was available. These results use injected association states and exact
timestamps, independent of YOLO, ByteTrack, and geometry accuracy.

The reproducible scenario contains:

- Track 1: sustained positive helmet and vest evidence.
- Track 2: persistent `no_helmet`, a short unknown gap, conflict, resumed violation, then
  sustained helmet-present evidence.
- Track 3: a transient `no_vest` episode shorter than confirmation.

Expected and actual output matched exactly:

| Event | Expected | Actual |
|---|---:|---:|
| Track 2 helmet incident opened | 1.50 s | 1.50 s |
| Same incident closed | 4.00 s | 4.00 s |
| Total incidents opened | 1 | 1 |

The candidate began at 0.00 s. Unknown and conflict paused but did not add violation time.
The incident closed only after present evidence accumulated 1.00 second. Track 1 opened no
incident. Track 3's transient vest evidence opened no incident.

Automated tests additionally cover single-frame rejection, shorter-than-threshold
suspected state, deduplication, long unknown reset, brief positive evidence, conflict,
independent people and equipment, track expiration, cooldown suppression and reopening,
source-end closure, irregular timestamps, and invalid timestamp rejection.

See `diagnostic_timeline.md` for the human-readable state sequence and `metrics.json` for
the event snapshots and exact comparison. This is deterministic decision correctness, not
real-world accuracy.

## Limitations

Temporal stability cannot correct systematically wrong detector, tracker, or association
input. Track ID switches split state. Long unknown periods leave active incidents open
until positive evidence or explicit track/run termination because unknown is not proof of
compliance. Initial persistence values require real footage before calibration or safety
interpretation.
