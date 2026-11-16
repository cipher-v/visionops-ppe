# Tracker comparison

## Candidates

All candidates use the frozen Person threshold (`0.41`), identical detector outputs, frame
stride 1, and unchanged PPE association and temporal logic.

| Candidate | Implementation | ReID | Configuration |
|---|---|---|---|
| ByteTrack baseline | Ultralytics `BYTETracker` | No | Frozen `bytetrack_person.yaml` |
| BoT-SORT | Ultralytics `BOTSORT` | No | Official/default BoT-SORT values |
| BoT-SORT + ReID | Ultralytics `BOTSORT` | `yolo26n-cls.pt` | Same values plus external official encoder |

No thresholds, buffers, GMC method, proximity threshold, appearance threshold, or encoder
setting was swept. The ReID weight SHA-256 is
`0DD6F8DBC448870AC98A3CBB7156F923F7CE21FED3755D4019169FFFFD279E81` and remains ignored.

## Stable compliant clip

All candidates produced exactly two 315-observation tracks, 630 total observations, zero
lost/recovered/expired transitions, identical compliant temporal output, and zero incidents.
No stable-clip regression was observed.

## Helmet-transition duplicate

| Tracker | Track lengths | Lost/recovered/expired | Visual result |
|---|---|---|---|
| ByteTrack | 295, 3 | 3 / 2 / 1 | Three-frame duplicate remains |
| BoT-SORT | 295, 3 | 2 / 2 / 0 | Same duplicate remains |
| BoT-SORT + ReID | 295, 3 | 2 / 2 / 0 | Same duplicate remains |

In every candidate, Track 2 overlaps Track 1 at frame 106, is the returned track at frames
108-109, and Track 1 returns at frame 110. Neither BoT-SORT variant fixes this failure.

## Walking overlap/crossing

| Tracker | Observations by track | Lost/recovered/expired | Visual result |
|---|---|---|---|
| ByteTrack | 153, 545, 37, 263 | 25 / 23 / 2 | Probable identity transfer |
| BoT-SORT | 574, 124, 37, 263 | 26 / 24 / 2 | Transfer avoided; dark worker still fragments |
| BoT-SORT + ReID | 577, 121, 37, 263 | 22 / 21 / 1 | Same qualitative result as non-ReID |

The baseline Track 2 begins on the dark-clothed worker and later follows the red-vest
worker; Track 4 then starts on the dark-clothed worker. Under both BoT-SORT variants, Track
1 remains on the red-vest worker across the clip while Track 4 starts on the dark-clothed
worker after its earlier track is lost. This prevents state transfer between workers but
does not solve dark-worker fragmentation. ReID adds no visible benefit in the predeclared
window.

These are qualitative observations, not a formal ID-switch rate.

## Downstream impact

- `compliant_multiworker`: all candidates produce 1,260 compliant states and no incidents.
- `helmet_transition`: all candidates produce the same two incidents—vest confirmed at
  1.520 seconds and helmet confirmed at 5.560 seconds, both closed at 11.840 seconds with
  `source_ended`. Compliance-state counts are identical.
- `walking_multiworker`: ByteTrack and non-ReID BoT-SORT produce identical temporal counts
  (1,691 compliant, 180 unknown, 89 suspected, 36 conflict) and zero incidents. ReID changes
  one record from compliant to unknown (1,690/181) and still produces zero incidents.
- Non-ReID BoT-SORT assigns 1,764 PPE detections on the walking clip versus 1,766 for
  ByteTrack; the two-record difference is vest association during track availability.

No tracker fragments or transfers a confirmed incident in these clips. The BoT-SORT
walking identity improvement therefore reduces a risk that was visible qualitatively but
does not change emitted incident counts in this small sample.

## Decision

Select **BoT-SORT without ReID** in `configs/production_tracker.yaml`.

It materially improves the primary walking identity-transfer failure, preserves the stable
clip, introduces no additional unique-track count, and leaves downstream incident behavior
sensible. ReID is not selected because it does not improve either known failure and adds
substantial tracking cost. ByteTrack remains unchanged as experiment lineage and rollback.
