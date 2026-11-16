# Existing-video review queue

The generator reused Checkpoint 15/16 JSONL outputs for all three internal videos; it did
not rerun the detector. From 1,137 raw events, temporal dedup retained 102, visual dedup 75,
and budgets selected 28 (97.54% total reduction).

| Dimension | Selected composition |
|---|---|
| Type | 6 incident, 5 conflict, 10 detector uncertainty, 4 association, 3 drift |
| Equipment | helmet 11, vest 8, Person 2, no_helmet 2, no_vest 2, none 3 |
| Clip | helmet_transition 12, walking_multiworker 12, compliant_multiworker 4 |
| Priority | min 8.0, mean 29.78, max 40.0 |

Incident-linked candidates are 21.43%; drift-linked candidates are 10.71%. All 28 human
verdict/label/note fields are blank.

Normal rules naturally surfaced helmet incident confirmation/context, the ambiguous
helmet-transition duplicate-track region, and walking conflict frames around the overlap
and head occlusion (including frame 308). Three high-confidence unassigned PPE candidates
survived. The previously reviewed walking background `no_vest` example did not survive the
final priority/visual/budget filters; it was not hard-coded back into the queue. Drift
candidates are not claims of error.

