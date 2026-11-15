# Qualitative review

## Judgment policy

Measurements come from JSONL and run metadata. The observations below are visual judgments
from selected review sheets, stated as “appears” rather than ground-truth labels. The CSV
manifest keeps formal `human_verdict` cells blank for independent review.

## Track continuity

- Both `compliant_multiworker` tracks appear visually stable from the first through final
  frame; each has 315 observations and no lost transition.
- `helmet_transition` Track 1 appears stable across the clip. Track 2 is a three-frame
  duplicate Person track over the same worker, which also creates an ambiguous association.
- `walking_multiworker` Track 4 appears consistent on the dark-clothed worker. Track 2 starts
  on that worker but later follows the red-vest worker, a probable ID switch during overlap
  or crossing. The 25 lost and 23 recovered transitions reinforce that continuity is the
  main weakness in this clip, although those counts alone do not prove identity errors.

## PPE association and temporal behavior

- Helmet and vest assignments in `compliant_multiworker` appear consistent in the sampled
  frames, and the system emits no incident.
- In `helmet_transition`, the helmet incident aligns visually with the worker removing the
  yellow helmet. The vest incident also appears consistent with no high-visibility vest in
  the sampled frames. These are plausible observations, not annotated correctness claims.
- In `walking_multiworker`, carried material repeatedly occludes the dark-clothed worker’s
  head. The resulting unknown and conflict states preserve uncertainty rather than declaring
  compliance. No confirmed incident is emitted.
- One sampled unassigned `no_vest` detection is visibly located on background safety tape,
  indicating a false PPE-class detection that association correctly leaves unassigned.

## Observed failure taxonomy

Only categories supported by actual output and visual review are listed:

- Probable ID switch: Track 2 changes apparent person in `walking_multiworker`.
- Duplicate/fragmented track: the three-frame Track 2 in `helmet_transition` overlaps the
  primary track on the same person.
- PPE left unassigned: 4 detections in `helmet_transition` and 125 in
  `walking_multiworker`.
- False PPE detection: sampled background safety tape is detected as `no_vest`.
- Conflict: 36 temporal conflict records in `walking_multiworker`, including simultaneous
  helmet/no-helmet evidence on an occluded worker.
- Unknown state under occlusion: 180 temporal records in `walking_multiworker`.
- False suspected violation appears possible where a visible helmet coexists with
  `no_helmet` evidence; no confirmed incident results in this clip.

No visually supported incorrect clearing or false confirmed incident was established.

## Candidate follow-ups—not applied

- Acquire independently annotated real footage for Person detection, track identity, PPE
  association, and incident-level evaluation.
- Evaluate appearance-assisted tracking or explicit crossing/occlusion tests for the
  observed probable ID switch; do not tune on these three clips alone.
- Investigate duplicate Person tracks and background `no_vest` false detections on a broader
  labeled set.
- Review unknown/conflict duration under real occlusion before changing grace thresholds.
- Profile decoding, high-resolution rendering, and encoding separately from inference.
- Improve overlay readability for portrait/4K footage without changing perception logic.
- Resolve video provenance and license before using these clips in a public portfolio.

Review artifacts are under ignored `results/real_video_validation/review_artifacts/`, and
the row-level inspection queue is in `review_manifest.csv`.
