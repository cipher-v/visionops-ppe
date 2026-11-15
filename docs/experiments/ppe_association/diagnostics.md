# PPE association diagnostics

These diagnostics use known injected geometry, not YOLO output or real-world video. Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_ppe_association_diagnostics.py
```

## Required deterministic cases

Automated tests verify:

- a head-centered helmet assigns correctly;
- a torso-centered vest assigns correctly;
- PPE outside a Person is unassigned;
- an ambiguous PPE box receives exactly one deterministic owner;
- two people receive their spatially distinct PPE independently;
- simultaneous `helmet` and `no_helmet` produce `conflict`;
- missing PPE produces `unknown`, never `violation`;
- malformed and zero-area PPE is rejected; and
- multiple same-class matches remain recorded while one primary item is selected.

## Crowded synthetic scene

Three partially overlapping Person boxes were supplied with three helmets, three vests,
one boundary `no_helmet`, and one outside `no_vest`. Expected owner or unassigned status is
known for all eight detections.

- Correct known decisions: `8/8` (`100%` synthetic decision accuracy)
- Assigned PPE: `7`
- Unassigned PPE: `1`
- Ambiguous detections: `1`
- Mean accepted association score: `0.9304`
- Assignment counts: helmet `3`, vest `3`, no_helmet `1`
- Helmet states: present `2`, conflict `1`
- Vest states: present `3`

The ambiguous boundary `no_helmet` had two valid candidates. The deterministic score/tie
policy assigned it once to track 1 and retained track 2 as an alternative. This also made
track 1's helmet state `conflict`, as intended by the synthetic ground truth.

Complete scores, alternatives, and records are in `diagnostic_results.json`.

## Failure modes and limitations

Geometry cannot resolve every crowded or occluded scene. Loose Person boxes, unusual pose,
perspective, PPE carried in a hand, partial bodies, and overlapping workers can cause an
incorrect or unassigned match. The priors are not learned, and synthetic 100% accuracy is
not evidence of real-world quality. No suitable local real PPE video exists, so deployment
video validation is still required before quality claims or threshold tuning.
