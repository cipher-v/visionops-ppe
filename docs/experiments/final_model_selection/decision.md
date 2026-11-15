# Final candidate and calibration design

## Final candidate decision

Candidate C, the full-budget mixed-source YOLO26n at 960 pixels, is selected and frozen
for production v1. It has higher mAP50-95 than
D on Construction-PPE validation (`0.6642` versus `0.6409`) and leakage-filtered RF100
validation (`0.4948` versus `0.4722`) with identical architecture and inference cost. D
has similar fixed-threshold recall but lower confidence-ranked accuracy in both domains.
At the prior fixed operating point, C also has higher precision and fewer false positives:
Construction-PPE precision `0.9424` versus `0.9097` with 67 versus 104 false positives,
and filtered RF100 precision `0.8843` versus `0.7936` with 38 versus 56 false positives.
D records 11 more correct RF100 detections and 15 fewer complete misses at that operating
point, but this recall-side advantage does not outweigh C's higher AP on both domains,
lower false-positive burden, and identical deployment cost. Threshold calibration is
therefore performed only after the candidate decision and does not drive model selection.

YOLO26s is not selected because the earlier capacity experiment increased inference cost
without closing the external-domain gap. The original 12-class models are not selected
because RF100 supplies labels for only five common classes; mixing it into the 12-class
task would create contradictory background supervision. Production v1 is therefore the
five-class vocabulary `helmet`, `vest`, `Person`, `no_helmet`, `no_vest`.

No training is performed in Checkpoint 8.

## Calibration data policy

Calibration uses only Construction-PPE five-class validation and the leakage-filtered
118-image RF100 validation view. Construction-PPE is the source domain. RF100 is held-out
target-domain validation for C, not an unseen domain. Training data and RF100 test are
excluded. No pristine final unseen-domain set currently exists.

## Predeclared threshold procedure

- Confidence grid: inclusive `0.05` through `0.90` in steps of `0.01`.
- NMS IoU: fixed at `0.70`.
- Assignment: class-aware one-to-one matching at IoU `>= 0.50`.
- Global objective: mean of the two domain-level micro F1 scores.
- Exact ties: higher mean precision, then higher confidence threshold.
- Per-class candidates: mean domain F1 for all five classes.
- Violation diagnostics: separately report mean-domain F2 candidates for `no_helmet` and
  `no_vest`; F2 does not automatically become production policy.
- Small-sample guard: a `no_helmet` class threshold is adopted only if it improves both
  mean-domain F1 by at least `0.01` and source-domain F1 by at least `0.01`; otherwise the
  global threshold is retained for that class.

Class-specific policy is adopted only if it improves balanced micro F1 by at least `0.01`
or improves mean recall of either violation class by at least `0.05`, while combined false
positives grow no more than `20%` and neither domain micro F1 regresses more than `0.02`.
Otherwise the simpler global policy is frozen.

These rules are fixed before threshold results are generated.
