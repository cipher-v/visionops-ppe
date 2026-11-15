# Operating-threshold calibration

## Frozen setup

Candidate C is frozen as production v1. RF100 test was not used. Calibration uses source-domain Construction-PPE validation and held-out target-domain filtered RF100 validation. Predictions were generated once at confidence 0.05, NMS IoU 0.7; every threshold uses class-aware one-to-one IoU 0.5 assignment.

## Global threshold

The domain-balanced micro-F1 optimum is **0.27**. Ties were resolved by mean precision then higher threshold.

## Per-class candidates

| Class | F1 threshold | Balanced F1 | F2 threshold | Balanced F2 |
|---|---:|---:|---:|---:|
| helmet | 0.32 | 0.9190 | -- | -- |
| vest | 0.19 | 0.9252 | -- | -- |
| Person | 0.41 | 0.9407 | -- | -- |
| no_helmet | 0.11 | 0.6750 | 0.11 | 0.6593 |
| no_vest | 0.11 | 0.7961 | 0.05 | 0.7835 |

F2 thresholds for violations are diagnostic alternatives, not automatic production choices. Filtered RF100 contains only 11 `no_helmet` instances; the small-sample guard result is `False`.

## Violation-class trade-offs

| Class | Objective | Domain | Instances | Threshold | Precision | Recall | F1 | F2 | FP | FN |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_helmet | F1 candidate | cppe_validation | 179 | 0.11 | 0.6171 | 0.7654 | 0.6833 | 0.7303 | 85 | 42 |
| no_helmet | F1 candidate | rf100_filtered_validation | 11 | 0.11 | 0.8571 | 0.5455 | 0.6667 | 0.5882 | 1 | 5 |
| no_helmet | F2 candidate | cppe_validation | 179 | 0.11 | 0.6171 | 0.7654 | 0.6833 | 0.7303 | 85 | 42 |
| no_helmet | F2 candidate | rf100_filtered_validation | 11 | 0.11 | 0.8571 | 0.5455 | 0.6667 | 0.5882 | 1 | 5 |
| no_vest | F1 candidate | cppe_validation | 130 | 0.11 | 0.8017 | 0.7154 | 0.7561 | 0.7311 | 23 | 37 |
| no_vest | F1 candidate | rf100_filtered_validation | 90 | 0.11 | 0.8506 | 0.8222 | 0.8362 | 0.8277 | 13 | 16 |
| no_vest | F2 candidate | cppe_validation | 130 | 0.05 | 0.7402 | 0.7231 | 0.7315 | 0.7264 | 33 | 36 |
| no_vest | F2 candidate | rf100_filtered_validation | 90 | 0.05 | 0.7857 | 0.8556 | 0.8191 | 0.8406 | 21 | 13 |

The `no_helmet` F1 candidate is not adopted because it loses source-domain F1 and the
11-instance RF100 result cannot establish stability. The F2 choices are shown as
review-cost alternatives only; neither is silently substituted into production.

## Global policy

| Domain | Precision | Recall | F1 | FP | FN | no_helmet R | no_helmet FP | no_vest R | no_vest FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cppe_validation | 0.9239 | 0.8785 | 0.9006 | 103 | 173 | 0.7318 | 31 | 0.7000 | 16 |
| rf100_filtered_validation | 0.9170 | 0.8848 | 0.9006 | 57 | 82 | 0.3636 | 1 | 0.7222 | 11 |

## Candidate class-specific policy

| Domain | Precision | Recall | F1 | FP | FN | no_helmet R | no_helmet FP | no_vest R | no_vest FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cppe_validation | 0.9299 | 0.8757 | 0.9020 | 94 | 177 | 0.7318 | 31 | 0.7154 | 23 |
| rf100_filtered_validation | 0.9217 | 0.8933 | 0.9073 | 54 | 76 | 0.3636 | 1 | 0.8222 | 13 |

Complexity-gate result: `True`. Final policy: **class_specific**. The gate requires a material balanced-F1 or violation-recall gain, FP growth <=20%, and no domain F1 regression beyond 0.02.

## Final operating results

| Domain | Scope | Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cppe_validation | overall | -- | 0.9299 | 0.8757 | 0.9020 | 1247 | 94 | 177 |
| cppe_validation | helmet | 0.32 | 0.9590 | 0.8741 | 0.9146 | 257 | 11 | 37 |
| cppe_validation | vest | 0.19 | 0.9612 | 0.9471 | 0.9541 | 322 | 13 | 18 |
| cppe_validation | Person | 0.41 | 0.9652 | 0.9231 | 0.9437 | 444 | 16 | 37 |
| cppe_validation | no_helmet | 0.27 | 0.8086 | 0.7318 | 0.7683 | 131 | 31 | 48 |
| cppe_validation | no_vest | 0.11 | 0.8017 | 0.7154 | 0.7561 | 93 | 23 | 37 |
| rf100_filtered_validation | overall | -- | 0.9217 | 0.8933 | 0.9073 | 636 | 54 | 76 |
| rf100_filtered_validation | helmet | 0.32 | 0.9336 | 0.9134 | 0.9234 | 211 | 15 | 20 |
| rf100_filtered_validation | vest | 0.19 | 0.9308 | 0.8643 | 0.8963 | 121 | 9 | 19 |
| rf100_filtered_validation | Person | 0.41 | 0.9339 | 0.9417 | 0.9378 | 226 | 16 | 14 |
| rf100_filtered_validation | no_helmet | 0.27 | 0.8000 | 0.3636 | 0.5000 | 4 | 1 | 7 |
| rf100_filtered_validation | no_vest | 0.11 | 0.8506 | 0.8222 | 0.8362 | 74 | 13 | 16 |

Confidence calibration does not change mAP. Frozen candidate C retains Construction-PPE mAP50 `0.8739`, mAP50-95 `0.6642`, and filtered RF100 mAP50 `0.8824`, mAP50-95 `0.4948`.

## Detector policy

Detections below their configured threshold are ignored. These outputs are detector-level evidence only. A one-frame `no_helmet` or `no_vest` detection is not an incident; tracking, person-PPE association, temporal persistence, dwell time, and cooldown remain explicitly deferred.
