# Controlled mixed-source target-domain adaptation results

RF100 test remained frozen and was not read. RF100 validation is unseen external
development data for A/B and held-out target-domain validation for C.

## Derived datasets

| Dataset | Split | Images | Retained boxes | Removed out-of-task boxes |
|---|---|---:|---:|---:|
| Construction-PPE | train | 800 | 5791 | 6111 |
| Construction-PPE | validation | 200 | 1424 | 1537 |
| RF100 | train | 997 | 6386 | 0 |
| RF100 | validation | 119 | 715 | 0 |

## Optimization budget

| Condition | Unique images | Entries/epoch | Epochs | Image exposures | Optimizer steps | Minutes |
|---|---:|---:|---:|---:|---:|---:|
| cppe5_baseline | 800 | 800 | 100 | 80000 | 1304 | 55.7 |
| quantity_control | 800 | 1797 | 45 | 80865 | 1389 | 55.7 |
| mixed_domain | 1797 | 1797 | 100 | 179700 | 2936 | 115.2 |

The A-run optimizer-step count is reconstructed from the recorded deterministic
Ultralytics accumulation schedule because its callback was initially registered too late.
The B and C counts are measured by runtime callbacks.

## Overall validation

| Model | Dataset | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| cppe5_baseline | cppe_validation | 0.9334 | 0.8206 | 0.8797 | 0.6691 |
| cppe5_baseline | rf100_validation | 0.6608 | 0.6693 | 0.6670 | 0.3348 |
| quantity_control | cppe_validation | 0.9412 | 0.8021 | 0.8710 | 0.6566 |
| quantity_control | rf100_validation | 0.6354 | 0.6668 | 0.6499 | 0.3157 |
| mixed_domain | cppe_validation | 0.9424 | 0.8131 | 0.8739 | 0.6642 |
| mixed_domain | rf100_validation | 0.8845 | 0.8162 | 0.8814 | 0.4936 |

## Complete per-class validation

| Model | Dataset | Class | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---:|---:|---:|---:|
| cppe5_baseline | cppe_validation | helmet | 0.9496 | 0.8605 | 0.9093 | 0.7343 |
| cppe5_baseline | cppe_validation | vest | 0.9694 | 0.9308 | 0.9566 | 0.7807 |
| cppe5_baseline | cppe_validation | Person | 0.9620 | 0.9376 | 0.9733 | 0.8249 |
| cppe5_baseline | cppe_validation | no_helmet | 0.8744 | 0.6614 | 0.7711 | 0.4738 |
| cppe5_baseline | cppe_validation | no_vest | 0.9115 | 0.7127 | 0.7879 | 0.5318 |
| cppe5_baseline | rf100_validation | helmet | 0.8856 | 0.8345 | 0.8734 | 0.4209 |
| cppe5_baseline | rf100_validation | vest | 0.7568 | 0.8085 | 0.8206 | 0.4226 |
| cppe5_baseline | rf100_validation | Person | 0.8133 | 0.8548 | 0.8652 | 0.4915 |
| cppe5_baseline | rf100_validation | no_helmet | 0.0756 | 0.1818 | 0.0543 | 0.0226 |
| cppe5_baseline | rf100_validation | no_vest | 0.7729 | 0.6667 | 0.7217 | 0.3163 |
| quantity_control | cppe_validation | helmet | 0.9466 | 0.8605 | 0.9165 | 0.7252 |
| quantity_control | cppe_validation | vest | 0.9687 | 0.9059 | 0.9594 | 0.7692 |
| quantity_control | cppe_validation | Person | 0.9649 | 0.9144 | 0.9618 | 0.7901 |
| quantity_control | cppe_validation | no_helmet | 0.9140 | 0.6530 | 0.7477 | 0.4657 |
| quantity_control | cppe_validation | no_vest | 0.9118 | 0.6769 | 0.7693 | 0.5328 |
| quantity_control | rf100_validation | helmet | 0.8775 | 0.8339 | 0.8729 | 0.4092 |
| quantity_control | rf100_validation | vest | 0.7737 | 0.8014 | 0.8246 | 0.4105 |
| quantity_control | rf100_validation | Person | 0.7354 | 0.7925 | 0.7984 | 0.4373 |
| quantity_control | rf100_validation | no_helmet | 0.0687 | 0.2727 | 0.0686 | 0.0272 |
| quantity_control | rf100_validation | no_vest | 0.7216 | 0.6334 | 0.6851 | 0.2942 |
| mixed_domain | cppe_validation | helmet | 0.9693 | 0.8571 | 0.9140 | 0.7226 |
| mixed_domain | cppe_validation | vest | 0.9842 | 0.9176 | 0.9571 | 0.7800 |
| mixed_domain | cppe_validation | Person | 0.9672 | 0.9272 | 0.9585 | 0.8122 |
| mixed_domain | cppe_validation | no_helmet | 0.8714 | 0.6592 | 0.7666 | 0.4801 |
| mixed_domain | cppe_validation | no_vest | 0.9196 | 0.7044 | 0.7734 | 0.5260 |
| mixed_domain | rf100_validation | helmet | 0.9022 | 0.9310 | 0.9335 | 0.5168 |
| mixed_domain | rf100_validation | vest | 0.9172 | 0.8642 | 0.9256 | 0.5461 |
| mixed_domain | rf100_validation | Person | 0.9320 | 0.9627 | 0.9635 | 0.6514 |
| mixed_domain | rf100_validation | no_helmet | 0.8082 | 0.5455 | 0.6973 | 0.2994 |
| mixed_domain | rf100_validation | no_vest | 0.8629 | 0.7778 | 0.8873 | 0.4542 |

## Fixed-threshold error decomposition

Confidence 0.25, NMS IoU 0.7, match IoU 0.5, localization floor 0.1.

| Model | Dataset | Correct | Localization | Classification | Miss | False positive |
|---|---|---:|---:|---:|---:|---:|
| cppe5_baseline | cppe_validation | 1262 | 21 | 19 | 122 | 89 |
| cppe5_baseline | rf100_validation | 581 | 36 | 23 | 75 | 109 |
| quantity_control | cppe_validation | 1248 | 29 | 24 | 123 | 97 |
| quantity_control | rf100_validation | 558 | 50 | 16 | 91 | 135 |
| mixed_domain | cppe_validation | 1251 | 21 | 21 | 131 | 67 |
| mixed_domain | rf100_validation | 633 | 9 | 10 | 63 | 38 |

Size-stratified recall, including every per-class cell count, is recorded in `metrics.json`.

## Size-stratified diagnostic recall

| Model | Dataset | Size | Correct | GT count | Recall |
|---|---|---|---:|---:|---:|
| cppe5_baseline | cppe_validation | tiny | 225 | 351 | 0.6410 |
| cppe5_baseline | cppe_validation | small | 374 | 404 | 0.9257 |
| cppe5_baseline | cppe_validation | large | 663 | 669 | 0.9910 |
| cppe5_baseline | rf100_validation | tiny | 3 | 8 | 0.3750 |
| cppe5_baseline | rf100_validation | small | 92 | 130 | 0.7077 |
| cppe5_baseline | rf100_validation | large | 486 | 577 | 0.8423 |
| quantity_control | cppe_validation | tiny | 216 | 351 | 0.6154 |
| quantity_control | cppe_validation | small | 373 | 404 | 0.9233 |
| quantity_control | cppe_validation | large | 659 | 669 | 0.9851 |
| quantity_control | rf100_validation | tiny | 3 | 8 | 0.3750 |
| quantity_control | rf100_validation | small | 88 | 130 | 0.6769 |
| quantity_control | rf100_validation | large | 467 | 577 | 0.8094 |
| mixed_domain | cppe_validation | tiny | 219 | 351 | 0.6239 |
| mixed_domain | cppe_validation | small | 372 | 404 | 0.9208 |
| mixed_domain | cppe_validation | large | 660 | 669 | 0.9865 |
| mixed_domain | rf100_validation | tiny | 3 | 8 | 0.3750 |
| mixed_domain | rf100_validation | small | 102 | 130 | 0.7846 |
| mixed_domain | rf100_validation | large | 528 | 577 | 0.9151 |

### Per-class size cells

Small cells are retained with their counts and must not be overinterpreted.

| Model | Dataset | Class | Size | Correct | GT count | Recall |
|---|---|---|---|---:|---:|---:|
| cppe5_baseline | cppe_validation | helmet | tiny | 88 | 124 | 0.7097 |
| cppe5_baseline | cppe_validation | helmet | small | 75 | 78 | 0.9615 |
| cppe5_baseline | cppe_validation | helmet | large | 92 | 92 | 1.0000 |
| cppe5_baseline | cppe_validation | vest | tiny | 51 | 64 | 0.7969 |
| cppe5_baseline | cppe_validation | vest | small | 111 | 114 | 0.9737 |
| cppe5_baseline | cppe_validation | vest | large | 160 | 162 | 0.9877 |
| cppe5_baseline | cppe_validation | Person | tiny | 34 | 46 | 0.7391 |
| cppe5_baseline | cppe_validation | Person | small | 132 | 140 | 0.9429 |
| cppe5_baseline | cppe_validation | Person | large | 294 | 295 | 0.9966 |
| cppe5_baseline | cppe_validation | no_helmet | tiny | 44 | 84 | 0.5238 |
| cppe5_baseline | cppe_validation | no_helmet | small | 38 | 46 | 0.8261 |
| cppe5_baseline | cppe_validation | no_helmet | large | 47 | 49 | 0.9592 |
| cppe5_baseline | cppe_validation | no_vest | tiny | 8 | 33 | 0.2424 |
| cppe5_baseline | cppe_validation | no_vest | small | 18 | 26 | 0.6923 |
| cppe5_baseline | cppe_validation | no_vest | large | 70 | 71 | 0.9859 |
| cppe5_baseline | rf100_validation | helmet | tiny | 3 | 5 | 0.6000 |
| cppe5_baseline | rf100_validation | helmet | small | 77 | 94 | 0.8191 |
| cppe5_baseline | rf100_validation | helmet | large | 116 | 133 | 0.8722 |
| cppe5_baseline | rf100_validation | vest | tiny | 0 | 0 | n/a |
| cppe5_baseline | rf100_validation | vest | small | 5 | 8 | 0.6250 |
| cppe5_baseline | rf100_validation | vest | large | 108 | 133 | 0.8120 |
| cppe5_baseline | rf100_validation | Person | tiny | 0 | 2 | 0.0000 |
| cppe5_baseline | rf100_validation | Person | small | 3 | 8 | 0.3750 |
| cppe5_baseline | rf100_validation | Person | large | 209 | 231 | 0.9048 |
| cppe5_baseline | rf100_validation | no_helmet | tiny | 0 | 0 | n/a |
| cppe5_baseline | rf100_validation | no_helmet | small | 0 | 9 | 0.0000 |
| cppe5_baseline | rf100_validation | no_helmet | large | 0 | 2 | 0.0000 |
| cppe5_baseline | rf100_validation | no_vest | tiny | 0 | 1 | 0.0000 |
| cppe5_baseline | rf100_validation | no_vest | small | 7 | 11 | 0.6364 |
| cppe5_baseline | rf100_validation | no_vest | large | 53 | 78 | 0.6795 |
| quantity_control | cppe_validation | helmet | tiny | 90 | 124 | 0.7258 |
| quantity_control | cppe_validation | helmet | small | 73 | 78 | 0.9359 |
| quantity_control | cppe_validation | helmet | large | 92 | 92 | 1.0000 |
| quantity_control | cppe_validation | vest | tiny | 48 | 64 | 0.7500 |
| quantity_control | cppe_validation | vest | small | 113 | 114 | 0.9912 |
| quantity_control | cppe_validation | vest | large | 158 | 162 | 0.9753 |
| quantity_control | cppe_validation | Person | tiny | 32 | 46 | 0.6957 |
| quantity_control | cppe_validation | Person | small | 130 | 140 | 0.9286 |
| quantity_control | cppe_validation | Person | large | 290 | 295 | 0.9831 |
| quantity_control | cppe_validation | no_helmet | tiny | 41 | 84 | 0.4881 |
| quantity_control | cppe_validation | no_helmet | small | 39 | 46 | 0.8478 |
| quantity_control | cppe_validation | no_helmet | large | 49 | 49 | 1.0000 |
| quantity_control | cppe_validation | no_vest | tiny | 5 | 33 | 0.1515 |
| quantity_control | cppe_validation | no_vest | small | 18 | 26 | 0.6923 |
| quantity_control | cppe_validation | no_vest | large | 70 | 71 | 0.9859 |
| quantity_control | rf100_validation | helmet | tiny | 3 | 5 | 0.6000 |
| quantity_control | rf100_validation | helmet | small | 73 | 94 | 0.7766 |
| quantity_control | rf100_validation | helmet | large | 119 | 133 | 0.8947 |
| quantity_control | rf100_validation | vest | tiny | 0 | 0 | n/a |
| quantity_control | rf100_validation | vest | small | 5 | 8 | 0.6250 |
| quantity_control | rf100_validation | vest | large | 107 | 133 | 0.8045 |
| quantity_control | rf100_validation | Person | tiny | 0 | 2 | 0.0000 |
| quantity_control | rf100_validation | Person | small | 1 | 8 | 0.1250 |
| quantity_control | rf100_validation | Person | large | 190 | 231 | 0.8225 |
| quantity_control | rf100_validation | no_helmet | tiny | 0 | 0 | n/a |
| quantity_control | rf100_validation | no_helmet | small | 4 | 9 | 0.4444 |
| quantity_control | rf100_validation | no_helmet | large | 0 | 2 | 0.0000 |
| quantity_control | rf100_validation | no_vest | tiny | 0 | 1 | 0.0000 |
| quantity_control | rf100_validation | no_vest | small | 5 | 11 | 0.4545 |
| quantity_control | rf100_validation | no_vest | large | 51 | 78 | 0.6538 |
| mixed_domain | cppe_validation | helmet | tiny | 90 | 124 | 0.7258 |
| mixed_domain | cppe_validation | helmet | small | 75 | 78 | 0.9615 |
| mixed_domain | cppe_validation | helmet | large | 92 | 92 | 1.0000 |
| mixed_domain | cppe_validation | vest | tiny | 51 | 64 | 0.7969 |
| mixed_domain | cppe_validation | vest | small | 111 | 114 | 0.9737 |
| mixed_domain | cppe_validation | vest | large | 159 | 162 | 0.9815 |
| mixed_domain | cppe_validation | Person | tiny | 31 | 46 | 0.6739 |
| mixed_domain | cppe_validation | Person | small | 129 | 140 | 0.9214 |
| mixed_domain | cppe_validation | Person | large | 291 | 295 | 0.9864 |
| mixed_domain | cppe_validation | no_helmet | tiny | 42 | 84 | 0.5000 |
| mixed_domain | cppe_validation | no_helmet | small | 40 | 46 | 0.8696 |
| mixed_domain | cppe_validation | no_helmet | large | 49 | 49 | 1.0000 |
| mixed_domain | cppe_validation | no_vest | tiny | 5 | 33 | 0.1515 |
| mixed_domain | cppe_validation | no_vest | small | 17 | 26 | 0.6538 |
| mixed_domain | cppe_validation | no_vest | large | 69 | 71 | 0.9718 |
| mixed_domain | rf100_validation | helmet | tiny | 3 | 5 | 0.6000 |
| mixed_domain | rf100_validation | helmet | small | 86 | 94 | 0.9149 |
| mixed_domain | rf100_validation | helmet | large | 125 | 133 | 0.9398 |
| mixed_domain | rf100_validation | vest | tiny | 0 | 0 | n/a |
| mixed_domain | rf100_validation | vest | small | 2 | 8 | 0.2500 |
| mixed_domain | rf100_validation | vest | large | 117 | 133 | 0.8797 |
| mixed_domain | rf100_validation | Person | tiny | 0 | 2 | 0.0000 |
| mixed_domain | rf100_validation | Person | small | 5 | 8 | 0.6250 |
| mixed_domain | rf100_validation | Person | large | 226 | 231 | 0.9784 |
| mixed_domain | rf100_validation | no_helmet | tiny | 0 | 0 | n/a |
| mixed_domain | rf100_validation | no_helmet | small | 2 | 9 | 0.2222 |
| mixed_domain | rf100_validation | no_helmet | large | 2 | 2 | 1.0000 |
| mixed_domain | rf100_validation | no_vest | tiny | 0 | 1 | 0.0000 |
| mixed_domain | rf100_validation | no_vest | small | 7 | 11 | 0.6364 |
| mixed_domain | rf100_validation | no_vest | large | 58 | 78 | 0.7436 |

## Required conclusions

1. **Does mixed-domain training improve RF100 validation?** Yes: mAP50-95 changed from 0.3348 to 0.4936 (+0.1588).
2. **Does it preserve Construction-PPE validation?** Yes under the predeclared tolerance: mAP50-95 changed from 0.6691 to 0.6642 (-0.0049).
3. **Does the quantity control achieve similar improvements?** Its RF100 result is 0.3157, a change of -0.0191 from A; mixed exceeds it by +0.1779.
4. **Is improvement attributable to diversity rather than only exposure?** The numerical control supports diversity. However, B early-stopped after 45 epochs while C ran 100, so C had more total exposures and optimizer steps; this residual confound prevents a strict causal claim that all gain came from diversity alone.
5. **Which failure types improve?** RF100 mixed-minus-baseline count changes are: false_positive -71, localization_failure -27, classification_confusion -13, complete_miss -12, correct +52. Negative error changes are improvements; positive correct change is an improvement.
6. **Which classes benefit most?** RF100 AP50-95 changes, greatest first: no_helmet +0.2768, Person +0.1599, no_vest +0.1380, vest +0.1235, helmet +0.0959.
7. **What regressions appear?** No RF100 class has a negative AP50-95 change. In-domain and size-cell changes remain visible above.
8. **Is this strong enough for the final model stage?** Yes: all predeclared material-success conditions pass.

The representative panels are deterministically selected from RF100 validation only. Blue boxes are ground truth and red boxes are predictions. Selection scores are recorded in `metrics.json`; a non-positive score means the requested failure category had no strict improvement/regression example and the closest deterministic case is shown rather than overstated.

The numerical rule and qualitative diagnostics are interpreted separately; disagreement is
reported rather than used to alter the predeclared threshold.

The quantity control matched the mixed condition's entries per epoch and nearly matched
the A baseline's total exposure, but early stopping prevented total-budget matching to C.
This was the predeclared residual confound; training was not extended after results.
