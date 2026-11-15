# Leakage-corrected exposure-control results

RF100 test was not read or evaluated. A/B see RF100 as an external development domain;
C/D use held-out RF100 target-domain validation.

## Leakage finding

The recomputed audit found 0 exact and 1 near duplicate. The single dHash-distance-0 validation copy has inconsistent classes and was excluded, producing 118 validation images. Source files remain unchanged.

## Legacy versus filtered RF100 validation

| Model | Legacy 119 mAP50-95 | Filtered 118 mAP50-95 | Delta |
|---|---:|---:|---:|
| cppe5_baseline | 0.3348 | 0.3349 | +0.0001 |
| quantity_control | 0.3157 | 0.3161 | +0.0004 |
| mixed_full_budget | 0.4936 | 0.4948 | +0.0012 |

## Training budget

| Model | Unique | Entries/epoch | Scheduler horizon | Completed | Exposures | Optimizer steps | Minutes | Best epoch | Stop reason |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cppe5_baseline | 800 | 800 | 100 | 100 | 80000 | 1304 | 55.7 | 83 | configured epoch horizon |
| quantity_control | 800 | 1797 | 100 | 45 | 80865 | 1389 | 55.7 | 26 | early stopping |
| mixed_exposure_matched | 1797 | 1797 | 100 | 45 | 80865 | 1389 | 85.7 | 43 | controlled stop after epoch 45 |
| mixed_full_budget | 1797 | 1797 | 100 | 100 | 179700 | 2936 | 115.2 | 88 | configured epoch horizon |

## Overall validation

| Model | Dataset | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| cppe5_baseline | cppe_validation | 0.9334 | 0.8206 | 0.8797 | 0.6691 |
| cppe5_baseline | rf100_filtered_validation | 0.6715 | 0.6648 | 0.6675 | 0.3349 |
| quantity_control | cppe_validation | 0.9412 | 0.8021 | 0.8710 | 0.6566 |
| quantity_control | rf100_filtered_validation | 0.6347 | 0.6664 | 0.6491 | 0.3161 |
| mixed_exposure_matched | cppe_validation | 0.9097 | 0.8120 | 0.8648 | 0.6409 |
| mixed_exposure_matched | rf100_filtered_validation | 0.7936 | 0.8149 | 0.8281 | 0.4722 |
| mixed_full_budget | cppe_validation | 0.9424 | 0.8131 | 0.8739 | 0.6642 |
| mixed_full_budget | rf100_filtered_validation | 0.8843 | 0.8184 | 0.8824 | 0.4948 |

## Complete five-class results

| Model | Dataset | Class | Instances | Precision | Recall | AP50 | AP50-95 |
|---|---|---|---:|---:|---:|---:|---:|
| cppe5_baseline | cppe_validation | helmet | 294 | 0.9496 | 0.8605 | 0.9093 | 0.7343 |
| cppe5_baseline | cppe_validation | vest | 340 | 0.9694 | 0.9308 | 0.9566 | 0.7807 |
| cppe5_baseline | cppe_validation | Person | 481 | 0.9620 | 0.9376 | 0.9733 | 0.8249 |
| cppe5_baseline | cppe_validation | no_helmet | 179 | 0.8744 | 0.6614 | 0.7711 | 0.4738 |
| cppe5_baseline | cppe_validation | no_vest | 130 | 0.9115 | 0.7127 | 0.7879 | 0.5318 |
| cppe5_baseline | rf100_filtered_validation | helmet | 231 | 0.8959 | 0.8312 | 0.8828 | 0.4228 |
| cppe5_baseline | rf100_filtered_validation | vest | 140 | 0.7776 | 0.8071 | 0.8185 | 0.4227 |
| cppe5_baseline | rf100_filtered_validation | Person | 240 | 0.8285 | 0.8500 | 0.8644 | 0.4919 |
| cppe5_baseline | rf100_filtered_validation | no_helmet | 11 | 0.0796 | 0.1818 | 0.0543 | 0.0226 |
| cppe5_baseline | rf100_filtered_validation | no_vest | 90 | 0.7759 | 0.6539 | 0.7175 | 0.3145 |
| quantity_control | cppe_validation | helmet | 294 | 0.9466 | 0.8605 | 0.9165 | 0.7252 |
| quantity_control | cppe_validation | vest | 340 | 0.9687 | 0.9059 | 0.9594 | 0.7692 |
| quantity_control | cppe_validation | Person | 481 | 0.9649 | 0.9144 | 0.9618 | 0.7901 |
| quantity_control | cppe_validation | no_helmet | 179 | 0.9140 | 0.6530 | 0.7477 | 0.4657 |
| quantity_control | cppe_validation | no_vest | 130 | 0.9118 | 0.6769 | 0.7693 | 0.5328 |
| quantity_control | rf100_filtered_validation | helmet | 231 | 0.8730 | 0.8334 | 0.8718 | 0.4082 |
| quantity_control | rf100_filtered_validation | vest | 140 | 0.7718 | 0.8000 | 0.8233 | 0.4146 |
| quantity_control | rf100_filtered_validation | Person | 240 | 0.7385 | 0.7917 | 0.7980 | 0.4361 |
| quantity_control | rf100_filtered_validation | no_helmet | 11 | 0.0684 | 0.2727 | 0.0685 | 0.0272 |
| quantity_control | rf100_filtered_validation | no_vest | 90 | 0.7218 | 0.6343 | 0.6840 | 0.2946 |
| mixed_exposure_matched | cppe_validation | helmet | 294 | 0.9393 | 0.8673 | 0.9134 | 0.7070 |
| mixed_exposure_matched | cppe_validation | vest | 340 | 0.9720 | 0.9196 | 0.9546 | 0.7612 |
| mixed_exposure_matched | cppe_validation | Person | 481 | 0.9394 | 0.9189 | 0.9573 | 0.7739 |
| mixed_exposure_matched | cppe_validation | no_helmet | 179 | 0.8441 | 0.6369 | 0.7187 | 0.4500 |
| mixed_exposure_matched | cppe_validation | no_vest | 130 | 0.8535 | 0.7173 | 0.7799 | 0.5126 |
| mixed_exposure_matched | rf100_filtered_validation | helmet | 231 | 0.8938 | 0.9091 | 0.9207 | 0.5100 |
| mixed_exposure_matched | rf100_filtered_validation | vest | 140 | 0.8819 | 0.9000 | 0.9385 | 0.5395 |
| mixed_exposure_matched | rf100_filtered_validation | Person | 240 | 0.8869 | 0.9667 | 0.9538 | 0.6286 |
| mixed_exposure_matched | rf100_filtered_validation | no_helmet | 11 | 0.4807 | 0.4545 | 0.4579 | 0.2257 |
| mixed_exposure_matched | rf100_filtered_validation | no_vest | 90 | 0.8249 | 0.8444 | 0.8695 | 0.4571 |
| mixed_full_budget | cppe_validation | helmet | 294 | 0.9693 | 0.8571 | 0.9140 | 0.7226 |
| mixed_full_budget | cppe_validation | vest | 340 | 0.9842 | 0.9176 | 0.9571 | 0.7800 |
| mixed_full_budget | cppe_validation | Person | 481 | 0.9672 | 0.9272 | 0.9585 | 0.8122 |
| mixed_full_budget | cppe_validation | no_helmet | 179 | 0.8714 | 0.6592 | 0.7666 | 0.4801 |
| mixed_full_budget | cppe_validation | no_vest | 130 | 0.9196 | 0.7044 | 0.7734 | 0.5260 |
| mixed_full_budget | rf100_filtered_validation | helmet | 231 | 0.8997 | 0.9319 | 0.9337 | 0.5180 |
| mixed_full_budget | rf100_filtered_validation | vest | 140 | 0.9172 | 0.8702 | 0.9229 | 0.5522 |
| mixed_full_budget | rf100_filtered_validation | Person | 240 | 0.9321 | 0.9667 | 0.9634 | 0.6498 |
| mixed_full_budget | rf100_filtered_validation | no_helmet | 11 | 0.8092 | 0.5455 | 0.7033 | 0.3022 |
| mixed_full_budget | rf100_filtered_validation | no_vest | 90 | 0.8631 | 0.7778 | 0.8889 | 0.4517 |

`no_helmet` and `no_vest` counts are shown explicitly. Sparse `no_helmet` results are not described as solved.

## Fixed-threshold error decomposition

| Model | Dataset | Correct | Localization | Classification | Miss | False positive |
|---|---|---:|---:|---:|---:|---:|
| cppe5_baseline | cppe_validation | 1262 | 21 | 19 | 122 | 89 |
| cppe5_baseline | rf100_filtered_validation | 578 | 36 | 23 | 75 | 109 |
| quantity_control | cppe_validation | 1248 | 29 | 24 | 123 | 97 |
| quantity_control | rf100_filtered_validation | 555 | 50 | 16 | 91 | 135 |
| mixed_exposure_matched | cppe_validation | 1242 | 25 | 19 | 138 | 104 |
| mixed_exposure_matched | rf100_filtered_validation | 642 | 13 | 10 | 47 | 56 |
| mixed_full_budget | cppe_validation | 1251 | 21 | 21 | 131 | 67 |
| mixed_full_budget | rf100_filtered_validation | 631 | 9 | 10 | 62 | 38 |

## Size-stratified diagnostic recall

| Model | Dataset | Size | Correct | Count | Recall |
|---|---|---|---:|---:|---:|
| cppe5_baseline | cppe_validation | tiny | 225 | 351 | 0.6410 |
| cppe5_baseline | cppe_validation | small | 374 | 404 | 0.9257 |
| cppe5_baseline | cppe_validation | large | 663 | 669 | 0.9910 |
| cppe5_baseline | rf100_filtered_validation | tiny | 3 | 8 | 0.3750 |
| cppe5_baseline | rf100_filtered_validation | small | 92 | 130 | 0.7077 |
| cppe5_baseline | rf100_filtered_validation | large | 483 | 574 | 0.8415 |
| quantity_control | cppe_validation | tiny | 216 | 351 | 0.6154 |
| quantity_control | cppe_validation | small | 373 | 404 | 0.9233 |
| quantity_control | cppe_validation | large | 659 | 669 | 0.9851 |
| quantity_control | rf100_filtered_validation | tiny | 3 | 8 | 0.3750 |
| quantity_control | rf100_filtered_validation | small | 88 | 130 | 0.6769 |
| quantity_control | rf100_filtered_validation | large | 464 | 574 | 0.8084 |
| mixed_exposure_matched | cppe_validation | tiny | 210 | 351 | 0.5983 |
| mixed_exposure_matched | cppe_validation | small | 372 | 404 | 0.9208 |
| mixed_exposure_matched | cppe_validation | large | 660 | 669 | 0.9865 |
| mixed_exposure_matched | rf100_filtered_validation | tiny | 3 | 8 | 0.3750 |
| mixed_exposure_matched | rf100_filtered_validation | small | 96 | 130 | 0.7385 |
| mixed_exposure_matched | rf100_filtered_validation | large | 543 | 574 | 0.9460 |
| mixed_full_budget | cppe_validation | tiny | 219 | 351 | 0.6239 |
| mixed_full_budget | cppe_validation | small | 372 | 404 | 0.9208 |
| mixed_full_budget | cppe_validation | large | 660 | 669 | 0.9865 |
| mixed_full_budget | rf100_filtered_validation | tiny | 3 | 8 | 0.3750 |
| mixed_full_budget | rf100_filtered_validation | small | 102 | 130 | 0.7846 |
| mixed_full_budget | rf100_filtered_validation | large | 526 | 574 | 0.9164 |

Full per-class size cells and counts remain in `metrics.json`.

## Required interpretation

1. Removing the duplicate changed A/B/C RF100 mAP50-95 only by the deltas reported above; interpretation is based on the filtered values.
2. D changes filtered RF100 mAP50-95 by +0.1373 versus A and +0.1560 versus B at matched exposure.
3. D changes Construction-PPE mAP50-95 by -0.0282 versus A.
4. Full-budget C minus D is +0.0226 on filtered RF100 and +0.0232 on Construction-PPE.
5. Against A, D changes correct assignments by +64, localization failures by -23, confusions by -13, misses by -28, and false positives by -53. Against B, D changes false positives by -79 and misses by -44.
6. On filtered RF100, D `no_helmet` has n=11, precision 0.4807, recall 0.4545, AP50 0.4579, and AP50-95 0.2257; this cell is too small to call solved. D `no_vest` has n=90, precision 0.8249, recall 0.8444, AP50 0.8695, and AP50-95 0.4571.
7. Remaining limitations include one training seed, a small filtered RF100 `no_helmet` count, RF100 being a target domain for C/D, and RF100 test having influenced the earlier overall research path. B and D both recorded exactly 1,389 optimizer steps, so no optimizer-step mismatch remains in the primary matched-exposure comparison.
8. **Decision:** All predeclared matched-exposure criteria pass; target-domain data diversity remains strongly supported.

The result is evidence from a controlled development experiment, not perfect causal proof or universal-domain generalization. D captures most of C's target-domain gain at the same exposure as B, while C's additional 55 epochs add a comparatively modest +0.0226 RF100 mAP50-95 and recover +0.0232 CPPE mAP50-95. Target-domain data content is therefore the strongest supported explanation, with additional optimization exposure providing a secondary benefit.

## Scheduler-control evidence

D recorded 45 epochs against the first 45 C epochs. Excluding wall-clock time, exact metric/LR trajectory match: **True** (0 mismatched epochs). The training metadata also records the 100-epoch scheduler horizon, epoch-45 controlled stop, result-row count, absence of epoch 46, and unreached close-mosaic phase.
