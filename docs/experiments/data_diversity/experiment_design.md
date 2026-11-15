# Controlled mixed-source target-domain adaptation experiment

## Hypothesis

Adding target-domain RF100 training scenes should improve held-out RF100 validation more
than merely repeating Construction-PPE training exposure, while preserving
Construction-PPE validation performance. For A/B, RF100 is unseen external development
data. For C, it is held-out target-domain validation and is not evidence of universal
unseen-domain generalization.

## Five-class task

The immutable task vocabulary is:

| ID | Class |
|---:|---|
| 0 | helmet |
| 1 | vest |
| 2 | Person |
| 3 | no_helmet |
| 4 | no_vest |

Construction-PPE IDs `0,2,5,6,11` map respectively to `0,1,2,3,4`. RF100 IDs
`0,4,3,1,2` map respectively to `0,1,2,3,4`. Non-target Construction-PPE annotations are
explicitly outside this task and are removed only from the derived labels.

Directly mixing RF100 with the original 12-class task is unsafe: RF100 does not annotate
the other seven classes, so real instances of those classes could be treated as background
and create contradictory supervision.

## Training conditions

- A — `experiment_cppe5_yolo26n_960`: unique Construction-PPE five-class train images.
- B — `experiment_cppe5_quantity_control_yolo26n_960`: the same Construction-PPE images
  repeated deterministically in a text list to match A+RF100 entries per epoch.
- C — `experiment_mixed5_yolo26n_960`: Construction-PPE train plus RF100 train.

Ultralytics 8.4.212 `BaseDataset.get_img_files` sorts text-list entries without converting
them to a set; therefore duplicate entries are preserved. The generated manifest records
both list length and unique count. B and C have equal entries per epoch. A intentionally
has fewer entries, while B distinguishes additional exposure from additional content.

All runs use the same `yolo26n.pt`, 960-pixel input, batch 8 unless a genuine CUDA OOM,
device 0, seed 42, deterministic mode, AMP, four workers, no cache, 100-epoch cap,
patience 20, and unchanged Ultralytics optimizer, learning-rate, and augmentation defaults.
Early stopping remains a possible residual exposure confound and will be reported using
actual epochs, image exposures, and optimizer-step callbacks.

## Predeclared success rule

The mixed-domain intervention is materially successful only if all conditions hold:

1. RF100-validation mAP50-95 improves at least +0.03 versus A.
2. RF100-validation mAP50-95 exceeds B by at least +0.02.
3. Construction-PPE-validation mAP50-95 regresses no more than -0.03 versus A.

False positives, localization errors, and `no_helmet`/`no_vest` behavior are qualitative
cross-checks. The numerical rule will not be changed after results are observed.

## Frozen external test

RF100 test is excluded from derivation, training, validation, failure analysis, and model
selection. Only RF100 train enters condition C; RF100 validation is OOD-development data.
