# External generalization comparison

## Protocol

Both frozen best checkpoints were evaluated at 960 pixels on the same five mapped classes.
The official leakage-free 90-image RF100 test split is primary. Validation AP uses the
Ultralytics defaults; diagnostic recall alone uses fixed confidence 0.25, NMS IoU
0.7, and class-aware match IoU 0.5. No training, fine-tuning, threshold
tuning, annotation repair, or augmentation change was performed.

## Primary external test results

| Model | Precision | Recall | mAP50 | mAP50-95 | mAP50-95 gap vs internal |
|---|---:|---:|---:|---:|---:|
| yolo26n | 0.6730 | 0.7044 | 0.6908 | 0.3029 | -0.3582 |
| yolo26s | 0.6474 | 0.7177 | 0.6823 | 0.3012 | -0.3859 |

External YOLO26s minus YOLO26n: precision -0.0256, recall
+0.0133, mAP50 -0.0085, and mAP50-95
-0.0016.

## Interpretation

The generalization gap compares each external primary result with the arithmetic mean of
the same five classes on Construction-PPE validation, avoiding contamination by the seven
non-overlapping classes. The official test set is independent but only 90 images; the
secondary all-image metrics in `metrics.json` are descriptive and not independent.

Size-stratified recall and representative correct, false-negative, false-positive,
missing-helmet, missing-vest, and small-object examples are recorded in `metrics.json` and
`docs/assets/external_generalization/`.
