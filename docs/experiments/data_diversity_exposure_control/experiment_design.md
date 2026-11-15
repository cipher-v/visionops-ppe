# Leakage-corrected, exposure-matched target-domain control

## Scope and terminology

Conditions A and B train only on Construction-PPE, so RF100 validation is an unseen
external development domain for those models. Conditions C and D train on
Construction-PPE plus RF100 train; RF100 validation is therefore held-out target-domain
validation, not evidence of universal or unseen-domain generalization.

RF100 test is frozen. It is not read, enumerated, evaluated, or used for model selection
in this checkpoint. It was used previously in Checkpoint 5 and is not pristine for the
overall research process.

## Leakage correction

The earlier audit checked RF100 against Construction-PPE but did not check RF100 train
against RF100 validation. Checkpoint 7B recomputes exact SHA-256 and deterministic 64-bit
grayscale dHash matches at the predeclared Hamming-distance threshold `<= 4`.

Confirmed validation copies of RF100 training images are excluded from a derived,
Git-ignored validation view. Original RF100 files are unchanged. Existing 119-image
results remain preserved as legacy/unfiltered metrics; all primary four-way comparisons
use the filtered view.

## Condition D

D uses the exact C training data, vocabulary, pretrained `yolo26n.pt`, 960-pixel input,
batch 8, device 0, seed 42, deterministic mode, AMP, four workers, no cache, default
optimizer/loss/augmentation settings, and 100-epoch scheduler horizon.

Configuring `epochs: 45` would compress the learning-rate schedule and move the default
10-epoch close-mosaic phase to epoch 35. Instead, D retains `epochs: 100`, disables normal
early stopping with the Ultralytics 8.4.212-supported `patience: 0`, and requests a
controlled stop from `on_fit_epoch_end` after epoch 45 has completed validation, metric
writing, and checkpoint saving. Epoch 46 must not begin. Thus close-mosaic remains
scheduled for epoch 90 and is not reached.

Expected D exposure is `1,797 x 45 = 80,865` image entries, matching B. Optimizer steps
are measured at runtime and compared separately.

## Predeclared decision rule

Matched-exposure target-domain diversity is strongly supported only if D satisfies all:

1. Filtered RF100-validation mAP50-95 improves at least `+0.03` over A.
2. Filtered RF100-validation mAP50-95 improves at least `+0.02` over B.
3. Construction-PPE-validation mAP50-95 changes by no less than `-0.03` versus A.

These thresholds are fixed before D results are inspected. C-minus-D results estimate the
association with additional mixed-data training exposure, not perfect causal effects.
