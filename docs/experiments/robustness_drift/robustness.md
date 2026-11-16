# Detector robustness

## Clean baselines

| Validation set | Precision | Recall | mAP50 | mAP50-95 | Violation recall |
|---|---:|---:|---:|---:|---:|
| Construction-PPE five-class | 0.9354 | 0.8170 | 0.8601 | 0.6574 | 0.7275 |
| RF100 filtered | 0.8489 | 0.8207 | 0.8443 | 0.4884 | 0.5818 |

Violation recall is the mean operational-threshold recall of `no_helmet` and `no_vest`.
These are the clean baselines from the same OpenCV-array prediction path used for every
on-the-fly perturbation. As a sanity check, their mAP50-95 values are within 0.0068 (CPPE)
and 0.0064 (RF100) of the frozen model-selection `model.val` results (0.6642 and 0.4948).
Using a matched clean path makes condition deltas internally controlled; it does not
replace the canonical model-selection metrics.

## Severity results

Each cell is mild / moderate / severe. Losses are relative to the matching clean domain.

### Construction-PPE validation

| Perturbation | mAP50-95 | Relative loss | Violation recall | Point loss |
|---|---|---|---|---|
| Brightness decrease | .6578 / .6575 / .6545 | -0.1% / -0.0% / 0.4% | .7219 / .7285 / .7219 | 0.6 / -0.1 / 0.6 pp |
| Brightness increase | .6547 / .6470 / .6298 | 0.4% / 1.6% / 4.2% | .7219 / .7135 / .6796 | 0.6 / 1.4 / 4.8 pp |
| Contrast reduction | .6510 / .6417 / .6236 | 1.0% / 2.4% / 5.1% | .7135 / .6796 / .6712 | 1.4 / 4.8 / 5.6 pp |
| Gaussian blur | .6588 / .6586 / .6543 | -0.2% / -0.2% / 0.5% | .7313 / .7313 / .7324 | -0.4 / -0.4 / -0.5 pp |
| Motion blur | .6589 / .6551 / .5992 | -0.2% / 0.4% / 8.9% | .7369 / .7285 / .6275 | -0.9 / -0.1 / 10.0 pp |
| Gaussian noise | .6532 / .5920 / .4221 | 0.6% / 10.0% / 35.8% | .7173 / .6136 / .3809 | 1.0 / 11.4 / 34.7 pp |
| JPEG | .6529 / .6478 / .6261 | 0.7% / 1.5% / 4.8% | .7219 / .6908 / .6436 | 0.6 / 3.7 / 8.4 pp |
| Saturation reduction | .6568 / .6551 / .6415 | 0.1% / 0.4% / 2.4% | .7303 / .7191 / .6824 | -0.3 / 0.8 / 4.5 pp |

### RF100 filtered validation

| Perturbation | mAP50-95 | Relative loss | Violation recall | Point loss |
|---|---|---|---|---|
| Brightness decrease | .4890 / .4764 / .4693 | -0.1% / 2.5% / 3.9% | .5874 / .5874 / .5763 | -0.6 / -0.6 / 0.6 pp |
| Brightness increase | .4843 / .4708 / .4504 | 0.8% / 3.6% / 7.8% | .5707 / .5485 / .4808 | 1.1 / 3.3 / 10.1 pp |
| Contrast reduction | .4965 / .4985 / .4945 | -1.6% / -2.1% / -1.3% | .5763 / .6273 / .6894 | 0.6 / -4.5 / -10.8 pp |
| Gaussian blur | .4847 / .4784 / .4492 | 0.8% / 2.0% / 8.0% | .5763 / .5818 / .5540 | 0.6 / 0.0 / 2.8 pp |
| Motion blur | .4809 / .4565 / .3653 | 1.5% / 6.5% / 25.2% | .5485 / .5318 / .3409 | 3.3 / 5.0 / 24.1 pp |
| Gaussian noise | .4712 / .4339 / .2790 | 3.5% / 11.2% / 42.9% | .5929 / .5030 / .2566 | -1.1 / 7.9 / 32.5 pp |
| JPEG | .4825 / .4765 / .4654 | 1.2% / 2.4% / 4.7% | .5874 / .5707 / .5086 | -0.6 / 1.1 / 7.3 pp |
| Saturation reduction | .4841 / .4823 / .4768 | 0.9% / 1.2% / 2.4% | .5818 / .5818 / .5818 | 0.0 / 0.0 / 0.0 pp |

Gaussian noise is the clearest weakness: material ranking loss begins at severe on CPPE
(the moderate value is just below 10% before rounding) and moderate on RF100; violation
recall becomes material at moderate CPPE and severe RF100. Severe motion blur is material
on RF100 and exactly reaches the CPPE violation boundary. Other perturbations do not cross
the ranking rule, although severe brightness increase crosses RF100's violation rule.

At severe Gaussian noise, operational CPPE TP fell by 326 while FP fell by 42 and FN rose
by 326; RF100 TP fell by 274, FP fell by 28, and FN rose by 274. That pattern is dominated
by missed/low-confidence detections, not new false positives. Severe motion blur has the
same pattern (CPPE -109 TP/+109 FN; RF100 -147 TP/+147 FN) plus localization degradation.
Worst severe recall losses vary by domain: CPPE `no_helmet` under noise (-29.1 pp) and
`Person` under motion blur (-7.7 pp); RF100 `vest` under noise (-37.7 pp) and `no_vest`
under motion blur (-30.3 pp). Full per-class ranking and operational results are in
`metrics.json`.

