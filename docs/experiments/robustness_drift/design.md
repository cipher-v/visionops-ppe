# Checkpoint 16 design

## Boundaries

This experiment uses the frozen Candidate C YOLO26n PyTorch detector at 960 pixels and
the production class thresholds (`helmet` 0.32, `vest` 0.19, `Person` 0.41,
`no_helmet` 0.27, `no_vest` 0.11). It evaluates only Construction-PPE five-class
validation and the filtered RF100 validation split. RF100 test was not read. Images are
perturbed in memory; datasets are not rewritten. Seed 42 makes Gaussian noise repeatable.

The predeclared material-change rules are a 10% relative mAP50-95 loss and, separately,
a 10 percentage-point loss in the mean operational recall of `no_helmet` and `no_vest`.
No threshold, model, tracker, association, temporal, or zone setting was changed.

## Stress conditions

| Perturbation | Mild | Moderate | Severe |
|---|---:|---:|---:|
| Brightness decrease | 0.8x | 0.6x | 0.4x |
| Brightness increase | 1.15x | 1.3x | 1.5x |
| Contrast reduction | 0.8x | 0.6x | 0.4x |
| Gaussian blur | 3 px | 5 px | 9 px |
| Diagonal motion blur | 3 px | 5 px | 9 px |
| Gaussian noise sigma | 5 | 15 | 30 |
| JPEG quality | 70 | 40 | 20 |
| Saturation reduction | 0.8x | 0.5x | 0.2x |

These are engineering stress levels, not a standardized robustness benchmark.

## Metrics

Ranking metrics use confidence-ordered IoU matching over 0.50:0.95, then Ultralytics'
`ap_per_class`. Operational TP/FP/FN metrics use the frozen class thresholds and IoU
0.50. Every condition is compared with its own dataset's unmodified baseline.

Input monitoring uses a 256-pixel-maximum-dimension proxy while retaining original
width, height, and aspect ratio. It measures grayscale mean and standard deviation,
contrast, HSV saturation, Laplacian variance, a 32-bin grayscale histogram, 18-bin hue
histogram, and 16-bin saturation and value histograms. Output monitoring measures counts,
class frequency and confidence summaries, Person:PPE ratio, zero-Person frames, and
negative-PPE frames.

Width and height distributions are retained in the reference and records for direct
inspection. Aspect ratio participates in the scalar drift score; absolute resolution does
not, because mixed source resolutions and deployment resizing would otherwise dominate
the maximum score without describing visual degradation.

