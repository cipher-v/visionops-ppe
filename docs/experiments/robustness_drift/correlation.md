# Drift versus measured degradation

Across all 48 dataset/perturbation/severity conditions, correlations are weak and not
monotonic:

| Degradation target | Pearson | Spearman |
|---|---:|---:|
| Relative mAP50-95 loss | -0.146 | 0.084 |
| Person recall loss | -0.148 | -0.021 |
| `no_helmet` recall loss | -0.211 | 0.036 |
| `no_vest` recall loss | -0.196 | 0.093 |

This generic maximum-PSI score does not provide a reliable scalar estimate of detector
degradation. The negative Pearson values are largely explained by photometric changes
that move histograms strongly but preserve detections. Rank correlations near zero confirm
that the ordering is unreliable as well. These results do not establish causality.

The signals still provide useful telemetry: they expose camera/input shifts and detector
output-mix changes that an operator can investigate. They should be charted feature by
feature alongside model outputs and operational outcomes, not used alone as a failure
oracle. Post-hoc threshold tuning was intentionally avoided.

