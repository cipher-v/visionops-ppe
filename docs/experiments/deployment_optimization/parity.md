# Backend parity results

## Detection parity

The corrected ONNX FP32 export improved substantially over the rejected diagnostic export,
but still failed every predeclared gate:

| Metric | ONNX FP32 | Required |
| --- | ---: | ---: |
| Reference detections | 279 | — |
| ONNX detections | 277 | — |
| Matched | 273 | — |
| Match rate | 97.849% | ≥99.5% |
| Mean matched IoU | 0.981289 | ≥0.995 |
| p5 matched IoU | 0.938490 | — |
| Mean absolute confidence delta | 0.012838 | — |
| p95 absolute confidence delta | 0.056250 | ≤0.005 |

Unmatched counts were six PyTorch and four ONNX detections. Most per-class detections
matched, but borderline confidence changes affected final production-policy outputs. No
threshold was changed to improve parity.

## Validation metrics

| Domain | Backend | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| Construction-PPE val | PyTorch | 0.942335 | 0.813111 | 0.873890 | 0.663919 |
| Construction-PPE val | ONNX | 0.937093 | 0.816602 | 0.872232 | 0.661886 |
| Filtered RF100 val | PyTorch | 0.882332 | 0.818220 | 0.883380 | 0.496199 |
| Filtered RF100 val | ONNX | 0.875114 | 0.827999 | 0.876455 | 0.495623 |

ONNX minus PyTorch mAP50-95 was -0.002033 internally and -0.000576 on filtered RF100
validation. RF100 test was not used. Small ranking-metric deltas do not override failed
final-detection and downstream parity.

## Full-pipeline behavior

- `compliant_multiworker`: detector counts, tracks, associations, temporal states, and
  incidents were identical.
- `helmet_transition`: detector counts differed, PyTorch produced two unique tracks while
  ONNX produced one, state counts differed, and helmet incident confirmation moved from
  5.56 s to 5.36 s. Equipment and close reasons otherwise matched.
- `walking_multiworker`: both produced four tracks and no incidents, but detector,
  association, and temporal-state counts differed.

These divergences are material for a temporal safety pipeline, so ONNX is not approved as
the production backend.
