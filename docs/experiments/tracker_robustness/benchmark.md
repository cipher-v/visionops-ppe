# Tracker robustness benchmark

Hardware: NVIDIA GeForce RTX 3050 Laptop GPU 6 GB. Software: Python 3.12.6,
PyTorch 2.5.1+cu121, CUDA 12.1, OpenCV 5.0.0, and Ultralytics 8.4.212.

## Repository-measured tracker update latency

| Clip | Tracker | Mean | p50 | p95 |
|---|---|---:|---:|---:|
| Compliant | ByteTrack | 1.075 ms | 0.963 ms | 1.631 ms |
| Compliant | BoT-SORT | 51.366 ms | 48.777 ms | 62.734 ms |
| Compliant | BoT-SORT + ReID | 93.647 ms | 83.828 ms | 152.082 ms |
| Helmet transition | ByteTrack | 1.042 ms | 0.947 ms | 1.742 ms |
| Helmet transition | BoT-SORT | 61.520 ms | 59.799 ms | 74.389 ms |
| Helmet transition | BoT-SORT + ReID | 80.445 ms | 77.044 ms | 115.190 ms |
| Walking | ByteTrack | 1.142 ms | 1.050 ms | 1.589 ms |
| Walking | BoT-SORT | 20.386 ms | 18.077 ms | 35.701 ms |
| Walking | BoT-SORT + ReID | 26.526 ms | 25.202 ms | 34.161 ms |

BoT-SORT's sparse-optical-flow global motion compensation dominates its added cost on the
high-resolution clips. ReID adds another per-detection encoder pass and provides no observed
continuity improvement here.

## Complete pipeline

| Clip | Tracker | Effective FPS | End-to-end p50 | End-to-end p95 |
|---|---|---:|---:|---:|
| Compliant | ByteTrack | 5.420 | 144.868 ms | 203.083 ms |
| Compliant | BoT-SORT | 5.956 | 146.588 ms | 191.057 ms |
| Compliant | BoT-SORT + ReID | 4.134 | 201.714 ms | 331.421 ms |
| Helmet transition | ByteTrack | 5.407 | 144.724 ms | 215.071 ms |
| Helmet transition | BoT-SORT | 5.221 | 164.179 ms | 219.594 ms |
| Helmet transition | BoT-SORT + ReID | 5.022 | 172.580 ms | 232.841 ms |
| Walking | ByteTrack | 11.601 | 78.216 ms | 93.228 ms |
| Walking | BoT-SORT | 13.957 | 58.746 ms | 104.706 ms |
| Walking | BoT-SORT + ReID | 15.439 | 58.839 ms | 76.121 ms |

Whole-pipeline results were sequential runs and include detector, decoding, rendering, and
encoding variability. Later walking runs benefited from runtime warm state, so their higher
FPS must not be interpreted as ReID acceleration. Tracker-update measurements show the
actual repository-owned incremental cost.
