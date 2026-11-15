# Real-video validation results

All 1,189 frames were processed with zero read failures or dropped frames. Counts below are
system outputs and must not be interpreted as accuracy.

## Video and performance

| Clip | Frames | Duration | Effective FPS | Detector p50/p95 | Pipeline p50/p95 |
|---|---:|---:|---:|---:|---:|
| `compliant_multiworker` | 315 | 12.600 s | 5.420 | 55.469 / 107.776 ms | 144.868 / 203.083 ms |
| `helmet_transition` | 297 | 11.880 s | 5.407 | 55.672 / 106.853 ms | 144.724 / 215.071 ms |
| `walking_multiworker` | 577 | 24.066 s | 11.601 | 49.749 / 54.875 ms | 78.216 / 93.228 ms |

Aggregate processing time was 162.794 seconds (176.734 seconds including model loading and
warm-up), for a frame-weighted 7.304 FPS. Exact combined-sample latency was 53.471/105.261
ms detector p50/p95 and 125.933/197.636 ms end-to-end p50/p95.

Hardware and software: NVIDIA GeForce RTX 3050 Laptop GPU 6 GB, Python 3.12.6, PyTorch
2.5.1+cu121, CUDA 12.1, OpenCV 5.0.0, and Ultralytics 8.4.212.

## Detection counts

| Clip | Person | Helmet | Vest | No helmet | No vest |
|---|---:|---:|---:|---:|---:|
| `compliant_multiworker` | 630 | 630 | 630 | 0 | 0 |
| `helmet_transition` | 299 | 98 | 2 | 212 | 309 |
| `walking_multiworker` | 1,000 | 711 | 1,101 | 76 | 3 |

## Tracking

| Clip | Unique tracks | Observations/track | Median | Longest | Very short | Lost/recovered | Expired |
|---|---:|---|---:|---:|---:|---:|---:|
| `compliant_multiworker` | 2 | 315, 315 | 315 | 315 | 0 | 0 / 0 | 0 |
| `helmet_transition` | 2 | 295, 3 | 149 | 295 | 1 | 3 / 2 | 1 |
| `walking_multiworker` | 4 | 153, 545, 37, 263 | 208 | 545 | 0 | 25 / 23 | 2 |

There were no singleton tracks. “Very short” means fewer observations than one second of
source FPS.

## PPE association

| Clip | Considered | Assigned | Unassigned | Ambiguous | Mean score |
|---|---:|---:|---:|---:|---:|
| `compliant_multiworker` | 1,260 | 1,260 | 0 | 0 | 0.868 |
| `helmet_transition` | 621 | 617 | 4 | 2 | 0.892 |
| `walking_multiworker` | 1,891 | 1,766 | 125 | 4 | 0.850 |

Assignment rates by detected class:

| Clip | Helmet | Vest | No helmet | No vest |
|---|---:|---:|---:|---:|
| `compliant_multiworker` | 100.0% | 100.0% | n/a | n/a |
| `helmet_transition` | 98.0% | 0.0% | 100.0% | 100.0% |
| `walking_multiworker` | 100.0% | 88.9% | 98.7% | 33.3% |

## Temporal state and incidents

| Clip | Compliant | Suspected | Confirmed | Conflict | Unknown | Opened/closed |
|---|---:|---:|---:|---:|---:|---:|
| `compliant_multiworker` | 1,260 | 0 | 0 | 0 | 0 | 0 / 0 |
| `helmet_transition` | 96 | 84 | 415 | 0 | 1 | 2 / 2 |
| `walking_multiworker` | 1,691 | 89 | 0 | 36 | 180 | 0 / 0 |

`helmet_transition` emitted two incidents on Track 1:

| Equipment | Candidate start | Confirmed | Closed | Duration | Close reason |
|---|---:|---:|---:|---:|---|
| Vest | 0.000 s | 1.520 s | 11.840 s | 11.840 s | `source_ended` |
| Helmet | 3.840 s | 5.560 s | 11.840 s | 8.000 s | `source_ended` |

No incident was fabricated for either zero-incident clip. Full machine-readable values are
in `metrics.json`.
