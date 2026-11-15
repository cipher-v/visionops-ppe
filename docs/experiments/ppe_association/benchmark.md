# PPE association smoke benchmark

Measured local pipeline and repository-association timings will be recorded here. This is
a variable laptop smoke benchmark, not stable production throughput.

## Crowded injected geometry

For three Person tracks and eight PPE detections over 1,000 repeated calls:

- Mean: `0.856 ms/frame`
- p50: `0.858 ms/frame`
- p95: `1.230 ms/frame`

This isolates repository-owned association from detector and tracker runtime.

## Same-input pipeline smoke comparison

Both sequential runs used the same 90-frame generated 640x360 video, stride 1, rendering,
the frozen detector, ByteTrack, and the RTX 3050 laptop GPU.

| Measurement | Detector + tracking | Detector + tracking + association |
|---|---:|---:|
| Frames processed | 90 | 90 |
| End-to-end mean | 32.96 ms | 31.23 ms |
| End-to-end p50 | 30.25 ms | 28.98 ms |
| End-to-end p95 | 47.29 ms | 44.22 ms |
| Effective pipeline FPS | 28.75 | 30.39 |
| Read failures/dropped frames | 0/0 | 0/0 |

The synthetic video produced no Person tracks, so the association layer considered 68 PPE
detections and correctly left all 68 unassigned. Its measured empty-Person workload was:

- Mean: `0.019 ms/frame`
- p50: `0.018 ms/frame`
- p95: `0.027 ms/frame`

The association-enabled end-to-end run appears faster, which is system/GPU variability,
not an association speedup. The isolated crowded-geometry measurement above is more
representative of repository association work when Person candidates exist.
