# Person-tracking smoke benchmark

This is a measured local smoke comparison on a variable laptop, not stable production
throughput or tracking-accuracy validation.

## Setup

- Input: the same 90-frame generated MJPEG video for both runs
- Resolution/FPS: 640x360 at 30 FPS
- Frame stride: 1
- Rendering: enabled for both runs
- Detector: frozen YOLO26n at 960 with unchanged class thresholds
- Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- Tracker: Ultralytics ByteTrack 8.4.212, Person only

## Results

| Measurement | Detector only | Detector + tracking |
|---|---:|---:|
| Frames processed | 90 | 90 |
| End-to-end mean | 49.85 ms | 49.33 ms |
| End-to-end p50 | 46.97 ms | 47.19 ms |
| End-to-end p95 | 61.82 ms | 59.42 ms |
| Effective pipeline FPS | 19.13 | 19.29 |
| Read failures/dropped frames | 0/0 | 0/0 |

The direct ByteTrack update measured mean `0.492 ms`, p50 `0.445 ms`, and p95
`0.690 ms` per processed frame. This direct mean is the useful measured tracking overhead.
The observed end-to-end mean delta was `-0.515 ms` and FPS increased slightly, which is
noise rather than a tracking speedup; detector timing and laptop load varied between the
sequential runs.

## Smoke tracking output

The synthetic shapes produced 68 frozen-policy detections: 52 `helmet` and 16 `vest`.
They produced no `Person` detections, so the correct tracking output was zero unique tracks,
zero observations, and an empty `tracks.jsonl`. This run verifies integration and empty
Person handling only. Stable identity behavior is exercised separately with deterministic
injected Person detections and is not inferred from this video.
