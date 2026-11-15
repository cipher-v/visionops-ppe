# Video inference benchmark

No suitable real video existed locally, so no real-video result is claimed. The measured
synthetic smoke benchmark is recorded here after running the frozen pipeline; it validates
execution and timing instrumentation, not detector accuracy or real-scene throughput.

## Synthetic smoke run

- Date: 2026-11-15
- Input: 90 generated MJPEG frames, 640x360, 30 FPS, 3.0 seconds
- Frame stride: 1
- Annotated output: enabled
- Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- PyTorch: 2.5.1+cu121; CUDA runtime: 12.1
- Frames read/processed: 90/90
- Read failures/dropped frames: 0/0
- Frames with detections: 63; structured detections: 68

Three complete runs were measured while finalizing manifest validation and separating core
inference from detector-call throughput. The large spread is retained rather than
presenting the faster run as a stable hardware benchmark.

| Run | Warmup | Processing wall time | Core inference FPS | Detector-call FPS | Pipeline FPS | Detector p50/p95 | End-to-end p50/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initial smoke | 2.0954 s | 1.4427 s | 103.70 | 72.65 | 62.38 | 13.02/18.65 ms | 14.06/19.97 ms |
| Final verified | 6.9066 s | 3.9285 s | 35.94 | 26.45 | 22.91 | 34.61/54.67 ms | 37.27/56.59 ms |
| Instrumentation-final | 9.8279 s | 6.0777 s | 22.76 | 16.88 | 14.81 | 53.27/93.10 ms | 57.37/98.04 ms |

Instrumentation-final run component timings:

| Timing scope | Mean | p50 | p95 |
|---|---:|---:|---:|
| Detector call | 59.23 ms | 53.27 ms | 93.10 ms |
| End-to-end processed frame | 64.44 ms | 57.37 ms | 98.04 ms |
| Ultralytics preprocess | 7.51 ms | 6.84 ms | 12.76 ms |
| Ultralytics inference | 43.94 ms | 40.41 ms | 63.88 ms |
| Ultralytics postprocess | 6.61 ms | 4.48 ms | 8.85 ms |

The synthetic shapes triggered model detections, but those detections have no semantic
accuracy meaning. This run measures this generated input and local hardware only; codec,
resolution, scene complexity, rendering, source transport, power state, and competing GPU
load can change throughput. These runs demonstrate that this laptop was not a stable,
isolated benchmarking environment.

## Real-video and stream status

No suitable real video was present in the repository, and no external footage was
downloaded. Webcam and RTSP/HTTP sources were not exercised, so no measured claims are
made for them. A real local video can be run with:

```powershell
.\.venv\Scripts\python.exe scripts\run_video_inference.py `
  --source path\to\real_video.mp4 `
  --output-dir results\video_inference\real_demo `
  --render
```
