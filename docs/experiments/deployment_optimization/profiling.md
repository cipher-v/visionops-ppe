# Stage-level profiling

Pipeline instrumentation separates decode, Ultralytics-reported preprocessing/inference/
postprocessing, tracker update, PPE association, temporal compliance, optional zone policy,
overlay rendering, encoding/write, and total end-to-end latency. Detector subcomponents
are nested within detector wall time and must not be added to it again. Stage timers are
otherwise non-overlapping. Ultralytics boundaries are library-reported; detector-only wall
timing uses explicit CUDA synchronization.

Representative mean original-frame stages:

| Backend/clip | Decode | Tracker | Render | Encode |
| --- | ---: | ---: | ---: | ---: |
| PyTorch 2160×3840 | 18.11 ms | 53.31 ms | 7.97 ms | 36.50 ms |
| PyTorch 3840×2160 | 12.43 ms | 45.71 ms | 5.49 ms | 34.72 ms |
| PyTorch 1920×1080 | 3.50 ms | 13.74 ms | 1.57 ms | 10.11 ms |
| ONNX 2160×3840 | 14.88 ms | 46.06 ms | 5.75 ms | 35.15 ms |
| ONNX 3840×2160 | 16.84 ms | 54.77 ms | 8.07 ms | 36.15 ms |
| ONNX 1920×1080 | 3.76 ms | 14.41 ms | 1.66 ms | 10.39 ms |

PPE association and temporal logic remain sub-millisecond. Zones were disabled consistently.
The dominant 4K costs are detector preprocessing/inference plus original-frame BoT-SORT
GMC, decoding, and video encoding. Tracker cost rises roughly 3.3–3.9× and encoding about
3.4–3.6× from 1080p to 4K. Rendering also scales with source pixels. The model input stays
960, so those original-frame stages explain much of the application gap.

Timings are workstation observations, not deployment SLAs. Video decode and GPU scheduling
show run-to-run variation, particularly at 4K.
