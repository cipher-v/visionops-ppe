# Production-style video detector design

## Frozen detector contract

The pipeline loads `configs/production_detector.yaml` and accepts only the frozen
YOLO26n detector at 960 pixels. Before model construction it checks that the weight file
exists and that its SHA-256 matches the manifest. It does not search for or fall back to
another checkpoint.

The five class thresholds are `helmet=0.32`, `vest=0.19`, `Person=0.41`,
`no_helmet=0.27`, and `no_vest=0.11`; NMS IoU remains `0.70`. Ultralytics receives
`conf=0.11`, the minimum configured threshold, so it cannot discard a candidate that
might pass the frozen policy. Repository-owned logic then applies the exact threshold for
each predicted class.

## Input abstraction and streaming

`visionops.video.source` accepts a local file, a non-negative webcam/device index, or an
RTSP/RTSPS/HTTP/HTTPS URI. URI credentials are removed from recorded source identifiers.
RTSP is structurally supported through OpenCV but has not been validated against an actual
RTSP endpoint.

Frames are read and processed sequentially. The default stride is one, so every frame is
inferred. A larger explicit stride reads all frames but performs inference only when
`frame_index % stride == 0`. No complete video or unbounded frame queue is held in memory.
Unknown FPS falls back to elapsed time for timestamps and 30 FPS for annotated output;
both choices are recorded.

An unavailable or malformed source raises a clear error. File read termination is recorded
as end-of-file. A failed device/stream read increments the read-failure and dropped-frame
counts before the run stops. OpenCV captures, output writers, and output files are managed
with context managers so they close on normal completion and exceptions.

## Structured outputs

Each JSONL record contains:

```json
{
  "frame_index": 128,
  "timestamp_seconds": 4.2667,
  "class_id": 3,
  "class_name": "no_helmet",
  "confidence": 0.81,
  "bbox_xyxy": [412.0, 118.0, 476.0, 207.0],
  "bbox_normalized": [0.412, 0.118, 0.476, 0.207],
  "source_id": "demo.mp4"
}
```

Pixel boxes are clipped to original-frame bounds before normalization. There is no
`track_id`. Every run writes `detections.jsonl` and `run_metadata.json`; `--render` adds
an annotated video at the original dimensions and reported FPS, or the documented
fallback FPS.

## Timing

One zero frame warms the frozen detector after source metadata is known; warmup time is
recorded but excluded from latency summaries. Detector latency measures the complete
model call. End-to-end frame latency also includes repository filtering, serialization,
and optional rendering/writing. Ultralytics core inference FPS, complete detector-call
FPS, and effective pipeline FPS are reported separately. Ultralytics preprocess,
inference, and postprocess timings are retained when provided.

## Intentionally excluded

This checkpoint has no training, threshold changes, tracking, person-PPE association,
zones, compliance state, timers, dwell time, cooldown, incidents, adaptive skipping,
ONNX/TensorRT export, API, dashboard, or drift logic. A single-frame violation detection
remains detector evidence, not an incident.

## CLI

```powershell
.\.venv\Scripts\python.exe scripts\run_video_inference.py `
  --source path\to\video.mp4 `
  --output-dir results\video_inference\demo01 `
  --render
```
