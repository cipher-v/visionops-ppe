# Detector and application benchmarks

## Mode A — detector only

Five warmups were excluded. Each backend then processed the same 42 preloaded inputs three
times (126 measured samples).

| Backend | Mean | p50 | p95 | Throughput |
| --- | ---: | ---: | ---: | ---: |
| PyTorch | 14.708 ms | 14.390 ms | 18.873 ms | 67.99 FPS |
| ONNX Runtime GPU | 19.541 ms | 19.494 ms | 21.195 ms | 51.17 FPS |

PyTorch was approximately 1.33× faster by mean detector wall time. Its measured CUDA
allocator peak was 94.70 MiB. ONNX Runtime GPU memory was not reported because this API
does not expose a reliable comparable allocator peak.

## Mode B — native-resolution end to end

| Backend | Clip/resolution | FPS | Detector p50/p95 | End-to-end p50/p95 |
| --- | --- | ---: | ---: | ---: |
| PyTorch | compliant, 2160×3840 | 5.84 | 52.99 / 103.41 ms | 151.86 / 202.85 ms |
| ONNX | compliant, 2160×3840 | 6.65 | 49.86 / 56.19 ms | 137.07 / 149.90 ms |
| PyTorch | transition, 3840×2160 | 6.89 | 50.53 / 56.92 ms | 136.67 / 150.93 ms |
| ONNX | transition, 3840×2160 | 5.89 | 58.73 / 95.26 ms | 157.19 / 199.08 ms |
| PyTorch | walking, 1920×1080 | 19.92 | 22.41 / 24.26 ms | 47.79 / 52.99 ms |
| ONNX | walking, 1920×1080 | 17.92 | 24.91 / 27.04 ms | 52.06 / 56.83 ms |

Across all 1,189 frames, PyTorch achieved 9.44 effective FPS versus ONNX's 9.15 FPS.
The controlled preloaded benchmark is the better backend-isolation result; per-video
differences include decode, tracker workload, rendering, encoding, and normal run variance.

The optional downscale experiment was not performed. It would change coordinates entering
GMC/tracking and was unnecessary for selecting between backends after ONNX failed parity.
