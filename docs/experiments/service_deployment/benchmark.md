# Local service smoke benchmark

The real hash-verified PyTorch/CUDA runtime was exercised on Windows with TestClient and
actual multipart requests. Startup including model load, verification and warmup took
5.492 s. A synthetic 192x128 image request took 155.45 ms wall time, with 151.49 ms
reported detector processing time. It intentionally produced no detections.

A three-frame, 0.6-second synthetic MP4 traversed queued/running/completed, processed at
12.52 effective FPS, returned its result and zero incidents, and exposed metrics. This is
a functional smoke benchmark, not a load or production-capacity benchmark. TestClient
avoids network/socket overhead.

