# Docker strategy

The image targets a reproducible Linux CPU service using `python:3.12-slim`. PyTorch and
TorchVision CPU wheels are installed from the official PyTorch CPU index, then the local
package's `service` extra is installed. It runs as a non-root user, starts one Uvicorn
worker and probes `/health/live` without model inference.

Weights are not copied into the image or Git. Mount the frozen file at its manifest path:

```powershell
docker build -t visionops-ppe:cp17 .
docker run --rm -p 8000:8000 `
  -v "${PWD}/runs/experiment_mixed5_yolo26n_960/weights:/opt/visionops/runs/experiment_mixed5_yolo26n_960/weights:ro" `
  visionops-ppe:cp17
```

Startup performs the same SHA-256 verification inside the container. No internet model
download occurs. NVIDIA deployment can use an NVIDIA-compatible PyTorch base/runtime and
`--gpus all`, but that variant was not created or claimed here.

Docker CLI 29.3.1 was installed locally, but the Docker Desktop Linux daemon was not
running. Consequently the Dockerfile was inspected and linted through repository checks,
but an image build/run was not performed.

