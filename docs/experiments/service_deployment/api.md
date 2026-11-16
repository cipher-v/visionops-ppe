# API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health/live` | Process liveness; no inference |
| GET | `/health/ready` | Loaded runtime, backend, device and safe model identity |
| GET | `/v1/config` | Read-only frozen policy summary |
| POST | `/v1/infer/image` | Stateless multipart image detection |
| POST | `/v1/jobs/video` | Queue multipart video plus optional zone YAML |
| GET | `/v1/jobs/{job_id}` | Poll queued/running/completed/failed status |
| GET | `/v1/jobs/{job_id}/result` | Bounded completed-job summary and artifact names |
| GET | `/v1/jobs/{job_id}/incidents` | Structured incidents; track IDs are anonymous/run-local |
| GET | `/metrics` | Prometheus text exposition |

Default limits are 10 MiB per image and 250 MiB per video. Environment variables
`VISIONOPS_MAX_IMAGE_BYTES`, `VISIONOPS_MAX_VIDEO_BYTES`,
`VISIONOPS_IMAGE_CONCURRENCY`, `VISIONOPS_VIDEO_WORKERS`, and
`VISIONOPS_RUNTIME_DIR` control serving resources. Video worker count is intentionally
restricted to one for the laptop GPU.

Errors use `{"error":{"code":"...","message":"..."}}`; stack traces remain in
server logs. Video results reference artifact names rather than returning large JSONL.

