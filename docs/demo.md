# Demo guide

Public-demo media is not included because the internal validation videos have unresolved provenance. Supply a video you are authorized to process; do not use sensitive footage without appropriate notice, access control, and retention policy.

## Full pipeline

```powershell
python scripts\run_video_inference.py --source demo.mp4 `
  --output-dir results\demo --enable-tracking `
  --enable-ppe-association --enable-temporal-compliance `
  --zones configs\zones\example_zones.yaml --render `
  --overlay-mode demo --render-width 1920
```

`demo` is the default overlay: it hides raw PPE boxes and presents large per-Person
temporal status panels. Use `--overlay-mode debug` to retain detailed raw PPE boxes,
confidence values, association evidence, and diagnostic labels. `--render-width` changes
only the annotated video; inference, tracking, association, and temporal logic still run on
the original source frames.

Expected outputs:

- `annotated.mp4`: optional rendered detections, tracks, evidence, and temporal state.
- `detections.jsonl`: thresholded boxes and confidences by frame.
- `tracks.jsonl`: anonymous run-local Person track observations.
- `ppe_associations.jsonl`: per-track helmet/vest evidence and association diagnostics.
- `compliance_states.jsonl`: timestamp-driven temporal state for each track/equipment pair.
- `zone_membership.jsonl`: optional normalized-zone membership and applicable PPE policy.
- `incidents.jsonl`: incident open/close events and timing context.
- `run_metadata.json`: configuration, hashes, environment, diagnostics, and timing.

## API quickstart

```powershell
uvicorn visionops.api.app:app --host 127.0.0.1 --port 8000
curl.exe http://127.0.0.1:8000/health/live
curl.exe http://127.0.0.1:8000/health/ready
curl.exe -F "file=@sample.jpg" http://127.0.0.1:8000/v1/infer/image
curl.exe -F "file=@demo.mp4" http://127.0.0.1:8000/v1/jobs/video
curl.exe http://127.0.0.1:8000/v1/jobs/JOB_ID
curl.exe http://127.0.0.1:8000/v1/jobs/JOB_ID/result
curl.exe http://127.0.0.1:8000/v1/jobs/JOB_ID/incidents
curl.exe http://127.0.0.1:8000/metrics
```

The service is a bounded single-node reference implementation. Docker packaging exists, but local build/runtime validation was not completed because Docker Desktop's Linux daemon was unavailable. See [service Docker notes](experiments/service_deployment/docker.md).
