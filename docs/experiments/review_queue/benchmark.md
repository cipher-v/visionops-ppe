# Performance and limitations

Candidate-rule evaluation averaged 0.0044 ms/event (p50 0.0006, p95 0.0232). Deduplication
and budget selection took 4.48 ms total. Extracting and JPEG-encoding 28 full-resolution
review frames took 33.83 s and stored 20,365,826 bytes; video decode and artifact I/O,
rather than selection logic, dominate cost.

Limitations: rules reflect available telemetry; JSONL lacks precise timestamps for every
tracker diagnostic; perceptual hashes can collapse semantically different similar frames;
priority weights are engineering choices; full-resolution artifacts are relatively large;
the three videos are limited and have unresolved public-use provenance; selected samples
are not guaranteed errors; human review labels are not automatically valid training
annotations. No automatic relabeling, dataset merge, retraining or drift action exists.

Manual export:

```powershell
.\.venv\Scripts\python.exe scripts\export_review_candidates.py `
  --queue results\review_queue\cp18-real-v1 `
  --output results\review_exports\cp18-reviewed
```

Only candidates explicitly marked reviewed/accepted with a human verdict export. Training
manifests remain untouched.

