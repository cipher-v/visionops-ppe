# Testing and validation

Local verification:

```powershell
pytest
ruff check .
ruff format --check .
python -m pip check
git diff --check
```

The pytest suite covers repository-owned schemas, geometry, state machines, diagnostics, drift, API behavior, upload limits, hash guards, review selection and manual export. CPU integration tests use synthetic images/videos and fake runtime components where appropriate. The actual hash-verified GPU service smoke test is `python scripts/smoke_service.py`; it is manual because CI does not receive weights or GPU access.

Synthetic diagnostics exercise tracking continuity, association ambiguity/conflicts, temporal incidents, zones, drift math, and review deduplication. Three internal videos provide qualitative pipeline validation only. They do not provide formal detection accuracy, MOT metrics, association ground truth, or incident sensitivity/specificity.

Passing tests demonstrates software behavior, not real-world safety performance. Deployment still requires annotated, camera-representative validation, security review, privacy controls, load testing, and operational acceptance criteria.
