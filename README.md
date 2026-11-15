# VisionOps PPE

VisionOps PPE is a production-oriented computer vision project intended to detect personal protective equipment (PPE) compliance and support reliable operational monitoring.

## Planned ML lifecycle

The planned lifecycle covers data preparation and validation, model training and evaluation, inference and tracking, service delivery, observability, and drift detection. These capabilities are intentionally deferred until later checkpoints.

## Repository structure

```text
visionops-ppe/
├── src/visionops/  # Installable Python package
├── configs/        # Version-controlled configuration
├── scripts/        # Project automation and entry-point scripts
├── tests/          # Automated tests
├── results/        # Local generated results (ignored by Git)
├── docs/           # Project documentation
├── pyproject.toml  # Package, tooling, and dependency configuration
├── .gitignore      # Files excluded from version control
└── README.md       # Project overview and setup guidance
```

## Current status

**Checkpoint 11 - frame-level Person-to-PPE association**

Production v1 and Person-only ByteTrack remain frozen. The streaming pipeline now
optionally associates helmet and vest evidence with active Person tracks using documented
head/torso geometry, PPE-box containment, deterministic one-to-one assignment, and
explicit present/violation/conflict/unknown frame states. Temporal persistence, compliance
decisions, zones, incidents, model export, and APIs remain deferred.

