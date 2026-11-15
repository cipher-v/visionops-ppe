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

**Checkpoint 12 - temporal PPE state and incident generation**

Production v1, ByteTrack, and frame-level PPE association remain frozen. The streaming
pipeline now optionally converts explicit frame evidence into timestamp-driven per-track
helmet/vest state, deduplicated incident open/close events, unknown/conflict grace,
sustained-positive clearing, cooldown, and track-expiration closure. Spatial zones, model
export, APIs, and drift monitoring remain deferred.

