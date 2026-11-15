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

**Checkpoint 10 - person-only multi-object tracking**

Production v1 remains frozen as the mixed-source YOLO26n model at 960 pixels with its
calibrated five-class confidence policy. The streaming pipeline now optionally adds
ByteTrack identities for Person detections only, with structured track records and
ground-truth-free lifecycle diagnostics. PPE-person association, compliance decisions,
zones, temporal incidents, model export, and APIs remain deferred.

