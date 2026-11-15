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

**Checkpoint 8 - final model selection, calibration, and freeze**

Production v1 freezes the full-budget mixed-source YOLO26n model at 960 pixels with a
five-class vocabulary and a calibrated detector-level confidence policy. Calibration uses
only Construction-PPE validation and the leakage-filtered 118-image RF100 validation
view. RF100 test remains frozen, and video inference, tracking, temporal event logic, and
model export remain deferred.

