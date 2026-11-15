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

**Checkpoint 7 — controlled training-data diversity experiment**

The repository now includes a controlled five-class comparison of Construction-PPE-only,
exposure-matched, and mixed-domain YOLO26n training. All three models are evaluated on
Construction-PPE validation and RF100 validation under a predeclared decision rule. RF100
test remains frozen. Tracking, inference APIs, export, event logic, dashboards, and
monitoring remain deferred.

