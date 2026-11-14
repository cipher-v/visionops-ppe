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

**Checkpoint 4 — controlled model-capacity experiment**

The repository now includes the fixed YOLO26n baselines and a controlled YOLO26n-versus-YOLO26s comparison at 960 pixels, with best-checkpoint validation, deterministic size-stratified recall diagnostics, and curated failure-analysis artifacts. Only model capacity changed in the latest experiment. No broader tuning, tracking, inference API, export, event logic, dashboards, or monitoring components have been added.

