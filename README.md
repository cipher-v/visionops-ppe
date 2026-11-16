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

**Checkpoint 16 - robustness stress testing and practical drift monitoring**

The frozen PyTorch detector has been evaluated under 24 deterministic stress conditions
on each of two validation domains. A lightweight, five-second-window drift monitor now
records input and detector-output shifts. Gaussian noise and motion blur are the principal
robustness weaknesses; generic PSI alerts detect distribution change but correlate poorly
with accuracy loss and over-alert on the limited real-video clips. No model or operating
threshold was changed.

