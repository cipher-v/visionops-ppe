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

**Checkpoint 7B — leakage-corrected exposure-matched target-domain control**

The repository now includes a controlled five-class comparison of Construction-PPE-only,
quantity-controlled, exposure-matched mixed-source, and full-budget mixed-source YOLO26n
training. Primary RF100 results use a leakage-filtered 118-image held-out target-domain
split. The matched-exposure model passes the predeclared diversity criteria. RF100 test
remains frozen; final model selection and production work remain deferred.

