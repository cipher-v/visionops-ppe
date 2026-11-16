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

**Checkpoint 15 - deployment export, backend parity, and performance profiling**

The frozen production pipeline now has a hash-verified detector-backend abstraction and
stage-level performance instrumentation. ONNX FP32 was exported and rigorously evaluated,
but it failed the predeclared detection and downstream behavioral parity gates and was
slower in the controlled detector benchmark, so PyTorch remains production. TensorRT was
not forced into an incompatible local runtime stack. APIs and drift monitoring remain
deferred.

