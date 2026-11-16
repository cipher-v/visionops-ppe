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

**Checkpoint 13B - bounded real-video tracking robustness comparison**

Three locally supplied real videos were processed end to end with the frozen detector,
tracked Person IDs, PPE association, temporal compliance, and structured event records.
A predeclared comparison selected supported BoT-SORT without ReID as the production tracker
because it prevented an observed identity transfer; ByteTrack remains preserved as the
reproducible baseline. This is qualitative operational evidence, not a formally annotated
tracking benchmark. Spatial zones, model export, APIs, and drift monitoring remain deferred.

