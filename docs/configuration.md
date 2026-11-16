# Configuration index

## Frozen production contract

| File | Purpose |
|---|---|
| `configs/production_detector.yaml` | Architecture, weight path/hash, 960 input, classes, NMS, immutable thresholds |
| `configs/production_tracker.yaml` | Selected BoT-SORT parameters without ReID |
| `configs/ppe_association.yaml` | Head/torso regions, geometry gates, scoring, deterministic tie-breaking |
| `configs/compliance_temporal.yaml` | Confirmation, clearing, unknown/conflict grace and cooldown durations |
| `configs/drift_reference.json` | Clean development input/output distributions and diagnostic PSI thresholds |

Runtime startup validates these files and the model SHA-256. Changes require a new evaluated model/policy version, not an API mutation.

## Optional/example configuration

- `configs/zones/example_zones.yaml`: normalized image-plane example; camera-specific, not world calibrated.
- `configs/deployment_backends.yaml`: evaluated backend artifacts; PyTorch remains selected.
- `configs/bytetrack_person.yaml`, `botsort_person.yaml`, `botsort_reid_person.yaml`: experiment lineage and rollback/comparison, not the production selector.
- `configs/experiment_*.yaml`, `baseline_yolo26n_640.yaml`: reproducible training experiments, not service configuration.

Review selection defaults currently live in `visionops.review.selection.ReviewConfig`: two-second temporal gap, dHash distance four, 40/video, 12/video-minute, and 12/type. They govern review workload only and never detector or incident policy.
