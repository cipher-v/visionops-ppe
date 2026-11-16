# System architecture

```mermaid
flowchart TB
    IN[Image, file video, or stream] --> DET[Frozen YOLO26n detect<br/>960 px, PyTorch]
    DET --> POL[Class-specific thresholds<br/>NMS IoU 0.70]
    POL --> TRK[BoT-SORT Person tracks<br/>anonymous and run-local]
    POL --> PPE[PPE detections]
    TRK --> ASC[Deterministic geometric association]
    PPE --> ASC
    ASC --> TMP[Temporal compliance<br/>unknown/conflict/suspected/confirmed/cleared]
    TMP --> ZONE[Optional normalized image-plane zones]
    ZONE --> INC[Structured incident events]
    POL -. aggregate features .-> DRIFT[PSI drift telemetry<br/>diagnostic only]
    TRK -. uncertainty/events .-> REVIEW[Deduplicated review queue]
    ASC -. ambiguity/conflict .-> REVIEW
    TMP -. incident context .-> REVIEW
    DRIFT -. nomination only .-> REVIEW
    API[FastAPI runtime] --> DET
    INC --> API
    API --> METRICS[Prometheus-compatible metrics]
```

The solid perception path determines evidence and incidents. Drift never controls inference, incidents, or retraining. Review candidates require a human verdict and do not automatically become training annotations. Track IDs are not identities. Zone geometry is image-normalized and does not imply world-coordinate calibration.
