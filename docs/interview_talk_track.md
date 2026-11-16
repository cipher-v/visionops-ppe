# Interview talk track

## 30 seconds

VisionOps PPE is an end-to-end safety-assistance prototype, not just a YOLO notebook. I developed a five-class detector, measured a major domain gap, showed with a matched-exposure control that relevant data diversity mattered more than a larger model, then built tracking, person-to-PPE association, temporal incidents, zones, monitoring, a human review queue, and a hash-verified FastAPI service. I kept negative results—ONNX parity failure and weak drift/accuracy correlation—and used them to constrain the production design.

## 2 minutes

The first 640 baseline improved at 960, especially on tiny PPE, but a YOLO26s model doubled inference cost without improving the external RF100 test result. Distribution/error analysis showed the gap was not simply blur or smaller boxes, so I introduced RF100 training data. Source-only RF100 validation mAP50-95 was 0.3349; an exposure-matched mixed model reached 0.4722 and the full mixed model 0.4948 while retaining 0.6642 on Construction-PPE. I froze Candidate C and calibrated class thresholds to 0.9020 and 0.9073 operational F1 across the two validation domains.

For video, detections feed BoT-SORT, geometry-based PPE association, and timestamp-based compliance states. Missing evidence stays unknown. Zones can determine which PPE is required, and incidents are structured open/close events. Internal footage exposed identity transfer and occlusion; a bounded tracker comparison selected BoT-SORT without ReID. ONNX failed strict behavioral parity and speed gates, so PyTorch stayed production. Robustness testing found noise/motion blur weaknesses, while PSI drift had poor accuracy correlation, so it is telemetry only. FastAPI, metrics, Docker packaging, and a human-only review queue complete the engineering surface.

## 5-minute technical walkthrough

1. **Problem framing:** frame detections alone cannot safely decide compliance; evidence must be assigned to a person and persist in time.
2. **Data/model controls:** audited 1,000 Construction-PPE images and RF100, mapped five common classes, corrected one validation near-duplicate, and fixed deterministic protocols.
3. **Experiments:** 960 helped tiny objects; larger capacity helped source data but not external generalization; mixed-domain data produced the decisive target gain; matched exposure supported a diversity interpretation.
4. **Operating policy:** froze the model before class-threshold calibration; reported sparse-class limitations rather than optimizing them away.
5. **Video system:** tracked anonymous people, associated PPE by interpretable geometry, preserved unknown/conflict, accumulated elapsed-time evidence, and emitted zone-aware incidents.
6. **Production decisions:** selected BoT-SORT after a real failure, rejected ReID/ONNX when evidence did not support them, kept TensorRT unclaimed, serialized GPU-heavy work, and hash-verified weights.
7. **Monitoring/review:** measured perturbation degradation, showed drift was not a failure predictor, and reduced 1,137 uncertain events to 28 review candidates without auto-labeling.
8. **Limitations:** no sealed camera-domain test set, no formal MOT/incident accuracy, unresolved public video provenance, and no certified safety claim.

## Likely questions

**Why not YOLO26s?** It improved source mAP50-95 from 0.6012 to 0.6343 but roughly doubled latency and did not improve RF100 test mAP50-95 (0.3012 vs 0.3029).

**How do you know diversity, not training volume, helped?** An exposure-matched mixed model reached 0.4722 RF100 validation mAP50-95 versus 0.3349 source-only, recovering most of the full mixed model's 0.4948.

**Why is missing PPE not inferred from missing detection?** Occlusion and out-of-frame regions make absence of a box ambiguous; explicit `no_helmet`/`no_vest` evidence is required, otherwise state is unknown.

**Why reject ONNX?** The exported runtime changed final detections/downstream behavior and was slower in the controlled local benchmark, violating predeclared parity-first gates.

**Does drift indicate failure?** No. PSI detected shifts but correlations with degradation were weak and it over-alerted on real clips; it is diagnostic telemetry only.

**Is this production deployed?** No. It is a production-oriented prototype with serving and safeguards, but lacks a sealed deployment-domain evaluation, platform hardening, and certified operational validation.
