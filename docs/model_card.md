# Model card: VisionOps PPE detector v1

## Purpose and intended use

The model supplies visible PPE/person detections to a prototype safety-assistance pipeline. It is intended for controlled evaluation, operator decision support, and engineering demonstrations. It is not a certified safety system, a replacement for site supervision, or evidence for disciplinary/employment decisions.

## Model

- Architecture: Ultralytics YOLO26n detection model
- Input: aspect-preserving letterbox, 960 pixels
- Classes: `helmet`, `vest`, `Person`, `no_helmet`, `no_vest`
- Production backend: PyTorch FP32
- Parameters: 2,535,830
- Weight SHA-256: `BF99E872A9B57A7C33A4EBF06F0620F9EB8A215AEC29DDE2A90CA1F72F9581F4`

The weight is not stored in Git. Runtime startup verifies its hash against the production configuration and model manifest.

## Training data

Final mixed-source training used 800 five-class-mapped Construction-PPE images plus 997 RF100 Construction Safety v2 training images: 1,797 unique images per epoch, 100 epochs, deterministic seed 42. Construction-PPE validation and leakage-filtered RF100 validation were excluded from training. Dataset details and licenses are in [dataset_card.md](dataset_card.md).

## Calibration and results

The frozen class thresholds are helmet 0.32, vest 0.19, Person 0.41, no_helmet 0.27, and no_vest 0.11; NMS IoU is 0.70. Calibration balanced Construction-PPE source validation and filtered RF100 target validation without using RF100 test.

| Domain | mAP50-95 | Operational F1 |
|---|---:|---:|
| Construction-PPE validation | 0.6642 | 0.9020 |
| Filtered RF100 validation | 0.4948 | 0.9073 |

RF100 validation is held-out target-domain evidence for the mixed model, not universally unseen data. RF100 contains only 11 filtered validation `no_helmet` instances, so that class is especially uncertain.

## Known weaknesses

- Domain shift was severe for source-only models; mixed training improved but did not solve it.
- Gaussian noise and motion blur caused the largest synthetic degradation.
- Tiny/occluded PPE, visible missing-PPE evidence, crowded overlap, and background lookalikes remain difficult.
- Missing positive PPE is not proof of a violation; downstream association uses explicit `unknown` semantics.
- The three internal videos are qualitative integration validation, not accuracy ground truth.
- Simple drift PSI is not an accuracy predictor.

## Out-of-scope and safety cautions

Do not use this prototype for biometric identification, worker identity, autonomous enforcement, legal compliance certification, or unreviewed adverse decisions. Camera placement, population, PPE policy, lighting, environment, and privacy controls require site-specific assessment. Every incident should remain explainable and reviewable by an authorized human.
