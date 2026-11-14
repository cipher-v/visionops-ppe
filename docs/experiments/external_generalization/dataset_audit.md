# External dataset audit

## Provenance

- Dataset: RF100 Construction Safety, version 2 (`release`)
- Source: https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb/dataset/2
- License: CC BY 4.0
- Export timestamp recorded by the archive: 2022-08-30 10:11 GMT
- Local source: `E:\visionops-ppe\data\external\construction-safety-gsnvb-v2`
- Archive: `construction safety.v2-release.yolov5pytorch.zip` (182128587 bytes)
- Archive SHA-256: `0CB3FFDAFB40C0F00D2D708631773F3ECFCA3977A27909CEF5806683D7FA6A2E`
- Annotation format: YOLO normalized `class_id x_center y_center width height`

The extracted source is retained unchanged. Mapped labels and image hard links/copies live
only under the Git-ignored generated-data directory.

## Integrity

| Split | Images | Labels | Boxes |
|---|---:|---:|---:|
| train | 997 | 997 | 6386 |
| valid | 119 | 119 | 715 |
| test | 90 | 90 | 623 |

Integrity problems found: **0**. No missing pairs, corrupt images, malformed
annotations, out-of-range coordinates, or non-positive boxes were found.

## Class mapping and counts

| RF100 class | Total boxes | Construction-PPE class |
|---|---:|---|
| helmet | 2543 | helmet |
| no-helmet | 129 | no_helmet |
| no-vest | 892 | no_vest |
| person | 2817 | Person |
| vest | 1343 | vest |

This mapping is explicit and no other Construction-PPE classes are evaluated.

## Leakage audit

Every one of the 1,206 external images was compared with all 1,000 Construction-PPE
train/validation images. Exact matching used file SHA-256. Near-duplicate matching used a
deterministic 64-bit grayscale difference hash after EXIF orientation; the threshold was
predeclared as Hamming distance <= 4. Exact matches:
**0**. Near matches: **0**.
Excluded external images: **0**. Any match is
excluded from independent evaluation before metrics are computed.

## Evaluation policy

The official 90-image test split is the primary independent evaluation. It is small, so
metrics should be interpreted with that sampling limitation. A secondary result over all
leakage-free external images is reported separately and is not presented as independent.
No thresholds, model weights, augmentation, or dataset annotations are tuned or repaired.
