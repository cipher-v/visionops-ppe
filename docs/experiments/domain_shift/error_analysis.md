# Error and confidence analysis

## Fixed protocol

- Confidence: 0.25
- NMS IoU: 0.7
- Correct/confusion overlap IoU: 0.5
- Localization diagnostic overlap: 0.1 <= IoU < 0.5
- Assignment precedence: correct, classification confusion, localization failure, miss
- Each prediction may be assigned to at most one ground-truth object
- RF100 test is not read or evaluated

## Error counts

| Model | Dataset | Correct | Localization | Classification | Miss | False positive |
|---|---|---:|---:|---:|---:|---:|
| yolo26n | construction_ppe_validation | 1247 | 19 | 22 | 136 | 103 |
| yolo26n | rf100_validation | 603 | 39 | 19 | 54 | 132 |
| yolo26s | construction_ppe_validation | 1271 | 30 | 23 | 100 | 70 |
| yolo26s | rf100_validation | 603 | 33 | 18 | 61 | 101 |

Confidence quantiles, normalized confusion matrices, and class-by-size error matrices are
recorded in `metrics.json`. A missed object has no detection confidence; confidence for
localization and classification errors is reported when an assigned marginal prediction
exists.

## Measured interpretation

For YOLO26n, correct detections fall from 87.6% in-domain to 84.3% on RF100 validation. Localization errors rise from 1.3% to 5.5%, and false positives per GT rise from 7.2% to 18.5%. Complete misses do not increase for YOLO26n at this operating point.

For YOLO26s, correct detections fall from 89.3% to 84.3%; false positives per GT rise from 4.9% to 14.1%. YOLO26s produces fewer RF100 false positives and localization errors than YOLO26n, but more complete misses, and both have exactly 603 correct RF100-validation assignments.

Median true-positive confidence falls only modestly: 0.898 to 0.884 for YOLO26n and 0.924 to 0.905 for YOLO26s. In contrast, median false-positive confidence increases from 0.382 to 0.581 for YOLO26n and 0.393 to 0.583 for YOLO26s. This is not a uniform confidence collapse; external false predictions are more confident.

The robust size/class regressions include `helmet` small, `vest` large, `Person` large, and `no_vest` large. Very large percentage changes in RF100 tiny `Person` and `no_helmet` bins have only 2-9 objects and must not drive intervention selection. `no_vest` confused as `vest` rises externally for both models. Measured errors are dominated by false positives and increased localization/unmatched errors rather than wrong-class confusion alone.

## Evidence-based next intervention category

**More diverse training data** should be tested next. The shift spans scene composition, annotation/class frequencies, confident false positives, localization, and multiple well-populated class/size cells while RF100 objects are generally larger and photometric medians are not severely shifted. Model capacity did not close the independent gap. Photometric or scale augmentation alone is therefore less directly supported; explicit domain adaptation can remain a later predeclared alternative if broader representative training data are insufficient.

## Feature-space analysis

Skipped. Stable intermediate YOLO26 feature extraction would require coupling this audit
to private model-layer structure, making the result fragile and invasive. No unrelated
embedding model or dependency was introduced.
