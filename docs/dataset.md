# Ultralytics Construction-PPE dataset

## Source and attribution

VisionOps PPE uses the official [Ultralytics Construction-PPE dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe). It was created and open-sourced by the Ultralytics team in November 2026. The dataset combines human-collected and generated images; Ultralytics reports that expert human annotation and AI auto-annotation were completed in November 2026.

Suggested attribution: *Ultralytics Construction-PPE Dataset, Ultralytics team, November 2026, CC BY 4.0.*

The dataset images and annotations are licensed under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0). This dataset license does not select or imply a license for the VisionOps PPE repository.

## Structure

The fixed split contains 1,000 YOLO-format object-detection images:

| Split | Images | Purpose |
| --- | ---: | --- |
| Train | 800 | Model development |
| Validation | 200 | Evaluation during development |

There is no independent test split. Final evaluation will therefore require a separately collected, deployment-representative test set that is not used during development.

Images are stored under `images/train` and `images/val`; matching YOLO label files are under `labels/train` and `labels/val`. Each annotation row has `class_id x_center y_center width height`, with coordinates normalized relative to image width and height.

## Classes and annotation meaning

| ID | Class | Annotation target |
| ---: | --- | --- |
| 0 | `helmet` | Protective helmet or hard hat |
| 1 | `gloves` | Work gloves |
| 2 | `vest` | High-visibility safety vest, jacket, or shirt |
| 3 | `boots` | Protective work boots or work shoes |
| 4 | `goggles` | Protective eyewear |
| 5 | `Person` | Person |
| 6 | `no_helmet` | Visible head without a helmet |
| 7 | `no_goggle` | Visible eye region without protective eyewear |
| 8 | `no_gloves` | Visible hand without a glove |
| 9 | `no_boots` | Visible foot or footwear without protective work footwear |
| 10 | `forklift` | Forklift |
| 11 | `no_vest` | Visible torso without high-visibility safety clothing |

The missing-PPE classes have an important semantic constraint: they mark a **visible body region where the corresponding PPE is absent**. An occluded or out-of-frame head, eye region, hand, foot, or torso does not by itself establish missing PPE. Downstream evaluation must not interpret lack of a positive equipment box as proof of non-compliance.

## Suitability and limitations

The dataset directly represents the core VisionOps PPE domain: people, forklifts, five worn-equipment types, and explicit visible missing-equipment annotations across construction sites, workshops, warehouses, loading areas, and maintenance work. Its official YOLO annotations, varied image sizes and viewpoints, and fixed development split make it suitable for building and auditing the initial object-detection baseline.

It is not sufficient as a final deployment benchmark. The images mix human-collected and generated content, may not match a target camera domain, and provide no independent test split. Class imbalance, small objects, occlusion, and the visible-region semantics of missing-PPE labels must be considered when evaluating later models.
