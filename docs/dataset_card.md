# Dataset card

Datasets are downloaded locally and ignored by Git; this repository does not redistribute them.

## Ultralytics Construction-PPE

- Source: [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe)
- License: CC BY 4.0
- Split: 800 training, 200 validation images; no test split
- Format: YOLO normalized boxes
- Total boxes: 14,863

| Class | Boxes | Class | Boxes |
|---|---:|---|---:|
| helmet | 1,576 | gloves | 1,929 |
| vest | 1,816 | boots | 2,484 |
| goggles | 751 | Person | 2,443 |
| no_helmet | 813 | no_goggle | 347 |
| no_gloves | 1,042 | no_boots | 747 |
| forklift | 348 | no_vest | 567 |

The final detector uses only helmet, vest, Person, no_helmet, and no_vest. Missing-PPE labels mean the relevant body region is visible and PPE is absent; occluded or out-of-frame body parts do not establish a violation. Limitations include generated/human-collected mixture, imbalance, tiny objects, small annotation boundary tolerances, and no independent test split.

## RF100 Construction Safety v2

- Source: [Roboflow Universe RF100 Construction Safety v2](https://universe.roboflow.com/roboflow-100/construction-safety-gsnvb/dataset/2)
- License: CC BY 4.0
- Archive date: 2022-08-30
- Splits: 997 train, 119 validation, 90 test
- Classes mapped explicitly: helmet→helmet, vest→vest, person→Person, no-helmet→no_helmet, no-vest→no_vest

The project found one validation near-duplicate with inconsistent labels and excluded it, creating a 118-image filtered validation view without changing source files. RF100 was initially external diagnostic data; its training split later entered mixed-source training, making filtered validation target-domain validation for the final model.

RF100 test cannot serve as pristine final evidence: it is only 90 images and was examined earlier during model-family external-generalization diagnosis. It was not used for final threshold calibration or mixed-model selection, but a new sealed deployment-representative set is still required.
