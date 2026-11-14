# Distribution comparison

RF100 test is frozen and excluded. This report compares Construction-PPE train+validation
with RF100 train+validation only. Pixel statistics are image-weighted and computed on
deterministic, aspect-preserving thumbnails no larger than 256 x 256. Contrast is the
per-image luminance p90-p10 range; sharpness is variance of a discrete grayscale
Laplacian. Full native annotations are used for overall box statistics; class comparisons
use only the five mapped classes.

## Image statistics (median)

| Statistic | Construction-PPE | RF100 |
|---|---:|---:|
| width | 2048.0000 | 900.0000 |
| height | 1360.0000 | 667.0000 |
| aspect_ratio | 1.5059 | 1.4993 |
| mean_luminance | 0.4630 | 0.5150 |
| luminance_std | 0.2278 | 0.2289 |
| contrast | 0.6118 | 0.6078 |
| saturation | 0.2098 | 0.2484 |
| laplacian_variance | 0.0150 | 0.0161 |

## Bounding boxes

| Dataset | Images | Boxes | Median boxes/image | Median relative area | Tiny | Small | Large |
|---|---:|---:|---:|---:|---:|---:|---:|
| Construction-PPE | 1000 | 14863 | 12.00 | 0.004583 | 0.272 | 0.402 | 0.327 |
| RF100 | 1116 | 7101 | 5.50 | 0.036434 | 0.014 | 0.225 | 0.761 |

## Overlapping class frequencies

| Class | Construction boxes | Construction overlap frequency | RF100 boxes | RF100 overlap frequency |
|---|---:|---:|---:|---:|
| helmet | 1576 | 0.2184 | 2348 | 0.3307 |
| vest | 1816 | 0.2517 | 1214 | 0.1710 |
| Person | 2443 | 0.3386 | 2603 | 0.3666 |
| no_helmet | 813 | 0.1127 | 105 | 0.0148 |
| no_vest | 567 | 0.0786 | 831 | 0.1170 |

## Co-occurrence scope

Co-occurrence is image-level dataset context only. It does not infer which PPE item belongs
to which person. Full values, per-class size proportions, image frequencies, and context
ratios are in `metrics.json`.

## Measured interpretation

RF100 images have lower native median resolution (900 x 667 versus 2048 x 1360), but nearly the same median aspect ratio. RF100 is brighter at the median (0.515 versus 0.463) and more saturated (0.248 versus 0.210). Median contrast and Laplacian variance are similar; the data do not support darker, lower-contrast, or blurrier RF100 imagery as the dominant dataset-wide explanation.

RF100 boxes are substantially larger: median relative area is 0.0364 versus 0.0046. Tiny boxes are 1.4% of RF100 annotations versus 27.2% in Construction-PPE. Therefore, the external degradation cannot be attributed primarily to a greater prevalence of small objects.

Class composition changes materially. RF100 contains proportionally more `helmet`, `Person`, and `no_vest`, while `no_helmet` is only 1.5% of overlapping-class boxes versus 11.3%. RF100 also has more multi-person images (59.1% versus 53.1%).
