# Bounded tracker robustness comparison

## Trigger and purpose

Checkpoint 13 exposed two specific continuity failures in the frozen ByteTrack baseline:
a short duplicate Person track in `helmet_transition` and a probable identity switch during
overlap in `walking_multiworker`. This experiment compares only supported Ultralytics
BoT-SORT variants on the same inputs. It is not a hyperparameter search and has no formal
MOT ground truth.

## Frozen system

The detector, PPE association, temporal engine, and baseline tracker configurations remain
unchanged. Their Checkpoint 13 hashes are:

| Configuration | SHA-256 |
|---|---|
| `production_detector.yaml` | `A5CB6E081E61E3316888AD35168D29CA7F3417E1F751C1032216517FCAFA1560` |
| `bytetrack_person.yaml` | `9E3EAB03889E86C3F5A3EFD74B9E7C3918EC95D497C07CB1E32E3A267ED6B42D` |
| `ppe_association.yaml` | `258B1C269D7340F535CB47F2D694224C856DB20F6EDF33B584EA48E7681CF048` |
| `compliance_temporal.yaml` | `C5154F40EEFB20C698223B7AF2ECEFD3560C5F84770480DEEEA7BE74478FD9DA` |

All clips use frame stride 1. Their SHA-256 values must match Checkpoint 13:

- `compliant_multiworker.mp4`: `B9183548A9EC8BFDDD2298BEA144476AA14C29DC7E67C7CE009A2821D0567EF5`
- `helmet_transition.mp4`: `423513545128921B90FCABED8A9207EEA96678A8F6CEFE6639DB959F97D92120`
- `walking_multiworker.mp4`: `DF2F645CAAD8D90EEE93AD1C5B012DE4394981E5EEABD92E37BFC16101EA2FA4`

## Installed support inspected before alternatives

Ultralytics 8.4.212 includes `BOTSORT` and its official `botsort.yaml`. Defaults are
`track_high_thresh=0.25`, `track_low_thresh=0.10`, `new_track_thresh=0.25`,
`track_buffer=30`, `match_thresh=0.80`, `fuse_score=true`,
`gmc_method=sparseOptFlow`, `proximity_thresh=0.50`, and
`appearance_thresh=0.80`. ReID is supported through `with_reid` and either detector-native
features (`model=auto`) or an external encoder.

The repository supplies postprocessed detections to the tracker, so detector-native
features are unavailable. `model=auto` must not be described as active ReID in this adapter.
An external official encoder may be evaluated only if it initializes cleanly without an
unrelated framework.

## Predeclared primary windows

These windows were fixed from the ByteTrack outputs before running alternatives:

| Case | Exact failure | Window | Comparison frames |
|---|---|---|---|
| Helmet duplicate | Track 2 exists for frames 106-109 over primary Track 1 | frames 104-112, 4.160-4.480 s at 25 FPS | 104, 106, 108, 110, 112 |
| Walking crossing | Track 4 begins at frame 269 as Track 2 changes apparent worker | frames 252-300, 10.511-12.513 s at 23.976 FPS | 252, 268, 269, 284, 300 |

The primary questions are whether one ID remains on the helmet worker without a duplicate,
and whether distinct IDs remain attached to the dark-clothed and red-vest workers through
the walking overlap. The compliant clip at frames 0, 157, and 314 is the non-regression
case. Longest walking tracks remain secondary continuity references.

## Selection rule

An alternative is selected only for a clear improvement in the two primary windows without
stable-clip regression, new obvious fragmentation, unreasonable downstream behavior, or
unreasonable cost. Aggregate track counts cannot substitute for visual identity review. If
no candidate clearly wins, ByteTrack remains production.

## Provenance policy

Video provenance and license remain unresolved. Public-demo eligibility is false, and raw
videos plus extracted frames remain ignored by Git.
