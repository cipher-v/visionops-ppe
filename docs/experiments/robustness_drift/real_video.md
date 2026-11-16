# Real-video drift observation

The existing frozen PyTorch detections from Checkpoint 15 were reused; the detector was
not rerun. Frames were decoded only for monitoring features and grouped by timestamps into
fixed five-second windows, so variable FPS is respected. These clips are unannotated and
support drift observation, not accuracy claims.

| Clip | Frames / FPS | Brightness | Contrast | Sharpness | Output summary | Window levels |
|---|---:|---:|---:|---:|---|---|
| compliant_multiworker | 315 / 25.0 | 168.8 | 52.6 | 4193 | 6.00 det/frame; 2 Persons/frame; no negative PPE | 3 high |
| helmet_transition | 297 / 25.0 | 126.9 | 62.0 | 7117 | 3.10 det/frame; negative PPE in every frame | 3 high |
| walking_multiworker | 577 / 23.976 | 99.8 | 48.5 | 1642 | 5.01 det/frame; negative PPE in 13.7% of frames | 5 high |

All 11 windows are high. Overall score ranges are 2.475 for compliant, 1.136-7.043 for
helmet transition, and 0.793-2.484 for walking. The portrait aspect ratio and unusually
regular class mix drive compliant-video differences; the changing negative-PPE class mix
dominates helmet transition; low sharpness/darker imagery and output variation contribute
for walking. This is evidence that the current reference and generic thresholds are too
sensitive for these legitimate deployment-like clips, not evidence that every frame is
wrong.

The observations are qualitatively consistent with earlier review: the walking clip had
occlusion-related unknown states and conflicts, and a sampled background safety-tape
`no_vest` false detection. A drift score did not cause or prove those failures. The full
window records are under `results/robustness_drift/video/`.

