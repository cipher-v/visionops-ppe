# Deterministic person-tracking diagnostics

These are injected-detection adapter checks, not MOTA, IDF1, HOTA, or formal MOT
evaluation. They are reproducible with:

```powershell
.\.venv\Scripts\python.exe scripts\run_tracking_diagnostics.py
```

## Results

| Scenario | Observed result |
|---|---|
| One smooth Person trajectory | One ID retained for all 20 observations |
| Two non-crossing trajectories | Two distinct IDs retained for all 20 frames |
| Two-frame disappearance with buffer 3 | Original ID recovered; one lost and one recovered transition |
| Disappearance beyond buffer 2 | Old ID expired; reappearance received a new, non-reused ID |
| Ten frames with no Person detections | Zero tracks and zero observations |

Reduced buffers are used only to make lifecycle diagnostics concise. Production remains at
30 processed frames.

## Crossing and occlusion

Two trajectories crossed over 25 frames. Near the crossing, the diagnostic supplied one
detection on frames 11 and 13 and no detections on frame 12. ByteTrack temporarily lost
both tracks, recovered both, and preserved the original trajectory IDs in this deterministic
case. It produced two lost and two recovered transitions, no expired tracks, and no new
IDs.

This single controlled result is not evidence that identities will always survive
crossing. ByteTrack here uses motion and IoU with no appearance/ReID, so severe overlap,
similar motion, long occlusion, camera motion, or detection errors may still cause an ID
switch, loss, or new ID. A ReID model is intentionally not added in this checkpoint.

The complete frame-level diagnostic record is in `diagnostic_results.json`.
