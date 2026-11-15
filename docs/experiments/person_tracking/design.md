# Person-only multi-object tracking design

## Scope

Checkpoint 10 tracks only class 2, `Person`. Helmet, vest, `no_helmet`, and
`no_vest` remain independent frame-level detections. No code associates PPE boxes with a
person, decides compliance, or creates an incident. Track IDs are anonymous run-local
labels and are not real-world identities.

## Tracker selection

The adapter uses `ultralytics.trackers.byte_tracker.BYTETracker` from installed
Ultralytics 8.4.212. This is the tracker implementation supported by the existing model
stack and avoids copying another ByteTrack implementation. It requires the lightweight
`lap==0.5.12` assignment dependency. Appearance/ReID is disabled.

The checked-in `configs/bytetrack_person.yaml` uses the Ultralytics ByteTrack defaults:

| Setting | Value |
|---|---:|
| High match threshold | 0.25 |
| Low second-stage threshold | 0.10 |
| New-track threshold | 0.25 |
| Match cost threshold | 0.80 |
| Lost-track buffer | 30 processed frames |
| Fuse detection score | true |
| Appearance/ReID | false |

These tracker thresholds do not replace or lower detector confidence. The repository first
applies the frozen `Person=0.41` detector threshold, and only those surviving Person boxes
are passed to ByteTrack. PPE boxes never enter the tracker. Because every input Person box
is at least 0.41, it enters ByteTrack's high-confidence stage despite the tracker's lower
internal boundaries.

## Integration

`PersonByteTracker` is independent of YOLO inference and rendering. It accepts structured
frame detections, validates frame order and boxes, filters Person input, calls ByteTrack
once, and returns structured active observations plus lifecycle transitions. The video
pipeline performs its existing single model call, writes unchanged `detections.jsonl`, and
optionally passes the Person subset to the adapter. There is no duplicate inference.

Tracking remains opt-in with `--enable-tracking`. Disabled runs preserve Checkpoint 9
behavior. Enabled runs always write `tracks.jsonl` and `annotated.mp4`; Person annotations
are `Person #<run-local-id> <confidence>`, while PPE labels remain ordinary detections.

## Track schema

Each record contains frame index, timestamp, run-local track ID, class ID/name,
confidence, original-frame pixel box, normalized box, source ID, and one of `new`,
`tracked`, or `recovered`. It contains no PPE or compliance fields.

## Lifecycle semantics

- A sufficiently matched Person detection creates a new monotonic track ID. Late-frame
  ByteTrack candidates may require confirmation before they are emitted.
- A matched active track retains its ID and is emitted as `tracked`.
- An unmatched track becomes lost and remains buffered for 30 processed frames.
- A buffered track matched again retains its ID and is emitted as `recovered`.
- Once the buffer is exceeded, the track expires. Later detections receive a new ID after
  normal ByteTrack confirmation; expired IDs are not reused within the run.
- Empty Person frames update the tracker normally and do not invent observations.
- Tracks still active or buffered when input ends are reported as live-at-end; resources
  close without synthesizing terminal observations.

With frame stride greater than one, the buffer is measured in processed frames, not every
source frame. IDs are never persisted across separate video or camera sessions.

## Failure behavior

Missing or malformed tracker configuration fails before frame processing. Tracker
initialization errors are wrapped with a clear repository exception. Non-finite,
zero-area, below-threshold Person boxes are rejected and counted. PPE boxes are ignored by
the tracker. Frame indices must increase and timestamps cannot move backward.

## CLI

```powershell
.\.venv\Scripts\python.exe scripts\run_video_inference.py `
  --source path\to\video.mp4 `
  --enable-tracking `
  --output-dir results\video_inference\tracked_demo
```
