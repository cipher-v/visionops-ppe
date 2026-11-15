# Real-video validation design

## Purpose

Checkpoint 13 observes the complete frozen VisionOps PPE chain on three locally supplied
videos: detector, Person-only ByteTrack, geometric Person-to-PPE association, temporal
compliance, incident events, and annotated output. It is an operational and qualitative
validation, not an accuracy benchmark: the clips have no formal detection, identity,
association, or compliance ground truth.

## Frozen-system rule

Every frame was processed (`frame_stride=1`) with the production model and these unchanged
configuration files:

| Configuration | SHA-256 |
|---|---|
| `configs/production_detector.yaml` | `A5CB6E081E61E3316888AD35168D29CA7F3417E1F751C1032216517FCAFA1560` |
| `configs/bytetrack_person.yaml` | `9E3EAB03889E86C3F5A3EFD74B9E7C3918EC95D497C07CB1E32E3A267ED6B42D` |
| `configs/ppe_association.yaml` | `258B1C269D7340F535CB47F2D694224C856DB20F6EDF33B584EA48E7681CF048` |
| `configs/compliance_temporal.yaml` | `C5154F40EEFB20C698223B7AF2ECEFD3560C5F84770480DEEEA7BE74478FD9DA` |

No confidence threshold, tracker setting, anatomical region, association gate/weight, or
temporal duration was changed after viewing the clips.

## Video provenance

The project owner supplied `compliant_multiworker.mp4`, `helmet_transition.mp4`, and
`walking_multiworker.mp4` locally under `data/demo_videos/`. Original URLs, creators, and
license/redistribution terms were not supplied. Raw videos and extracted frames therefore
remain ignored by Git. Exact hashes and technical properties are recorded in
`data/demo_videos/README.md` and `metrics.json`.

## Method

Each clip was run with tracking, PPE association, temporal compliance, real-video metadata,
and annotated rendering enabled. The reporting script parses the resulting JSONL files,
combines raw per-frame latency samples, and generates compact review selections. Contact
sheets prioritize longest tracks, tracks with gaps, and incident tracks. Incident sheets
show the nearest frames to confirmation minus one second, confirmation, and confirmation
plus one second.

The review manifest deliberately leaves `human_verdict` blank. Model output selects cases
for inspection but does not grade itself.

## Limitations

- Three short, curated clips cannot establish field accuracy or generalization.
- There is no annotated ground truth for misses, false detections, ID switches, assignments,
  or compliance incidents.
- Review observations are human qualitative judgments over selected frames, not metrics.
- Video licensing is unresolved, so outputs are unsuitable for public redistribution until
  provenance is documented or the media is replaced.
- High-resolution input makes rendering and video encoding part of end-to-end latency.
