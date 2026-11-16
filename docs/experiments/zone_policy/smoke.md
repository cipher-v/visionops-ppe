# Illustrative real-video integration smoke

The first 120 frames of `data/demo_videos/walking_multiworker.mp4` were processed with
the frozen production detector, BoT-SORT without ReID, PPE association, temporal
compliance, and `configs/zones/example_zones.yaml`.

This is only a system-integration check. The video has no authoritative site-zone
annotation, and the rendered `Illustrative Work Area` is not an actual construction-site
safety policy. Its geometry and debounce values were not tuned from incident outcomes.

- 120/120 requested frames processed with no read failures or dropped frames.
- 223 Person observations produced 223 zone-membership JSONL records.
- The annotated MP4 was written and reopened successfully.
- Polygon/name rendering, Person overlays, structured membership emission, and
  zone-aware temporal gating all remained functional.
- Both tracked workers' bottom-center anchors stayed outside this illustrative polygon in
  the bounded segment. Therefore there were no raw/stable entries and 13 negative helmet
  observations were recorded but suppressed as outside-policy evidence.
- No incident opened. This is expected technical gating for the observed anchors, not an
  accuracy conclusion.

Artifacts are under `results/zone_policy/illustrative_walking_smoke/` and remain ignored
by Git.
