# Temporal compliance benchmark

Repository-owned temporal state updates were measured over 2,000 timestamped observations
with present, unknown, and conflict transitions:

- Mean: `0.035 ms/observation`
- p50: `0.022 ms/observation`
- p95: `0.042 ms/observation`

This isolates the CPU temporal engine and confirms it is small relative to detector
inference. The existing synthetic video has no Person tracks, so an end-to-end temporal
comparison would measure only empty-engine and GPU/system noise; it is used as an
integration smoke test, not a meaningful temporal workload benchmark.

The 90-frame integration smoke test completed at `32.672 FPS`. Its empty temporal stage
measured `0.008 ms/frame` mean and `0.009 ms/frame` p95. It wrote all temporal artifacts,
but those figures must not be interpreted as populated-workload performance or real-video
temporal validation.
