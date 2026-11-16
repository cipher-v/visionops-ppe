# Monitoring cost and limitations

On validation images, monitoring feature extraction averaged 4.92 ms for clean CPPE and
4.57 ms for clean RF100. On real video it averaged 5.99 ms/frame (p50 5.79, p95 8.12) for
compliant, 4.84 ms (p50 4.80, p95 6.76) for helmet transition, and 4.57 ms (p50 4.44,
p95 6.35) for walking. Five-second window aggregation/scoring averaged 2.22-2.62 ms per
window with worst clip p95 3.58 ms. The proxy resize keeps this well below the existing
PyTorch detector mean of roughly 54 ms/frame; video decode is excluded from feature cost.

Limitations:

- Synthetic perturbations do not represent all camera, weather, compression, or site shifts.
- Drift does not prove model failure, and low drift does not guarantee correctness.
- There is no sealed production deployment distribution.
- Alert thresholds are engineering defaults, not universal statistical guarantees.
- The three real clips are limited and unannotated.
- Histogram and sharpness features can be camera-specific.
- The combined reference still warns on clean RF100 and alerts high on every real-video
  window, demonstrating substantial false-alarm risk.
- A maximum-feature aggregation is interpretable but can be dominated by one harmless
  shift and can miss degradation not represented by these features.

