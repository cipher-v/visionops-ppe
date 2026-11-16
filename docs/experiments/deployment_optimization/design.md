# Deployment optimization design

Checkpoint 15 evaluates deployment backends for the exact frozen YOLO26n Candidate C
detector. PyTorch is the behavioral reference. The detector remains static batch 1 at
960×960 with the existing letterbox preprocessing, confidence floor, class-specific
thresholds, and IoU 0.70 runtime postprocessing policy.

The parity set was declared before measurements and contains 42 inputs: 15
Construction-PPE validation images, 15 filtered RF100 validation images, and four fixed
frames from each of the three internal real videos. RF100 test is excluded. Dataset images
were selected deterministically with class-aware coverage before filling lexicographically;
video indices are start, quarter, midpoint, and final frame. Paths and hashes are recorded
in `parity_set.json`.

Acceptance criteria were declared before results:

- ONNX FP32 final-detection match rate ≥99.5%
- Mean matched IoU ≥0.995
- p95 absolute confidence delta ≤0.005

Matching is deterministic, greedy one-to-one within class, ordered by descending IoU,
with a minimum matching IoU of 0.90. Match rate is matched detections divided by the
larger backend detection count.

Detector benchmarking uses five warmups followed by three measured passes over all 42
preloaded inputs. CUDA is synchronized around wall-clock samples. Native video profiling
uses all frames at original resolution with production BoT-SORT, PPE association, temporal
compliance, rendering, and encoding. Zones are disabled identically for both backends.

TensorRT FP16 was conditional on a clean local runtime. No INT8 work was performed.
