# Engineering decision log

| Finding | Evidence | Decision |
|---|---|---|
| Larger YOLO26s improved source metrics but not external performance | External mAP50-95 0.3012 vs YOLO26n 0.3029; ~2× latency | Keep YOLO26n |
| Data diversity mattered more than capacity | Source-only RF100 val 0.3349; exposure-matched mixed 0.4722 | Invest in relevant data |
| Duplicate leakage existed | One inconsistent dHash-identical RF100 validation image | Exclude from filtered validation; retain source unchanged |
| Diversity gain was not only more optimization | Matched-exposure D recovered most of C's target gain | Treat intervention as domain diversity evidence |
| ByteTrack transferred identity in walking overlap | Visual bounded review plus track diagnostics | Compare trackers without tuning sweep |
| BoT-SORT reduced the primary transfer | Stable clip preserved; downstream remained sensible | Freeze BoT-SORT |
| ReID added cost without visible benefit | Same known failures; slightly altered unknown output | Do not select ReID |
| ONNX failed strict parity and speed gates | Detection and downstream divergence; slower controlled detector run | Keep PyTorch |
| TensorRT environment was incompatible | No clean pinned runtime available | Do not force or claim TensorRT |
| Noise/motion blur were major weaknesses | Severe noise −35.8%/−42.9% relative mAP across domains | Document and monitor; no post-hoc retuning |
| PSI detected change, not failure | Weak/negative Pearson and near-zero Spearman correlations | Diagnostic-only telemetry |
| Review selection is uncertain evidence | Deduped candidates include errors and correct hard examples | Human verdict only; no automatic retraining |

Negative results are preserved because they directly shaped the final architecture and prevent unsupported production claims.
