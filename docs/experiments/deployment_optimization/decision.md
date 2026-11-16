# Backend decision

PyTorch remains the production backend and behavioral reference.

The decision follows the predeclared order:

1. ONNX failed final-detection parity.
2. ONNX caused downstream tracking and incident-timestamp divergence.
3. ONNX was slower in the repeated detector-only benchmark and slightly slower overall.
4. Comparable ONNX GPU memory was not reliably available.
5. ONNX adds runtime/deployment complexity without a demonstrated benefit here.

The new abstraction keeps `--detector-backend pytorch` as the default and supports explicit
ONNX selection for continued engineering investigation. It does not duplicate tracking,
association, temporal, or zone logic. The ONNX option is not approved for production under
the current artifact/runtime combination.

TensorRT FP16 remains untested because a clean compatible runtime was absent. A future
checkpoint may evaluate it in an isolated, pinned environment, but no TensorRT or INT8
performance claim is made here.
