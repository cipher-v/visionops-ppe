# Export record

The source is the frozen Candidate C checkpoint:

- Path: `runs/experiment_mixed5_yolo26n_960/weights/best.pt`
- SHA-256: `BF99E872A9B57A7C33A4EBF06F0620F9EB8A215AEC29DDE2A90CA1F72F9581F4`
- Architecture: YOLO26n detect, five classes
- Input: static `[1, 3, 960, 960]`, FP32, batch 1

The accepted export invocation used Ultralytics 8.4.212, ONNX 1.19.1, opset 17,
`dynamic=False`, `simplify=False`, and no embedded NMS. In this Ultralytics version,
explicit `nms=False` selects YOLO26's one-to-one end-to-end head. A diagnostic first export
therefore changed production semantics and was rejected. Leaving `nms` unset preserved the
one-to-many production head and runtime postprocessing.

Correct ONNX artifact:

- Path: `runs/experiment_mixed5_yolo26n_960/weights/best.onnx`
- SHA-256: `FCC24211F719B5BEB2B889DFD8F28EA3123403844B72F7C1128866772CFA11A5`
- Size: 10,020,230 bytes
- Output shape: `[1, 9, 18900]`
- ONNX checker: passed
- Runtime: ONNX Runtime GPU 1.23.2, CUDAExecutionProvider
- Build GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU

The binary remains ignored by Git. Its full contract is in
`configs/deployment_backends.yaml`.

## TensorRT

TensorRT FP16 was not built. There is no installed TensorRT Python runtime, TensorRT DLL
set, or `trtexec`. The machine has CUDA Toolkit 11.8, PyTorch CUDA 12.1, and a driver
advertising CUDA 12.3. ONNX Runtime listing a TensorRT provider does not mean its external
TensorRT libraries are installed. Adding another CUDA/TensorRT stack was not forced into
the known-good environment. Consequently no engine hash, size, or TensorRT metric is
claimed.
