"""Safe runtime health payloads."""

import torch


def readiness(runtime: object) -> dict[str, object]:
    config = runtime.config
    return {
        "status": "ok",
        "backend": "pytorch",
        "model_architecture": config.architecture,
        "model_hash_prefix": config.weights_sha256[:12],
        "device": str(runtime.device),
        "cuda_available": torch.cuda.is_available(),
    }
