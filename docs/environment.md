# Checkpoint 1 environment

The verified local environment uses Python 3.12.6 on Windows 11. The project supports Python 3.11 and 3.12 through `requires-python = ">=3.11,<3.13"`.

## GPU and PyTorch selection

`nvidia-smi` detected an NVIDIA GeForce RTX 3050 6GB Laptop GPU (6,144 MiB, compute capability 8.6) with NVIDIA driver 546.18. The driver reports CUDA 12.3 compatibility. The environment therefore uses the official PyTorch CUDA 12.1 wheels, which are compatible with this driver:

```powershell
python -m pip install torch==2.5.1 torchvision==0.20.1 `
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e ".[dev]"
```

Verified versions:

- Python 3.12.6
- pip 26.3
- setuptools 78.1.0
- PyTorch 2.5.1+cu121
- torchvision 0.20.1+cu121
- Ultralytics 8.4.212
- NumPy 2.5.4
- Matplotlib 3.11.2
- Pillow 13.0.0
- PyYAML 6.0.3
- pytest 9.2.0
- Ruff 0.17.5

PyTorch reports CUDA runtime 12.1, `torch.cuda.is_available() == True`, and device `NVIDIA GeForce RTX 3050 6GB Laptop GPU`.
