FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VISIONOPS_RUNTIME_DIR=/var/lib/visionops

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/visionops

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs

RUN python -m pip install --no-cache-dir -e ".[service]" \
    && python -c "import sys, torch, torchvision; assert (3, 11) <= sys.version_info[:2] < (3, 13); assert torch.__version__.startswith('2.5.1'); assert torchvision.__version__.startswith('0.20.1'); assert torch.version.cuda == '12.1'"

RUN groupadd --gid 10001 visionops \
    && useradd --uid 10001 --gid visionops --create-home --shell /usr/sbin/nologin visionops \
    && mkdir -p \
        /var/lib/visionops \
        /opt/visionops/runs/experiment_mixed5_yolo26n_960/weights \
    && chown -R visionops:visionops /var/lib/visionops \
    && chmod -R a=rX /opt/visionops

USER visionops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"

CMD ["uvicorn", "visionops.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
