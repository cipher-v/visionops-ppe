FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VISIONOPS_RUNTIME_DIR=/var/lib/visionops

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/visionops
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1 torchvision==0.20.1 && \
    python -m pip install --no-cache-dir ".[service]"

RUN useradd --create-home --uid 10001 visionops && \
    mkdir -p /var/lib/visionops /opt/visionops/runs && \
    chown -R visionops:visionops /var/lib/visionops /opt/visionops
USER visionops

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"

CMD ["uvicorn", "visionops.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
