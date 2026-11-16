# Local Docker deployment

The production container uses the official
`pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` image and intentionally runs one
Uvicorn worker. Docker Desktop must have NVIDIA GPU passthrough enabled.

## Prerequisites

- Docker Desktop with Compose and NVIDIA container support
- The verified production weight at
  `runs/experiment_mixed5_yolo26n_960/weights/best.pt`
- `ADMIN_USERNAME`, `ADMIN_PASSWORD_HASH`, and `SESSION_SECRET` supplied through
  the shell environment or a local `.env` file

Generate the password hash with `scripts/hash_admin_password.py`. Never put a
plaintext password or real secret in `compose.yaml`.

## Build and run

```powershell
docker compose build
docker compose up -d
docker compose ps
```

Compose bind-mounts the model at the path expected by the frozen detector
configuration and marks it read-only. It uses the named `visionops-data` volume
for `/var/lib/visionops`, which contains SQLite metadata, private uploads,
presentation media, and job artifacts.

Verify the runtime:

```powershell
docker compose exec visionops python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
docker compose exec visionops ffmpeg -version
docker compose exec visionops ffprobe -version
curl.exe http://127.0.0.1:8000/health/live
```

## Operations

```powershell
docker compose logs -f visionops
docker compose restart visionops
docker compose down
```

`docker compose down` retains the named application-data volume. Adding `-v`
deletes that volume and therefore removes the portfolio database and media; do
not use it unless permanent local data removal is intended.
