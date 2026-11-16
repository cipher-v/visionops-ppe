# Portfolio deployment

VisionOps PPE includes a small public gallery and a private, single-administrator
workspace built on the existing FastAPI video runtime. It does not add another
inference path.

## Administrator configuration

Create a password hash interactively so the plaintext password is not placed in
shell history:

```powershell
.\.venv\Scripts\python.exe scripts\hash_admin_password.py
```

Set the three values shown in `.env.example` in the deployment environment.
`SESSION_SECRET` should be a long random value. If any of the three values is
missing, administrator authentication remains disabled.

Password hashes use PBKDF2-SHA256 with a random salt and 600,000 iterations.
The signed session cookie is `HttpOnly`, `SameSite=Lax`, expires after 12 hours,
and receives the `Secure` attribute when FastAPI observes HTTPS directly or via
Nginx's `X-Forwarded-Proto` header.

## Local production-style run

Install the service dependencies, export the administrator environment values,
and start Uvicorn:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[service]"
.\.venv\Scripts\python.exe -m uvicorn visionops.api.app:app --host 127.0.0.1 --port 8000 --proxy-headers
```

The default runtime data is stored under `results/service_runtime/`, including
the SQLite catalog, private uploads while a job is active, per-job original
presentation copies, and generated job artifacts. This directory is excluded
from Git. Original and annotated presentation media are served through checked
application routes, never from a public static directory; both require either
an administrator session or a completed, published catalog record.

FFmpeg and ffprobe are required for video jobs. The service resolves them from
`VISIONOPS_FFMPEG_PATH`, the system `PATH`, or a local Windows WinGet FFmpeg
package. Compatible H.264/yuv420p originals are stream-copied into faststart
MP4 presentation files; incompatible originals and OpenCV-rendered annotated
artifacts are converted to H.264/yuv420p presentation MP4s after inference.
Engineering pipeline artifacts remain unchanged and private.

## Linux service preparation

- `deploy/nginx.conf` proxies requests to Uvicorn, forwards the original scheme,
  allows the existing 250 MB upload ceiling, and keeps generated media behind
  application authorization.
- `deploy/visionops.service` assumes the checkout is at `/opt/visionops-ppe`, the
  virtual environment is `/opt/visionops-ppe/.venv`, the service account is
  `visionops`, and secrets are supplied by `/etc/visionops-ppe.env`.
- TLS termination is expected to be added to Nginx for an internet deployment.

SQLite is appropriate for this first single-process portfolio deployment.
Queued jobs are held by the existing in-process runtime; an application restart
does not resume a job that was still queued or running. Completed metadata and
generated files remain available after restart.
