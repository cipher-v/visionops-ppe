"""Public portfolio pages and authenticated single-admin video management."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from visionops.api.auth import (
    SESSION_COOKIE,
    WebSettings,
    create_session,
    is_https,
    read_session,
    verify_password,
)
from visionops.api.errors import ServiceError
from visionops.api.runtime import ServiceSettings, VideoJob
from visionops.api.storage import VideoRecord, VideoRepository

VIDEO_TYPES = {
    ".mp4": {"video/mp4", "application/octet-stream"},
    ".mov": {"video/quicktime", "application/octet-stream"},
    ".avi": {"video/x-msvideo", "video/avi", "application/octet-stream"},
    ".mkv": {"video/x-matroska", "application/octet-stream"},
}
TEMPLATE_DIR = Path(__file__).with_name("templates")
STATIC_DIR = Path(__file__).with_name("static")


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    content = bytearray()
    while chunk := await upload.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > limit:
            raise ServiceError(413, "upload_too_large", f"Upload exceeds {limit} bytes")
    if not content:
        raise ServiceError(400, "empty_upload", "Upload is empty")
    return bytes(content)


class PortfolioService:
    def __init__(
        self, service_settings: ServiceSettings, web_settings: WebSettings
    ) -> None:
        self.service_settings = service_settings
        self.web_settings = web_settings
        self.repository = VideoRepository(web_settings.database_path)
        self.templates = Jinja2Templates(directory=TEMPLATE_DIR)

    def initialize(self) -> None:
        self.repository.initialize()

    def current_admin(self, request: Request) -> str | None:
        settings = self.web_settings
        if not settings.authentication_configured:
            return None
        session = read_session(
            request.cookies.get(SESSION_COOKIE),
            settings.session_secret or "",
            max_age_seconds=settings.session_max_age_seconds,
        )
        if session is None or session.get("username") != settings.admin_username:
            return None
        return str(session["username"])

    def require_admin(self, request: Request) -> str:
        username = self.current_admin(request)
        if username is None:
            raise ServiceError(401, "authentication_required", "Admin login required")
        return username

    async def submit_video(
        self,
        request: Request,
        file: UploadFile,
        *,
        title: str | None = None,
        zone_config: UploadFile | None = None,
    ) -> VideoJob:
        original_name = Path(file.filename or "").name
        suffix = Path(original_name).suffix.lower()
        accepted_types = VIDEO_TYPES.get(suffix)
        content_type = (file.content_type or "").lower()
        if accepted_types is None or content_type not in accepted_types:
            raise ServiceError(
                415, "unsupported_video_type", "Unsupported video extension or type"
            )
        content = await _read_limited(file, self.service_settings.max_video_bytes)
        incoming = self.service_settings.runtime_dir / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        input_path = incoming / f"{uuid.uuid4().hex}{suffix}"
        presentation_temp = incoming / f"{uuid.uuid4().hex}-presentation{suffix}"
        input_path.write_bytes(content)
        capture = cv2.VideoCapture(str(input_path))
        valid = capture.isOpened() and capture.get(cv2.CAP_PROP_FRAME_COUNT) > 0
        capture.release()
        if not valid:
            input_path.unlink(missing_ok=True)
            raise ServiceError(422, "invalid_video", "Video could not be decoded")
        shutil.copyfile(input_path, presentation_temp)

        zone_path = None
        try:
            if zone_config is not None and zone_config.filename:
                zone_content = await _read_limited(zone_config, 1024 * 1024)
                zone_path = request.app.state.runtime.validate_zone(
                    zone_content, incoming
                )
            job = request.app.state.runtime.submit_video(
                input_path, zone_path, presentation_temp
            )
        except Exception:
            input_path.unlink(missing_ok=True)
            presentation_temp.unlink(missing_ok=True)
            raise
        clean_title = (title or "").strip()[:120]
        if not clean_title:
            clean_title = Path(original_name).stem.replace("_", " ").strip()
        clean_title = clean_title or "Processed PPE video"
        self.repository.create(
            video_id=job.job_id,
            title=clean_title,
            source_filename=original_name,
            status=job.status,
            original_video_path=job.presentation_original_path,
        )
        self.sync_record(request, job.job_id)
        return job

    def sync_record(self, request: Request, video_id: str) -> VideoRecord | None:
        record = self.repository.get(video_id)
        if record is None or record.status == "failed":
            return record
        if (
            record.status == "completed"
            and record.original_video_path
            and record.annotated_video_path
        ):
            return record
        try:
            job = request.app.state.runtime.get_job(video_id)
        except ServiceError:
            return record
        incidents = job.output_dir / "artifacts/incidents.jsonl"
        self.repository.update_job(
            video_id,
            status=job.status,
            original_video_path=job.presentation_original_path
            if job.status == "completed"
            and job.presentation_original_path is not None
            and job.presentation_original_path.is_file()
            else None,
            annotated_video_path=job.presentation_annotated_path
            if job.status == "completed"
            and job.presentation_annotated_path is not None
            and job.presentation_annotated_path.is_file()
            else None,
            incidents_path=incidents
            if job.status == "completed" and incidents.is_file()
            else None,
            error_message=job.error,
        )
        return self.repository.get(video_id)

    def sync_all(self, request: Request) -> list[VideoRecord]:
        records = self.repository.list_all()
        for record in records:
            if record.status not in {"completed", "failed"}:
                self.sync_record(request, record.id)
        return self.repository.list_all()

    def _safe_annotated_path(self, record: VideoRecord) -> Path | None:
        if not record.annotated_video_path:
            return None
        path = Path(record.annotated_video_path).resolve()
        jobs_root = (self.service_settings.runtime_dir / "jobs").resolve()
        expected_parent = (jobs_root / record.id / "presentation").resolve()
        if (
            path.parent != expected_parent
            or path.name != "annotated.mp4"
            or not path.is_file()
        ):
            return None
        return path

    def _safe_original_path(self, record: VideoRecord) -> Path | None:
        if not record.original_video_path:
            return None
        path = Path(record.original_video_path).resolve()
        jobs_root = (self.service_settings.runtime_dir / "jobs").resolve()
        expected_parent = (jobs_root / record.id / "presentation").resolve()
        if (
            path.parent != expected_parent
            or path.name != "original.mp4"
            or not path.is_file()
        ):
            return None
        return path

    @staticmethod
    def _incidents(record: VideoRecord) -> list[dict[str, Any]]:
        if not record.incidents_path:
            return []
        path = Path(record.incidents_path)
        if not path.is_file():
            return []
        incidents: dict[str, dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = raw.get("incident", raw)
            incident_id = source.get("incident_id")
            if not incident_id:
                continue
            incidents[str(incident_id)] = {
                key: source[key]
                for key in (
                    "incident_id",
                    "track_id",
                    "equipment",
                    "started_at_seconds",
                    "confirmed_at_seconds",
                    "cleared_at_seconds",
                    "duration_seconds",
                    "status",
                    "close_reason",
                )
                if key in source
            }
        return sorted(
            incidents.values(), key=lambda item: item.get("started_at_seconds", 0)
        )

    def _summary(self, record: VideoRecord) -> dict[str, Any]:
        path = self._safe_annotated_path(record)
        summary: dict[str, Any] = {
            "duration_seconds": None,
            "frames_processed": None,
            "unique_person_tracks": None,
            "processed_at": record.created_at,
        }
        if path is None:
            return summary
        summary["processed_at"] = datetime.fromtimestamp(
            path.stat().st_mtime, UTC
        ).isoformat()
        metadata_path = path.parents[1] / "artifacts/run_metadata.json"
        if not metadata_path.is_file():
            return summary
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            summary.update(
                {
                    "duration_seconds": metadata.get("source", {}).get(
                        "duration_seconds"
                    ),
                    "frames_processed": metadata.get("processing", {}).get(
                        "frames_processed"
                    ),
                    "unique_person_tracks": metadata.get("tracking", {})
                    .get("diagnostics", {})
                    .get("unique_tracks_created"),
                }
            )
        except (json.JSONDecodeError, OSError):
            pass
        return summary

    def serialize(self, record: VideoRecord, *, admin: bool) -> dict[str, Any]:
        incidents = self._incidents(record)
        value: dict[str, Any] = {
            "id": record.id,
            "title": record.title,
            "created_at": record.created_at,
            "status": record.status,
            "published": record.published,
            "incident_count": len(incidents),
            "summary": self._summary(record),
            "watch_url": f"/watch/{record.id}",
            "video_url": f"/media/videos/{record.id}"
            if self._safe_annotated_path(record)
            else None,
            "original_url": f"/media/originals/{record.id}"
            if self._safe_original_path(record)
            else None,
        }
        if admin:
            value.update(
                {
                    "source_filename": record.source_filename,
                    "error_message": record.error_message,
                    "incidents": incidents,
                }
            )
        return value

    def delete_output(self, record: VideoRecord) -> None:
        jobs_root = (self.service_settings.runtime_dir / "jobs").resolve()
        target = (jobs_root / record.id).resolve()
        if target.parent != jobs_root or target.name != record.id:
            raise ServiceError(500, "unsafe_delete_target", "Invalid job directory")
        if target.is_dir():
            shutil.rmtree(target)
        self.repository.delete(record.id)


def install_portfolio_routes(application: FastAPI, service: PortfolioService) -> None:
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.get("/", response_class=HTMLResponse)
    def public_home(request: Request) -> HTMLResponse:
        service.sync_all(request)
        videos = [
            service.serialize(record, admin=False)
            for record in service.repository.list_published()
        ]
        return service.templates.TemplateResponse(
            request=request, name="home.html", context={"videos": videos}
        )

    @application.get("/watch/{video_id}", response_class=HTMLResponse)
    def watch(video_id: str, request: Request) -> HTMLResponse:
        record = service.sync_record(request, video_id)
        if record is None or not record.published or record.status != "completed":
            raise HTTPException(status_code=404, detail="Video not found")
        video = service.serialize(record, admin=False)
        video["incidents"] = service._incidents(record)
        return service.templates.TemplateResponse(
            request=request, name="watch.html", context={"video": video}
        )

    @application.get("/media/videos/{video_id}")
    def annotated_video(video_id: str, request: Request) -> FileResponse:
        record = service.sync_record(request, video_id)
        if record is None or record.status != "completed":
            raise HTTPException(status_code=404, detail="Video not found")
        if not record.published and service.current_admin(request) is None:
            raise HTTPException(status_code=404, detail="Video not found")
        path = service._safe_annotated_path(record)
        if path is None:
            raise HTTPException(status_code=404, detail="Video not found")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"{record.id}.mp4",
            content_disposition_type="inline",
        )

    @application.get("/media/originals/{video_id}")
    def original_video(video_id: str, request: Request) -> FileResponse:
        record = service.sync_record(request, video_id)
        if record is None or record.status != "completed":
            raise HTTPException(status_code=404, detail="Video not found")
        if not record.published and service.current_admin(request) is None:
            raise HTTPException(status_code=404, detail="Video not found")
        path = service._safe_original_path(record)
        if path is None:
            raise HTTPException(status_code=404, detail="Video not found")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"{record.id}-original{path.suffix.lower()}",
            content_disposition_type="inline",
        )

    @application.get("/admin/login", response_class=HTMLResponse)
    def login_page(request: Request) -> Response:
        if service.current_admin(request):
            return RedirectResponse("/admin", status_code=303)
        return service.templates.TemplateResponse(
            request=request,
            name="login.html",
            context={
                "error": None,
                "configured": service.web_settings.authentication_configured,
            },
        )

    @application.post("/admin/login", response_class=HTMLResponse)
    def login(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ) -> Response:
        settings = service.web_settings
        valid = (
            settings.authentication_configured
            and username == settings.admin_username
            and verify_password(password, settings.admin_password_hash or "")
        )
        if not valid:
            return service.templates.TemplateResponse(
                request=request,
                name="login.html",
                context={
                    "error": "Invalid username or password.",
                    "configured": settings.authentication_configured,
                },
                status_code=401,
            )
        response = RedirectResponse("/admin", status_code=303)
        response.set_cookie(
            SESSION_COOKIE,
            create_session(username, settings.session_secret or ""),
            max_age=settings.session_max_age_seconds,
            httponly=True,
            samesite="lax",
            secure=is_https(request),
            path="/",
        )
        return response

    @application.post("/admin/logout")
    def logout(request: Request) -> RedirectResponse:
        response = RedirectResponse("/", status_code=303)
        response.delete_cookie(
            SESSION_COOKIE,
            path="/",
            httponly=True,
            samesite="lax",
            secure=is_https(request),
        )
        return response

    @application.get("/admin", response_class=HTMLResponse)
    def dashboard(request: Request) -> Response:
        if service.current_admin(request) is None:
            return RedirectResponse("/admin/login", status_code=303)
        videos = [
            service.serialize(record, admin=True)
            for record in service.sync_all(request)
        ]
        return service.templates.TemplateResponse(
            request=request, name="admin.html", context={"videos": videos}
        )

    @application.post("/api/admin/videos", status_code=202)
    async def admin_upload(
        request: Request,
        _admin: str = Depends(service.require_admin),
        file: UploadFile = File(...),
        title: str | None = Form(default=None),
    ) -> JSONResponse:
        job = await service.submit_video(request, file, title=title)
        record = service.repository.get(job.job_id)
        assert record is not None
        return JSONResponse(service.serialize(record, admin=True), status_code=202)

    @application.get("/api/admin/videos")
    def admin_videos(
        request: Request, _admin: str = Depends(service.require_admin)
    ) -> list[dict[str, Any]]:
        return [
            service.serialize(record, admin=True)
            for record in service.sync_all(request)
        ]

    @application.get("/api/admin/videos/{video_id}")
    def admin_video(
        video_id: str,
        request: Request,
        _admin: str = Depends(service.require_admin),
    ) -> dict[str, Any]:
        record = service.sync_record(request, video_id)
        if record is None:
            raise ServiceError(404, "video_not_found", "Video does not exist")
        return service.serialize(record, admin=True)

    def set_publication(
        video_id: str, request: Request, published: bool
    ) -> dict[str, Any]:
        record = service.sync_record(request, video_id)
        if record is None:
            raise ServiceError(404, "video_not_found", "Video does not exist")
        if published and (
            record.status != "completed"
            or service._safe_annotated_path(record) is None
            or service._safe_original_path(record) is None
        ):
            raise ServiceError(
                409,
                "video_not_publishable",
                "Completed original and annotated videos are required",
            )
        service.repository.set_published(video_id, published)
        updated = service.repository.get(video_id)
        assert updated is not None
        return service.serialize(updated, admin=True)

    @application.post("/api/admin/videos/{video_id}/publish")
    def publish(
        video_id: str,
        request: Request,
        _admin: str = Depends(service.require_admin),
    ) -> dict[str, Any]:
        return set_publication(video_id, request, True)

    @application.post("/api/admin/videos/{video_id}/unpublish")
    def unpublish(
        video_id: str,
        request: Request,
        _admin: str = Depends(service.require_admin),
    ) -> dict[str, Any]:
        return set_publication(video_id, request, False)

    @application.delete(
        "/api/admin/videos/{video_id}", status_code=204, response_class=Response
    )
    def delete_video(
        video_id: str,
        request: Request,
        _admin: str = Depends(service.require_admin),
    ) -> Response:
        record = service.sync_record(request, video_id)
        if record is None:
            raise ServiceError(404, "video_not_found", "Video does not exist")
        if record.status in {"queued", "running"}:
            raise ServiceError(409, "job_active", "Active job cannot be deleted")
        service.delete_output(record)
        return Response(status_code=204)
