"""SQLite metadata catalog for portfolio video jobs."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class VideoRecord:
    id: str
    title: str
    source_filename: str
    created_at: str
    status: str
    published: bool
    original_video_path: str | None
    annotated_video_path: str | None
    incidents_path: str | None
    error_message: str | None


class VideoRepository:
    """Small connection-per-operation repository safe for worker threads."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_filename TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'running', 'completed', 'failed')
                    ),
                    published INTEGER NOT NULL DEFAULT 0 CHECK (published IN (0, 1)),
                    original_video_path TEXT,
                    annotated_video_path TEXT,
                    incidents_path TEXT,
                    error_message TEXT
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(videos)").fetchall()
            }
            if "original_video_path" not in columns:
                connection.execute(
                    "ALTER TABLE videos ADD COLUMN original_video_path TEXT"
                )
            connection.execute(
                "UPDATE videos SET published = 0 WHERE original_video_path IS NULL"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _record(row: sqlite3.Row) -> VideoRecord:
        values = dict(row)
        values["published"] = bool(values["published"])
        return VideoRecord(**values)

    def create(
        self,
        *,
        video_id: str,
        title: str,
        source_filename: str,
        status: str,
        original_video_path: Path | None,
    ) -> VideoRecord:
        created_at = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO videos (
                    id, title, source_filename, created_at, status, published,
                    original_video_path
                ) VALUES (?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    video_id,
                    title,
                    source_filename,
                    created_at,
                    status,
                    str(original_video_path) if original_video_path else None,
                ),
            )
        record = self.get(video_id)
        assert record is not None
        return record

    def get(self, video_id: str) -> VideoRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM videos WHERE id = ?", (video_id,)
            ).fetchone()
        return self._record(row) if row is not None else None

    def list_all(self) -> list[VideoRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM videos ORDER BY created_at DESC"
            ).fetchall()
        return [self._record(row) for row in rows]

    def list_published(self) -> list[VideoRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM videos
                WHERE published = 1 AND status = 'completed'
                    AND original_video_path IS NOT NULL
                    AND annotated_video_path IS NOT NULL
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [self._record(row) for row in rows]

    def update_job(
        self,
        video_id: str,
        *,
        status: str,
        original_video_path: Path | None = None,
        annotated_video_path: Path | None = None,
        incidents_path: Path | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE videos
                SET status = ?, original_video_path = ?, annotated_video_path = ?,
                    incidents_path = ?, error_message = ?
                WHERE id = ?
                """,
                (
                    status,
                    str(original_video_path) if original_video_path else None,
                    str(annotated_video_path) if annotated_video_path else None,
                    str(incidents_path) if incidents_path else None,
                    error_message,
                    video_id,
                ),
            )

    def set_published(self, video_id: str, published: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE videos SET published = ? WHERE id = ?",
                (int(published), video_id),
            )

    def delete(self, video_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM videos WHERE id = ?", (video_id,))
