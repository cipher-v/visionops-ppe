"""Single-administrator password and signed-session helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Request

SESSION_COOKIE = "visionops_admin"
PASSWORD_ALGORITHM = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 600_000


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def hash_password(
    password: str, *, salt: bytes | None = None, iterations: int = PASSWORD_ITERATIONS
) -> str:
    """Return a portable PBKDF2-SHA256 password hash for ADMIN_PASSWORD_HASH."""
    if not password:
        raise ValueError("Password must not be empty")
    if iterations < 100_000:
        raise ValueError("PBKDF2 iteration count is too low")
    effective_salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), effective_salt, iterations
    )
    return "$".join(
        (PASSWORD_ALGORITHM, str(iterations), _encode(effective_salt), _encode(digest))
    )


def verify_password(password: str, encoded_hash: str) -> bool:
    """Verify a password without leaking malformed-hash details."""
    try:
        algorithm, raw_iterations, raw_salt, raw_digest = encoded_hash.split("$", 3)
        if algorithm != PASSWORD_ALGORITHM:
            return False
        iterations = int(raw_iterations)
        if iterations < 100_000:
            return False
        salt = _decode(raw_salt)
        expected = _decode(raw_digest)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


@dataclass(frozen=True)
class WebSettings:
    """Deployment metadata and single-administrator configuration."""

    database_path: Path
    admin_username: str | None = None
    admin_password_hash: str | None = None
    session_secret: str | None = None
    session_max_age_seconds: int = 12 * 60 * 60

    @classmethod
    def from_environment(cls, runtime_dir: Path) -> WebSettings:
        values = (
            os.getenv("ADMIN_USERNAME"),
            os.getenv("ADMIN_PASSWORD_HASH"),
            os.getenv("SESSION_SECRET"),
        )
        supplied = [bool(value) for value in values]
        if any(supplied) and not all(supplied):
            raise ValueError(
                "ADMIN_USERNAME, ADMIN_PASSWORD_HASH, and SESSION_SECRET "
                "must all be set"
            )
        return cls(
            database_path=runtime_dir / "portfolio.sqlite3",
            admin_username=values[0],
            admin_password_hash=values[1],
            session_secret=values[2],
        )

    @property
    def authentication_configured(self) -> bool:
        return all((self.admin_username, self.admin_password_hash, self.session_secret))


def _sign(payload: str, secret: str) -> str:
    return _encode(
        hmac.new(secret.encode("utf-8"), payload.encode(), hashlib.sha256).digest()
    )


def create_session(username: str, secret: str, now: int | None = None) -> str:
    payload = _encode(
        json.dumps(
            {"username": username, "issued_at": now or int(time.time())},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return f"{payload}.{_sign(payload, secret)}"


def read_session(
    value: str | None,
    secret: str,
    *,
    max_age_seconds: int,
    now: int | None = None,
) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        payload, signature = value.split(".", 1)
        if not hmac.compare_digest(signature, _sign(payload, secret)):
            return None
        data = json.loads(_decode(payload))
        issued_at = int(data["issued_at"])
        current = now or int(time.time())
        if issued_at > current + 60 or current - issued_at > max_age_seconds:
            return None
        return data
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def is_https(request: Request) -> bool:
    forwarded = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip()
    return request.url.scheme == "https" or forwarded == "https"
