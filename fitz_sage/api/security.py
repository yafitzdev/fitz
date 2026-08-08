"""REST API access control and server-local source path boundaries."""

from __future__ import annotations

import hmac
import ipaddress
import os
from pathlib import Path
from typing import Any

from starlette.responses import JSONResponse

API_KEY_ENV = "FITZ_API_KEY"
SOURCE_ROOTS_ENV = "FITZ_API_SOURCE_ROOTS"
ALLOWED_ORIGINS_ENV = "FITZ_API_ALLOWED_ORIGINS"
API_KEY_HEADER = b"x-fitz-api-key"


class LocalOrApiKeyMiddleware:
    """Allow loopback clients, or require the configured API key remotely."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or _is_loopback(scope.get("client")):
            await self.app(scope, receive, send)
            return

        configured_key = os.getenv(API_KEY_ENV)
        if not configured_key:
            response = JSONResponse(
                {"detail": f"Remote API access requires {API_KEY_ENV}."},
                status_code=403,
            )
            await response(scope, receive, send)
            return

        supplied_key = _header(scope, API_KEY_HEADER)
        if not supplied_key or not hmac.compare_digest(supplied_key, configured_key):
            response = JSONResponse({"detail": "Invalid or missing API key."}, status_code=401)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


def resolve_api_source(source: str | Path) -> Path:
    """Resolve a source and require it to stay within an allowed API root."""
    try:
        resolved = Path(source).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Invalid source path: {source}") from exc

    roots = _allowed_source_roots()
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise ValueError(f"Source path is outside the allowed API roots: {allowed}")
    return resolved


def allowed_api_origins() -> list[str]:
    """Return explicitly configured browser origins; default to no CORS access."""
    raw = os.getenv(ALLOWED_ORIGINS_ENV, "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _allowed_source_roots() -> tuple[Path, ...]:
    raw = os.getenv(SOURCE_ROOTS_ENV)
    candidates = raw.split(os.pathsep) if raw else [str(Path.cwd())]
    roots: list[Path] = []
    for candidate in candidates:
        if not candidate.strip():
            continue
        try:
            roots.append(Path(candidate).resolve(strict=True))
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Invalid API source root: {candidate}") from exc
    if not roots:
        raise ValueError(f"{SOURCE_ROOTS_ENV} does not contain a valid source root")
    return tuple(roots)


def _is_loopback(client: Any) -> bool:
    if not client:
        return False
    try:
        return ipaddress.ip_address(client[0]).is_loopback
    except ValueError:
        return False


def _header(scope: dict, name: bytes) -> str | None:
    for key, value in scope.get("headers", ()):
        if key.lower() == name:
            return str(value.decode("utf-8", errors="ignore"))
    return None


__all__ = [
    "API_KEY_ENV",
    "API_KEY_HEADER",
    "ALLOWED_ORIGINS_ENV",
    "SOURCE_ROOTS_ENV",
    "LocalOrApiKeyMiddleware",
    "allowed_api_origins",
    "resolve_api_source",
]
