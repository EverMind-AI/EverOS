"""While a purge runs, refuse every write route with 409; reads pass.

The routes are classified once, by method and path template (version prefix
removed). Only reads pass: a route in neither table -- a new one nobody
classified -- is refused like a write, and the classification test fails
until it is listed. ``GET`` is always a read; ``OPTIONS`` / ``HEAD`` (CORS
preflight) always pass.
"""

from __future__ import annotations

__all__ = ["READ_ROUTES", "WRITE_ROUTES", "PurgeGateMiddleware", "classify"]

import re

from starlette.requests import Request
from starlette.types import ASGIApp, Receive, Scope, Send

from everos.core.errors import ErrorCode
from everos.entrypoints.api.exception_handlers import error_response

READ_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/memory/search"),
        ("POST", "/memory/get"),
        ("POST", "/knowledge/search"),
        ("POST", "/memory/purge"),
        ("POST", "/memory/purge/confirm"),
        ("GET", "/memory/purge/{job_id}"),
        ("GET", "/health"),
        ("GET", "/metrics"),
        ("GET", "/tasks"),
        ("GET", "/tasks/{task_id}"),
        ("GET", "/knowledge/categories"),
        ("GET", "/knowledge/documents"),
        ("GET", "/knowledge/documents/{doc_id}"),
        ("GET", "/knowledge/topics/{topic_id}"),
        ("GET", "/docs"),
        ("GET", "/docs/oauth2-redirect"),
        ("GET", "/redoc"),
        ("GET", "/openapi.json"),
    }
)
"""Served during a purge."""

WRITE_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("POST", "/memory/add"),
        ("POST", "/memory/flush"),
        ("POST", "/memory/delete"),
        ("POST", "/memory/delete_by_ids"),
        ("POST", "/knowledge/documents"),
        ("PUT", "/knowledge/documents/{doc_id}"),
        ("PATCH", "/knowledge/documents/{doc_id}"),
        ("DELETE", "/knowledge/documents/{doc_id}"),
        ("POST", "/ome/trigger"),
        ("POST", "/cascade/quiesce"),
    }
)
"""Refused with 409 ``PURGE_IN_PROGRESS`` during a purge."""

_VERSION = re.compile(r"^/api/v[12](?=/)")
_ALWAYS = frozenset({"OPTIONS", "HEAD"})


def _pattern(template: str) -> re.Pattern[str]:
    return re.compile(
        "^" + re.sub(r"\\\{[^/]+\\\}", "[^/]+", re.escape(template)) + "$"
    )


_PATTERNS = [
    (method, _pattern(template), (method, template) in READ_ROUTES)
    for method, template in READ_ROUTES | WRITE_ROUTES
]


def classify(method: str, path: str) -> bool | None:
    """``True`` read, ``False`` write, ``None``: no such route."""
    path = _VERSION.sub("", path)
    for route_method, pattern, is_read in _PATTERNS:
        if route_method == method and pattern.match(path):
            return is_read
    return None


def _is_read(method: str, path: str) -> bool:
    return method == "GET" or classify(method, path) is True


class PurgeGateMiddleware:
    """409s write routes while the application's purge mark is up."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in _ALWAYS:
            await self.app(scope, receive, send)
            return
        application = getattr(scope["app"].state, "application", None)
        job_id = application.purge.active_job_id() if application else None
        if job_id is None or _is_read(scope["method"], scope["path"]):
            await self.app(scope, receive, send)
            return
        response = error_response(
            Request(scope),
            409,
            ErrorCode.PURGE_IN_PROGRESS,
            f"purge {job_id} in progress; poll GET /api/v2/memory/purge/{job_id}",
        )
        await response(scope, receive, send)
