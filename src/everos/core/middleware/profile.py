"""Performance profiling middleware (HTML report via pyinstrument).

Triggered with ``?profile=true`` query parameter when ``PROFILING_ENABLED=true``
is set. Gracefully no-ops if pyinstrument is not installed.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable

from fastapi import Request
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

_TRUTHY = frozenset({"1", "true", "yes"})


def _profiling_enabled() -> bool:
    """Read ``PROFILING_ENABLED`` env var (1 / true / yes => enabled)."""
    raw = os.getenv("PROFILING_ENABLED", os.getenv("PROFILING", "false")).lower()
    return raw in _TRUTHY


class ProfileMiddleware(BaseHTTPMiddleware):
    """Returns a pyinstrument HTML report when ``?profile=true`` is set."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._enabled = _profiling_enabled()
        self._available = False
        if self._enabled:
            try:
                import pyinstrument  # noqa: F401

                self._available = True
                logger.info("profiling_middleware_enabled")
            except ImportError:
                logger.warning("profiling_requested_but_pyinstrument_missing")
                self._enabled = False

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if not self._enabled or not self._available:
            return await call_next(request)

        if request.query_params.get("profile", "").lower() not in _TRUTHY:
            return await call_next(request)

        from pyinstrument import Profiler

        profiler = Profiler()
        profiler.start()
        logger.info("profile_started", method=request.method, path=request.url.path)
        try:
            await call_next(request)
        except Exception:
            logger.exception("profile_request_failed")
        profiler.stop()
        return HTMLResponse(content=profiler.output_html(), status_code=200)
