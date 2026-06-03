"""Global exception handler — uniform error envelope per v1 API brief §1.

Envelope shape (matches the v1 API brief §1 — ``request_id`` at the top
level alongside ``error``; the ``error`` object carries ``code`` /
``message`` plus ops-friendly ``timestamp`` / ``path`` for debugging)::

    {
      "request_id": "<32 lowercase hex chars — W3C trace_id format>",
      "error": {
        "code": "HTTP_ERROR" | "SYSTEM_ERROR",
        "message": "<reason>",
        "timestamp": "<ISO 8601 with tz>",
        "path": "<request path>"
      }
    }

Rules:
- 4xx (DTO / business validation / HTTPException) → ``code="HTTP_ERROR"``
  with the human-readable reason in ``message``.
- 5xx (unhandled exception) → ``code="SYSTEM_ERROR"`` with a fixed
  ``message="Internal server error"`` — internal exception details are
  logged but never leak to the client.
- ``request_id`` is sourced from ``request.state.request_id`` (set by
  upstream middleware); falls back to a freshly minted id when absent.
"""

from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from everos.component.utils.datetime import (
    get_now_with_timezone,
    to_iso_format,
)
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import gen_request_id

logger = get_logger(__name__)

_INTERNAL_ERROR_MESSAGE = "Internal server error"


def _request_id(request: Request) -> str:
    """Return the request_id set by middleware, or mint a fresh fallback."""
    rid = getattr(request.state, "request_id", None)
    if rid:
        return str(rid)
    return gen_request_id()


def _envelope(
    *,
    code: str,
    message: str,
    request: Request,
) -> dict[str, object]:
    """Build the canonical error envelope (wiki §1 shape — nested ``error``).

    ``request_id`` at the top level, ``error`` object carries the
    contract fields (``code`` / ``message``) plus ops-friendly
    ``timestamp`` / ``path``.
    """
    return {
        "request_id": _request_id(request),
        "error": {
            "code": code,
            "message": message,
            "timestamp": to_iso_format(get_now_with_timezone()),
            "path": str(request.url.path),
        },
    }


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Convert any exception into a uniform JSON error response."""
    path = str(request.url.path)
    method = request.method

    if isinstance(exc, RequestValidationError):
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
            msg = first.get("msg", "Validation error")
            message = f"{msg}: {loc}" if loc else msg
        else:
            message = "Request validation error"

        logger.warning("validation_error", method=method, path=path, message=message)
        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            content=_envelope(code="HTTP_ERROR", message=message, request=request),
        )

    if isinstance(exc, HTTPException):
        logger.warning(
            "http_exception",
            method=method,
            path=path,
            status_code=exc.status_code,
            detail=exc.detail,
        )
        # 5xx routed through HTTPException is rare but valid; still honour
        # the SYSTEM_ERROR code so the envelope is consistent.
        if exc.status_code >= 500:
            return JSONResponse(
                status_code=exc.status_code,
                content=_envelope(
                    code="SYSTEM_ERROR",
                    message=_INTERNAL_ERROR_MESSAGE,
                    request=request,
                ),
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(
                code="HTTP_ERROR",
                message=str(exc.detail),
                request=request,
            ),
        )

    logger.error(
        "unhandled_exception",
        method=method,
        path=path,
        exception_type=type(exc).__name__,
        exc_info=True,
    )
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content=_envelope(
            code="SYSTEM_ERROR",
            message=_INTERNAL_ERROR_MESSAGE,
            request=request,
        ),
    )
