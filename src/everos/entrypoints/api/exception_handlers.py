"""Exception handlers: every error leaves as the canonical error envelope.

The envelope shape is :class:`~everos.entrypoints.api.schemas.ErrorResponse`.
Application errors are mapped by one table, ``_ERROR_RULES``: exception
class -> status code, ``ErrorCode``, optional fixed message and optional log
line. Starlette dispatches on the exception's MRO, so a subclass with its own
row overrides its parent's; mapping a new exception type is one new row.
``LLMNotConfiguredError`` is a ``RuntimeError``, not an ``AppError``, and has
a row in the same table all the same.
FastAPI's own exceptions and the catch-all keep dedicated handlers, because
their message is derived rather than ``str(exc)``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from everos.core import errors
from everos.core.errors import ErrorCode, ProviderNotConfiguredError
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import get_now_with_timezone, to_iso_format

from .dependencies import extract_request_id
from .schemas import ErrorDetail, ErrorResponse

logger = get_logger(__name__)

_INTERNAL_ERROR_MESSAGE = "Internal server error"

type _Handler = Callable[[Request, Exception], Awaitable[JSONResponse]]


def _error_response(
    request: Request, status_code: int, code: ErrorCode, message: str
) -> JSONResponse:
    """Build a JSONResponse with the canonical error envelope."""
    body = ErrorResponse(
        request_id=extract_request_id(request),
        error=ErrorDetail(
            code=code,
            message=message,
            timestamp=to_iso_format(get_now_with_timezone()),
            path=str(request.url.path),
        ),
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _exception_fields(exc: Exception) -> Mapping[str, object]:
    return {"exception_type": type(exc).__name__, "message": str(exc)}


def _provider_fields(exc: Exception) -> Mapping[str, object]:
    provider_exc = cast(ProviderNotConfiguredError, exc)
    return {"provider": provider_exc.provider, "feature": provider_exc.feature}


@dataclass(frozen=True, slots=True)
class _Rule:
    """How one exception class becomes an error response."""

    status_code: int
    code: ErrorCode
    message: str | None = None
    """Fixed client-facing message; ``None`` sends ``str(exc)``."""
    log: tuple[Literal["warning", "error"], str] | None = None
    """``(level, event)`` of the log line, or ``None`` for no log line."""
    log_fields: Callable[[Exception], Mapping[str, object]] = _exception_fields


_ERROR_RULES: Mapping[type[Exception], _Rule] = {
    errors.PathTraversalError: _Rule(
        400,
        ErrorCode.BAD_REQUEST,
        message="Invalid input: path contains illegal characters.",
    ),
    errors.UnsupportedModalityError: _Rule(415, ErrorCode.UNSUPPORTED_FORMAT),
    # 413 / 429: the batch was not accepted; split it, go sync, or retry later.
    errors.PayloadTooLargeError: _Rule(413, ErrorCode.PAYLOAD_TOO_LARGE),
    errors.QueueSaturatedError: _Rule(429, ErrorCode.QUEUE_SATURATED),
    errors.NotFoundError: _Rule(404, ErrorCode.NOT_FOUND),
    errors.ConflictError: _Rule(409, ErrorCode.CONFLICT),
    errors.ExtractionEmptyError: _Rule(422, ErrorCode.EXTRACTION_EMPTY),
    errors.InvalidInputError: _Rule(422, ErrorCode.INVALID_INPUT),
    # Transient, retryable.
    errors.InfrastructureError: _Rule(
        503,
        ErrorCode.EXTERNAL_SERVICE_UNAVAILABLE,
        log=("warning", "infrastructure_error"),
    ),
    # Permanent, not retryable.
    errors.CapabilityError: _Rule(
        503, ErrorCode.CAPABILITY_UNAVAILABLE, log=("warning", "capability_error")
    ),
    # A missing provider is fixed by the caller editing everos.toml, so it is a
    # 422; any other ConfigurationError is an operator bug, a 500.
    errors.ProviderNotConfiguredError: _Rule(
        422,
        ErrorCode.PROVIDER_NOT_CONFIGURED,
        log=("warning", "provider_not_configured_error"),
        log_fields=_provider_fields,
    ),
    errors.ConfigurationError: _Rule(
        500, ErrorCode.CONFIGURATION_ERROR, log=("error", "configuration_error")
    ),
    # Same meaning as ProviderNotConfiguredError: an LLM section in everos.toml
    # is incomplete; the message says what to edit. The app is not built
    # without [llm], so in a request this is [multimodal] (parsing an upload).
    errors.LLMNotConfiguredError: _Rule(
        422, ErrorCode.PROVIDER_NOT_CONFIGURED, log=("warning", "llm_not_configured")
    ),
}


def _rule_handler(rule: _Rule) -> _Handler:
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        if rule.log is not None:
            level, event = rule.log
            fields = rule.log_fields(exc)
            getattr(logger, level)(event, path=str(request.url.path), **fields)
        message = str(exc) if rule.message is None else rule.message
        return _error_response(request, rule.status_code, rule.code, message)

    return handle


_FIELD_HINTS: dict[str, str] = {
    "doc_id": (
        "Invalid doc_id format. Use GET /documents to look up valid doc_id values."
    ),
    "topic_id": (
        "Invalid topic_id format. "
        "Use GET /documents/{doc_id} to look up valid topic_id values."
    ),
    "query": "Search query cannot be empty.",
    "title": "Invalid title. Must contain at least one letter or digit.",
}


def _validation_message(exc: RequestValidationError) -> str:
    """First field error, as a hint for known fields or ``"<msg>: <loc>"``."""
    field_errors = exc.errors()
    if not field_errors:
        return "Request validation error"
    first = field_errors[0]
    loc_parts = [str(p) for p in first.get("loc", []) if p != "body"]
    field = loc_parts[-1] if loc_parts else ""
    if field in _FIELD_HINTS:
        return _FIELD_HINTS[field]
    loc = ".".join(loc_parts)
    msg = first.get("msg", "Validation error")
    return f"{msg}: {loc}" if loc else msg


async def request_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """FastAPI RequestValidationError -> 422."""
    message = _validation_message(exc)
    return _error_response(request, 422, ErrorCode.INVALID_INPUT, message)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """FastAPI HTTPException -> its own status code; a 5xx hides ``detail``."""
    if exc.status_code < 500:
        return _error_response(
            request, exc.status_code, ErrorCode.BAD_REQUEST, str(exc.detail)
        )
    path = str(request.url.path)
    logger.error("http_exception_5xx", path=path, status_code=exc.status_code)
    return _error_response(
        request, exc.status_code, ErrorCode.INTERNAL_ERROR, _INTERNAL_ERROR_MESSAGE
    )


async def unexpected_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions -> 500 (no detail leak)."""
    logger.error(
        "unhandled_exception",
        path=str(request.url.path),
        exception_type=type(exc).__name__,
        exc_info=True,
    )
    return _error_response(
        request, 500, ErrorCode.INTERNAL_ERROR, _INTERNAL_ERROR_MESSAGE
    )


def register_handlers(app: FastAPI) -> None:
    """Register every exception handler on ``app``.

    Starlette resolves a handler by walking the raised exception's MRO, so
    registration order does not decide which row wins: the most specific
    registered class does.
    """
    for exc_type, rule in _ERROR_RULES.items():
        app.add_exception_handler(exc_type, _rule_handler(rule))
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_handler)
    app.add_exception_handler(Exception, unexpected_handler)
