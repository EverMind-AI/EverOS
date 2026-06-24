"""Per-type exception handlers for the EverOS FastAPI application.

Each handler converts a specific exception class (or hierarchy root) into
the canonical error envelope::

    {
      "request_id": "<32 lowercase hex chars>",
      "error": {
        "code": "<ErrorCode>",
        "message": "<reason>",
        "timestamp": "<ISO 8601 with tz>",
        "path": "<request path>"
      }
    }

Register all handlers at once with ``register_handlers(app)``.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from everos.component.utils.datetime import get_now_with_timezone, to_iso_format
from everos.core.errors import (
    CapabilityError,
    ConfigurationError,
    ConflictError,
    ErrorCode,
    ExtractionEmptyError,
    InfrastructureError,
    InvalidInputError,
    NotFoundError,
    PathTraversalError,
    UnsupportedModalityError,
)
from everos.core.observability.logging import get_logger

from .utils import extract_request_id

logger = get_logger(__name__)

_INTERNAL_ERROR_MESSAGE = "Internal server error"


# ---------------------------------------------------------------------------
# Response model (visible in OpenAPI docs)
# ---------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    """Inner ``error`` object in the canonical error envelope."""

    code: ErrorCode
    message: str
    timestamp: str
    path: str


class ErrorResponse(BaseModel):
    """Canonical error envelope returned by all error handlers."""

    request_id: str
    error: ErrorDetail


# ---------------------------------------------------------------------------
# Envelope builder
# ---------------------------------------------------------------------------


def _error_response(
    request: Request,
    status_code: int,
    code: ErrorCode,
    message: str,
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
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def not_found_handler(
    request: Request,
    exc: NotFoundError,
) -> JSONResponse:
    """NotFoundError (and subclasses) -> 404."""
    return _error_response(
        request,
        HTTP_404_NOT_FOUND,
        ErrorCode.NOT_FOUND,
        str(exc),
    )


async def conflict_handler(
    request: Request,
    exc: ConflictError,
) -> JSONResponse:
    """ConflictError (and subclasses) -> 409."""
    return _error_response(
        request,
        HTTP_409_CONFLICT,
        ErrorCode.CONFLICT,
        str(exc),
    )


async def extraction_empty_handler(
    request: Request,
    exc: ExtractionEmptyError,
) -> JSONResponse:
    """ExtractionEmptyError -> 422 with dedicated code."""
    return _error_response(
        request,
        HTTP_422_UNPROCESSABLE_CONTENT,
        ErrorCode.EXTRACTION_EMPTY,
        str(exc),
    )


async def invalid_input_handler(
    request: Request,
    exc: InvalidInputError,
) -> JSONResponse:
    """InvalidInputError (and subclasses) -> 422."""
    return _error_response(
        request,
        HTTP_422_UNPROCESSABLE_CONTENT,
        ErrorCode.INVALID_INPUT,
        str(exc),
    )


async def path_traversal_handler(
    request: Request,
    exc: PathTraversalError,
) -> JSONResponse:
    """PathTraversalError -> 400."""
    return _error_response(
        request,
        HTTP_400_BAD_REQUEST,
        ErrorCode.BAD_REQUEST,
        "Invalid input: path contains illegal characters.",
    )


async def unsupported_modality_handler(
    request: Request,
    exc: UnsupportedModalityError,
) -> JSONResponse:
    """UnsupportedModalityError -> 415."""
    return _error_response(
        request,
        HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        ErrorCode.UNSUPPORTED_FORMAT,
        str(exc),
    )


async def infrastructure_handler(
    request: Request,
    exc: InfrastructureError,
) -> JSONResponse:
    """InfrastructureError (and subclasses) -> 503."""
    logger.warning(
        "infrastructure_error",
        path=str(request.url.path),
        exception_type=type(exc).__name__,
        message=str(exc),
    )
    return _error_response(
        request,
        HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCode.EXTERNAL_SERVICE_UNAVAILABLE,
        str(exc),
    )


async def capability_handler(
    request: Request,
    exc: CapabilityError,
) -> JSONResponse:
    """CapabilityError (and subclasses) -> 503 (not retryable)."""
    logger.warning(
        "capability_error",
        path=str(request.url.path),
        exception_type=type(exc).__name__,
        message=str(exc),
    )
    return _error_response(
        request,
        HTTP_503_SERVICE_UNAVAILABLE,
        ErrorCode.CAPABILITY_UNAVAILABLE,
        str(exc),
    )


async def configuration_handler(
    request: Request,
    exc: ConfigurationError,
) -> JSONResponse:
    """ConfigurationError -> 500."""
    logger.error(
        "configuration_error",
        path=str(request.url.path),
        message=str(exc),
    )
    return _error_response(
        request,
        HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorCode.CONFIGURATION_ERROR,
        str(exc),
    )


# ---------------------------------------------------------------------------
# Pydantic / FastAPI built-in exceptions
# ---------------------------------------------------------------------------

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


async def request_validation_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """FastAPI RequestValidationError -> 422."""
    errors = exc.errors()
    if errors:
        first = errors[0]
        loc_parts = [str(p) for p in first.get("loc", []) if p != "body"]
        field = loc_parts[-1] if loc_parts else ""
        if field in _FIELD_HINTS:
            message = _FIELD_HINTS[field]
        else:
            loc = ".".join(loc_parts)
            msg = first.get("msg", "Validation error")
            message = f"{msg}: {loc}" if loc else msg
    else:
        message = "Request validation error"
    return _error_response(
        request,
        HTTP_422_UNPROCESSABLE_CONTENT,
        ErrorCode.INVALID_INPUT,
        message,
    )


async def http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    """FastAPI HTTPException -> envelope with original status code."""
    if exc.status_code >= 500:
        logger.error(
            "http_exception_5xx",
            path=str(request.url.path),
            status_code=exc.status_code,
        )
        return _error_response(
            request,
            exc.status_code,
            ErrorCode.INTERNAL_ERROR,
            _INTERNAL_ERROR_MESSAGE,
        )
    return _error_response(
        request,
        exc.status_code,
        ErrorCode.BAD_REQUEST,
        str(exc.detail),
    )


async def unexpected_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Catch-all for unhandled exceptions -> 500 (no detail leak)."""
    logger.error(
        "unhandled_exception",
        path=str(request.url.path),
        exception_type=type(exc).__name__,
        exc_info=True,
    )
    return _error_response(
        request,
        HTTP_500_INTERNAL_SERVER_ERROR,
        ErrorCode.INTERNAL_ERROR,
        _INTERNAL_ERROR_MESSAGE,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_handlers(app: FastAPI) -> None:
    """Register all per-type exception handlers on ``app``.

    Starlette walks the exception MRO and picks the first matching
    handler, so more-specific types are registered before their parents.
    """
    # Domain errors (specific before parent)
    app.add_exception_handler(PathTraversalError, path_traversal_handler)
    app.add_exception_handler(UnsupportedModalityError, unsupported_modality_handler)
    app.add_exception_handler(NotFoundError, not_found_handler)
    app.add_exception_handler(ConflictError, conflict_handler)
    app.add_exception_handler(ExtractionEmptyError, extraction_empty_handler)
    app.add_exception_handler(InvalidInputError, invalid_input_handler)
    # Infrastructure errors (transient, retryable)
    app.add_exception_handler(InfrastructureError, infrastructure_handler)
    # Capability errors (permanent, not retryable)
    app.add_exception_handler(CapabilityError, capability_handler)
    # Configuration errors
    app.add_exception_handler(ConfigurationError, configuration_handler)
    # FastAPI built-in exceptions
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(
        RequestValidationError,
        request_validation_handler,
    )
    # Catch-all
    app.add_exception_handler(Exception, unexpected_handler)
