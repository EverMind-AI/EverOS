"""
Global exception handler

Provides a unified exception handling mechanism for FastAPI applications, ensuring all HTTP exceptions
(including exceptions raised by middleware) are properly handled and returned to the client.
"""

from fastapi import Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_422_UNPROCESSABLE_CONTENT,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from core.observation.logger import get_logger
from common_utils.datetime_utils import to_iso_format, get_now_with_timezone
from core.constants.errors import ErrorCode
from core.di.utils import get_bean_by_type
from core.request.app_logic_provider import AppLogicProvider
from api_specs.dtos.base import ErrorApiResponse

logger = get_logger(__name__)


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler

    Handles all exceptions uniformly, including HTTPException and other exceptions,
    ensuring they are properly formatted and returned to the client.

    Args:
        request: FastAPI request object
        exc: Exception object

    Returns:
        JSONResponse: Formatted error response
    """
    request_id = get_bean_by_type(AppLogicProvider).get_current_request_id()

    # Handle Pydantic/FastAPI validation errors (422) — convert to ErrorApiResponse format
    if isinstance(exc, RequestValidationError):
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = " -> ".join(str(p) for p in first.get("loc", []) if p != "body")
            msg = first.get("msg", "Validation error")
            message = f"{msg}: {loc}" if loc else msg
        else:
            message = "Request validation error"

        logger.warning(
            "Validation Failed: %s %s - %s", request.method, str(request.url), message
        )
        error = ErrorApiResponse(
            code=ErrorCode.HTTP_ERROR.value,
            message=message,
            request_id=request_id,
            timestamp=to_iso_format(get_now_with_timezone()),
            path=str(request.url.path),
        )
        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT, content=error.model_dump()
        )

    # Handle HTTP exceptions
    if isinstance(exc, HTTPException):
        logger.warning(
            "HTTP exception: %s %s - Status code: %d, Detail: %s",
            request.method,
            str(request.url),
            exc.status_code,
            exc.detail,
        )

        error = ErrorApiResponse(
            code=ErrorCode.HTTP_ERROR.value,
            message=exc.detail,
            request_id=request_id,
            timestamp=to_iso_format(get_now_with_timezone()),
            path=str(request.url.path),
        )
        return JSONResponse(status_code=exc.status_code, content=error.model_dump())

    # Handle other exceptions
    logger.error(
        "Unhandled exception: %s %s - Exception type: %s, Detail: %s",
        request.method,
        str(request.url),
        type(exc).__name__,
        str(exc),
        exc_info=True,
    )

    error = ErrorApiResponse(
        code=ErrorCode.SYSTEM_ERROR.value,
        message="Internal server error",
        request_id=request_id,
        timestamp=to_iso_format(get_now_with_timezone()),
        path=str(request.url.path),
    )
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR, content=error.model_dump()
    )
