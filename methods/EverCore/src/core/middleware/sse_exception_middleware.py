"""
SSE (Server-Sent Events) exception handling middleware

Provides a decorator to convert HTTP exceptions and other exceptions into SSE event format,
ensuring that exceptions in streaming responses are returned to the client in a standard format.

This middleware belongs to the infrastructure layer and handles technical details related to the HTTP protocol.
"""

import json
import logging
from typing import Any, AsyncGenerator, Callable
from functools import wraps

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def yield_sse_data(data: Any) -> str:
    """
    Format data into SSE format

    Args:
        data: Data to be sent

    Returns:
        str: Data string in SSE format
    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_exception_handler(
    func: Callable[..., AsyncGenerator[str, None]]
) -> Callable[..., AsyncGenerator[str, None]]:
    """
    SSE stream exception handling decorator

    Converts HTTPException and other exceptions into SSE event format, ensuring the client can
    handle errors in streaming responses in a consistent manner.

    Exception conversion rules:
    - HTTPException -> {"type": "error", "data": {"code": status_code, "message": detail}}
    - Other exceptions -> {"type": "error", "data": {"code": 500, "message": "Internal server error: {error}"}}

    Usage:
        @sse_exception_handler
        async def my_sse_generator() -> AsyncGenerator[str, None]:
            # Generate SSE events
            yield yield_sse_data({"type": "message", "content": "hello"})

    Args:
        func: Asynchronous generator function returning AsyncGenerator[str, None]

    Returns:
        Decorated asynchronous generator function
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> AsyncGenerator[str, None]:
        try:
            async for event in func(*args, **kwargs):
                yield event
        except HTTPException as e:
            # Convert HTTPException into SSE error event
            error_data = {
                "type": "error",
                "data": {"code": e.status_code, "message": e.detail},
            }
            logger.error(
                f"HTTP exception occurred in SSE stream: {e.status_code} - {e.detail}"
            )
            yield yield_sse_data(error_data)
        except Exception as e:
            # Convert other exceptions into SSE error event
            error_data = {
                "type": "error",
                "data": {"code": 500, "message": f"Internal server error: {str(e)}"},
            }
            logger.error(
                f"Unknown exception occurred in SSE stream: {e}", exc_info=True
            )
            yield yield_sse_data(error_data)

    return wrapper
