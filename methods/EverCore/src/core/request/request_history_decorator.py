# -*- coding: utf-8 -*-
"""
Request history decorator module

Provides a decorator to capture and publish request information as events.
Used for request replay functionality.

Note: This feature is controlled by RequestHistoryConfig.
- Opensource version: disabled by default
- Enterprise version: can be enabled by overriding the config

The decorator captures raw request data without parsing.
Enterprise code is responsible for processing the data.
"""

import asyncio
from functools import wraps
from typing import Any, Callable, Optional

from fastapi import Request

from core.context.context import get_current_request
from core.di import get_bean_by_type
from core.events import ApplicationEventPublisher
from core.observation.logger import get_logger
from core.request.request_history_config import is_request_history_enabled
from core.request.request_history_event import RequestHistoryEvent
from project_meta import PROJECT_VERSION

logger = get_logger(__name__)


async def _extract_request_body(request: Request) -> Optional[str]:
    """
    Extract raw request body from FastAPI request

    Args:
        request: FastAPI Request object

    Returns:
        Raw body string or None
    """
    try:
        body_bytes = await request.body()
        if body_bytes:
            return body_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        logger.debug(f"Failed to extract request body: {e}")
    return None


def _build_request_history_event(
    request: Request,
    body: Optional[str],
    endpoint_name: Optional[str] = None,
    controller_name: Optional[str] = None,
    version: str = "",
) -> RequestHistoryEvent:
    """
    Build RequestHistoryEvent from FastAPI request

    Extracts raw data without parsing - enterprise code handles processing.

    Args:
        request: FastAPI Request object
        body: Raw request body string
        endpoint_name: Name of the endpoint function
        controller_name: Name of the controller class
        version: Code version string

    Returns:
        RequestHistoryEvent: Event containing raw request information
    """
    # Get client info
    client_host = None
    client_port = None
    if request.client:
        client_host = request.client.host
        client_port = request.client.port

    return RequestHistoryEvent(
        version=version,
        endpoint_name=endpoint_name,
        controller_name=controller_name,
        method=request.method,
        url=str(request.url),
        headers=dict(request.headers),
        body=body,
        client_host=client_host,
        client_port=client_port,
    )


async def _publish_request_history_event(event: RequestHistoryEvent) -> None:
    """
    Publish request history event

    Args:
        event: RequestHistoryEvent to publish
    """
    try:
        publisher = get_bean_by_type(ApplicationEventPublisher)
        await publisher.publish(event)
        logger.debug(
            f"Published request history event: {event.method} "
            f"endpoint={event.endpoint_name} version={event.version}"
        )
    except Exception as e:
        logger.warning(f"Failed to publish request history event: {e}")


def log_request(
    include_body: bool = True, async_publish: bool = True, version: Optional[str] = None
) -> Callable:
    """
    Decorator to log request information as an event

    Captures raw HTTP request information and publishes it as a
    RequestHistoryEvent for request replay functionality.

    Note: Raw data is captured without parsing. Enterprise code
    is responsible for processing the data.

    Args:
        include_body: Whether to include request body (default: True)
        async_publish: Whether to publish event asynchronously without waiting
                       (default: True)
        version: Code version string (default: from project_meta.PROJECT_VERSION)

    Returns:
        Decorator function

    Example:
        >>> from core.request import log_request
        >>>
        >>> class UserController(BaseController):
        ...     @post("/users")
        ...     @log_request()
        ...     async def create_user(self, request: Request, data: UserCreate):
        ...         # Request info is automatically logged as event
        ...         return {"user_id": "123"}
        ...
        ...     @get("/users/{user_id}")
        ...     @log_request(include_body=False)
        ...     async def get_user(self, request: Request, user_id: str):
        ...         return {"user_id": user_id}
    """
    # Use project version if not specified
    effective_version = version if version is not None else PROJECT_VERSION

    def decorator(func: Callable) -> Callable:
        # Get function metadata for event
        endpoint_name = func.__name__
        # Try to get controller name from qualname
        qualname = getattr(func, "__qualname__", "")
        controller_name = qualname.split(".")[0] if "." in qualname else None

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if request history logging is enabled
            if not is_request_history_enabled():
                return await func(*args, **kwargs)

            # Try to get request from kwargs or args
            request: Optional[Request] = kwargs.get("request")
            if request is None:
                # Try to find Request in args
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            # If still no request, try context
            if request is None:
                request = get_current_request()

            # Build and publish event if we have a request
            if request is not None:
                try:
                    # Extract body if needed
                    body = None
                    if include_body:
                        body = await _extract_request_body(request)

                    # Build event with raw data
                    event = _build_request_history_event(
                        request=request,
                        body=body,
                        endpoint_name=endpoint_name,
                        controller_name=controller_name,
                        version=effective_version,
                    )

                    # Publish event
                    if async_publish:
                        # Fire and forget - don't wait for publish to complete
                        asyncio.create_task(_publish_request_history_event(event))
                    else:
                        await _publish_request_history_event(event)

                except Exception as e:
                    logger.warning(f"Failed to log request: {e}")

            # Call the original function
            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if request history logging is enabled
            if not is_request_history_enabled():
                return func(*args, **kwargs)

            # Try to get request from kwargs or args
            request: Optional[Request] = kwargs.get("request")
            if request is None:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if request is None:
                request = get_current_request()

            # For sync functions, we publish synchronously
            if request is not None:
                try:
                    # For sync functions, we can't await body extraction
                    # Body will be None for sync endpoints
                    body = None

                    event = _build_request_history_event(
                        request=request,
                        body=body,
                        endpoint_name=endpoint_name,
                        controller_name=controller_name,
                        version=effective_version,
                    )

                    # Use sync publish
                    try:
                        publisher = get_bean_by_type(ApplicationEventPublisher)
                        publisher.publish_sync(event)
                    except Exception as e:
                        logger.warning(f"Failed to publish request history event: {e}")

                except Exception as e:
                    logger.warning(f"Failed to log request: {e}")

            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Convenience decorator with default settings
log_request_default = log_request()
