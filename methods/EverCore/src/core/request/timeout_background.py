"""
Timeout to background execution decorator

Used for endpoints. When business logic execution times out, automatically switches to background execution and returns a 202 response.
Works with AppLogicMiddleware.

Depends on AppLogicProvider:
- Uses get_current_request_id() to get request_id
- Uses on_request_complete() as the background completion callback

Background mode configuration:
- Background mode is enabled by default (automatically switches to background on timeout)
- Can disable background mode by passing sync_mode=true in request params (synchronously wait for execution to complete)
"""

from typing import Any, Callable, Coroutine, TypeVar, ParamSpec, Union, Optional
from functools import wraps
import asyncio
import traceback

from fastapi import Request
from fastapi.responses import JSONResponse

from core.observation.logger import get_logger
from core.di.utils import get_bean_by_type
from core.context.context import get_current_request
from core.request.app_logic_provider import AppLogicProvider

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

# Default blocking wait timeout (seconds)
DEFAULT_BLOCKING_TIMEOUT = 5.0

# Sync mode parameter name (used to disable background mode)
SYNC_MODE_PARAM = "sync_mode"


def is_background_mode_enabled(request: Optional[Request] = None) -> bool:
    """
    Check if background mode is enabled

    Background mode is enabled by default. Can be disabled via:
    - Passing sync_mode=true in request params

    Args:
        request: FastAPI request object, if None, get from context

    Returns:
        bool: True means background mode is enabled, False means disabled (synchronous execution)
    """
    if request is None:
        request = get_current_request()

    if request is None:
        # No request context, enable background mode by default
        return True

    # Check if sync_mode=true is in query params
    sync_mode = request.query_params.get(SYNC_MODE_PARAM, "true").lower()
    if sync_mode in ("false", "0", "no"):
        return True

    # Enable background mode by default
    if sync_mode in ("true", "1", "yes"):
        return False

    return True


def timeout_to_background(
    timeout: float = DEFAULT_BLOCKING_TIMEOUT,
    accepted_message: str = "Request accepted, processing in background",
) -> Callable[
    [Callable[P, Coroutine[Any, Any, T]]],
    Callable[P, Coroutine[Any, Any, Union[T, JSONResponse]]],
]:
    """
    Timeout to background execution decorator

    When the decorated endpoint execution exceeds the specified time:
    1. Return 202 Accepted response to client
    2. Business logic continues executing in the background
    3. Call AppLogicProvider.on_request_complete() when background execution completes/fails

    Works with AppLogicMiddleware:
    - Normal completion (no timeout): handled by middleware's on_request_complete
    - Timeout and switch to background (return 202): decorator calls on_request_complete, middleware skips

    Background mode configuration:
    - Background mode is enabled by default (automatically switches to background on timeout)
    - Can disable background mode by passing sync_mode=true in request params (synchronously wait for completion)

    Usage example:
    ```python
    @router.post("/memorize")
    @timeout_to_background(timeout=5.0)
    async def memorize(request: MemorizeRequest):
        # Business logic...
        return {"status": "ok"}

    # Client can disable background mode via query params:
    # POST /memorize?sync_mode=true
    ```

    Args:
        timeout: Timeout for blocking wait (seconds), default 5s
        accepted_message: Message content for 202 response

    Returns:
        Decorator function
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, T]]
    ) -> Callable[P, Coroutine[Any, Any, Union[T, JSONResponse]]]:

        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> Union[T, JSONResponse]:
            # Get AppLogicProvider instance
            provider = get_bean_by_type(AppLogicProvider)
            request_id = provider.get_current_request_id()
            task_name = f"{func.__name__}_{request_id}"

            # Check if background mode is enabled
            background_enabled = is_background_mode_enabled()

            if not background_enabled:
                # Sync mode: execute directly without timeout mechanism
                logger.debug(
                    "[TimeoutBackground] Task '%s' executing in sync mode", task_name
                )
                return await func(*args, **kwargs)

            # Background mode: create task and set timeout
            task = asyncio.create_task(func(*args, **kwargs))

            try:
                # First block and wait for specified time
                result = await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
                logger.debug(
                    "[TimeoutBackground] Task '%s' completed within %ss",
                    task_name,
                    timeout,
                )
                # Normal completion, do not call on_request_complete, let middleware handle
                return result

            except asyncio.TimeoutError:
                # Timeout not completed, switch to background execution
                logger.info(
                    "[TimeoutBackground] Task '%s' timed out (%ss), switching to background execution",
                    task_name,
                    timeout,
                )

                # Create background task to continue execution
                asyncio.create_task(_run_background_task(task, task_name, provider))

                # Return 202 Accepted
                return JSONResponse(
                    status_code=202,
                    content={"message": accepted_message, "request_id": request_id},
                )

        return wrapper

    return decorator


async def _run_background_task(
    task: asyncio.Task,
    task_name: str,
    provider: Any,  # AppLogicProvider, using Any to avoid circular import
) -> None:
    """
    Background task executor

    Args:
        task: The asyncio.Task to wait for
        task_name: Task name (for logging)
        provider: AppLogicProvider instance
    """
    try:
        await task
        logger.info("[TimeoutBackground] Background task '%s' completed", task_name)
        # Call provider's on_request_complete
        await _call_on_request_complete(provider, http_code=200, error_message=None)
    except asyncio.CancelledError:
        logger.warning(
            "[TimeoutBackground] Background task '%s' was cancelled", task_name
        )
    except Exception as e:
        logger.error(
            "[TimeoutBackground] Background task '%s' execution failed: %s",
            task_name,
            e,
        )
        traceback.print_exc()
        await _call_on_request_complete(provider, http_code=500, error_message=str(e))


async def _call_on_request_complete(
    provider: Any, http_code: int, error_message: Optional[str]
) -> None:
    """
    Call provider's on_request_complete

    Get request from context.
    """
    try:
        request = provider.get_current_request()

        if request is None:
            logger.warning(
                "[TimeoutBackground] Unable to get request, skipping on_request_complete"
            )
            return

        await provider.on_request_complete(
            request=request, http_code=http_code, error_message=error_message
        )
    except Exception as e:
        logger.warning(
            "[TimeoutBackground] on_request_complete callback execution failed: %s", e
        )
