"""
Application logic middleware
Responsible for extracting and setting application-level context information, and handling application-related logic (e.g., reporting)
"""

from typing import Callable, Dict, Any, Optional

from fastapi import Request, HTTPException
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from core.observation.logger import get_logger
from core.context.context import set_current_app_info, set_current_request
from core.di.utils import get_bean_by_type
from core.request.app_logic_provider import AppLogicProvider

logger = get_logger(__name__)


class AppLogicMiddleware(BaseHTTPMiddleware):
    """
    Application logic middleware

    Responsible for managing the request lifecycle and invoking callback methods of AppLogicProvider:
    - setup_app_context(): Extract and set application context (called on every request)
    - on_request_begin(): Called when request begins (controlled by should_process_request)
    - on_request_complete(): Called when request ends (controlled by should_process_request)
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._app_logic_provider = get_bean_by_type(AppLogicProvider)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # ========== Extract and set application context (called on every request) ==========
        app_info = self._app_logic_provider.setup_app_context(request)

        # Set context
        set_current_request(request)
        # _CachedRequest is a subclass of Request, it caches the request body in memory
        await request.body()
        if app_info:
            set_current_app_info(app_info)

        # ========== Check whether to process business logic for this request ==========
        should_process = self._app_logic_provider.should_process_request(request)
        if not should_process:
            # Skip business logic processing, directly call the next middleware
            return await call_next(request)

        response: Optional[Response] = None
        error_message: Optional[str] = None
        exception_status_code: Optional[int] = None

        try:
            # ========== Validate request: call validate_request ==========
            await self._app_logic_provider.validate_request(request)
            # If validation fails (e.g., quota exceeded), HTTPException will be raised
            # FastAPI will catch HTTPException and return appropriate response (e.g., 429)

            # ========== Request begins: call on_request_begin ==========
            await self._app_logic_provider.on_request_begin(request)

            # ========== Call next layer processing ==========
            response = await call_next(request)
            return response

        except HTTPException as e:
            logger.error("HTTPException in application logic middleware: %s", e)
            error_message = str(e.detail) if e.detail else str(e)
            exception_status_code = e.status_code
            raise

        except Exception as e:
            logger.error("Exception in application logic middleware: %s", e)
            error_message = str(e)
            raise

        finally:
            # ========== Request ends: call on_request_complete ==========
            # Determine HTTP status code
            if response is not None:
                http_code = response.status_code
            elif exception_status_code is not None:
                http_code = exception_status_code
            else:
                http_code = 500

            try:
                await self._app_logic_provider.on_request_complete(
                    request=request, http_code=http_code, error_message=error_message
                )
            except Exception as callback_error:
                logger.warning(
                    "on_request_complete execution failed: %s", callback_error
                )
