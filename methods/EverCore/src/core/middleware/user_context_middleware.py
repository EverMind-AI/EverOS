from core.authorize.enums import Role
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Callable

from core.context.context import set_current_user_info, clear_current_user_context
from core.component.auth_provider import AuthProvider
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

logger = get_logger(__name__)


class UserContextMiddleware(BaseHTTPMiddleware):
    """
    User context middleware

    Extract user information from each HTTP request and set it into the context variable,
    so that user information can be accessed via context throughout the entire request processing,
    without explicitly passing the request parameter.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.auth_provider = get_bean_by_type(AuthProvider)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Set user context for each request

        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler

        Returns:
            Response: Response object
        """
        # Clear any existing user context
        clear_current_user_context()

        # Set user context token
        token = None

        # Step 1: Try to get and set user context
        try:
            # Attempt to get full user data from the request
            # This method now:
            # 1. No authentication data -> returns anonymous user info
            # 2. Authentication failure -> raises HTTPException(401)
            user_data = await self.auth_provider.get_optional_user_data_from_request(
                request
            )

            if user_data is not None:
                # Set user context (including anonymous users)
                token = set_current_user_info(user_data)
                if user_data.get("role") == Role.ANONYMOUS.value:
                    logger.debug("Anonymous user context set")
                else:
                    logger.debug(
                        "User context set: User ID=%s, Role=%s",
                        user_data.get("user_id"),
                        user_data.get("role"),
                    )
            else:
                user_data = {"user_id": None, "role": Role.ANONYMOUS.value}
                token = set_current_user_info(user_data)
                logger.debug("No user data obtained, set anonymous user context")

        except HTTPException as e:
            # If it's a 401 authentication failure, re-raise it directly, don't swallow
            if e.status_code == 401:
                logger.debug(
                    "Authentication failed, returning 401 error directly: %s", e.detail
                )
                raise e
            else:
                logger.error(
                    "HTTP exception occurred while setting user context: %s - %s",
                    e.status_code,
                    e.detail,
                )
                # Other HTTP exceptions do not affect request processing continuation
        except Exception as e:
            logger.error("Exception occurred while setting user context: %s", str(e))
            # User context setup failure does not affect request processing continuation
            # Specific authentication checks are handled by individual endpoints

        # Step 2: Execute business logic
        try:
            response = await call_next(request)
            return response

        except Exception as e:
            logger.error("Business logic processing exception: %s", str(e))
            # Re-raise business logic exceptions for upper layers to handle
            raise

        finally:
            # Clean up user context
            if token is not None:
                try:
                    clear_current_user_context(token)
                    logger.debug("User context cleaned up")
                except Exception as reset_error:
                    logger.warning(
                        "Error occurred while cleaning up user context: %s",
                        str(reset_error),
                    )
