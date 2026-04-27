from fastapi import Request, Response
from starlette.responses import StreamingResponse
from starlette.middleware.base import _StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from typing import Callable, AsyncGenerator
from sqlmodel.ext.asyncio.session import AsyncSession

from core.context.context import (
    set_current_session,
    clear_current_session,
    get_current_session,
)
from core.component.database_session_provider import DatabaseSessionProvider
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

logger = get_logger(__name__)


class DatabaseSessionMiddleware(BaseHTTPMiddleware):
    """
    Simplified database session middleware

    Provides a database session for each HTTP request and intelligently handles cleanup:
    - Checks session state, automatically rolls back on error
    - Automatically rolls back on request failure
    - Automatically commits if there are uncommitted changes on successful request
    - Gives the application maximum freedom in transaction control
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.db_provider = get_bean_by_type(DatabaseSessionProvider)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Provide a database session for each request and intelligently handle transactions

        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler

        Returns:
            Response: Response object
        """
        # Create a new database session
        session = self.db_provider.create_session()
        token = set_current_session(session)
        response = None
        is_streaming = False

        try:
            # Execute request handling
            response = await call_next(request)

            # Check if it's a streaming response, which requires special handling
            if isinstance(response, StreamingResponse) or isinstance(
                response, _StreamingResponse
            ):
                is_streaming = True
                # Streaming response: wrap the generator to extend session lifetime
                wrapped_generator = self._wrap_streaming_generator(
                    response.body_iterator, session
                )
                return StreamingResponse(
                    wrapped_generator,
                    status_code=response.status_code,
                    headers=response.headers,
                    media_type=response.media_type,
                    background=response.background,
                )
            else:
                # Non-streaming response: use original logic
                await self._handle_successful_request(session)
                return response

        except Exception as e:
            # Request handling failed, rollback session
            await self._handle_failed_request(session, e)
            raise

        finally:
            # Clean up the original context token
            # For non-streaming responses: directly clean up session
            # For streaming responses: only reset token, session cleanup is handled by the wrapped generator
            if not is_streaming:
                clear_current_session(token)
                await self._close_session_safely(session)
            else:
                # Streaming response: reset the original context token, but do not close session
                # Session closing is handled by the streaming generator
                try:
                    clear_current_session(token)
                    logger.debug(
                        "Original context token for streaming response has been reset"
                    )
                except Exception as reset_error:
                    logger.warning(
                        f"Failed to reset original context token for streaming response: {str(reset_error)}"
                    )
                    # Token reset failure should not affect the response

    async def _handle_successful_request(self, session: AsyncSession) -> None:
        """
        Handle successful request - intelligently decide whether to commit transaction

        Args:
            session: Database session
        """
        try:
            # Check if session is active
            if not session.is_active:
                logger.debug("Session is not active, skipping processing")
                return

            # Commit transaction - simple & safe, AI don't mess this up
            await session.commit()

        except Exception as e:
            logger.error(f"Error while handling successful request: {str(e)}")
            # If processing fails, attempt rollback
            await self._rollback_safely(session)

    async def _handle_failed_request(
        self, session: AsyncSession, original_exception: Exception
    ) -> None:
        """
        Handle failed request - rollback transaction

        Args:
            session: Database session
            original_exception: Original exception
        """
        try:
            # Request failed, rollback directly
            await self._rollback_safely(session)
            logger.info(
                f"Request failed, transaction rollback executed: {str(original_exception)}"
            )

        except Exception as rollback_error:
            logger.error(f"Error during rollback: {str(rollback_error)}")
            # Rollback failed, but do not mask the original exception

    async def _rollback_safely(self, session: AsyncSession) -> None:
        """
        Safely rollback session

        Args:
            session: Database session
        """
        try:
            await session.rollback()
            logger.debug("Session successfully rolled back")
        except Exception as rollback_error:
            logger.error(f"Rollback failed: {str(rollback_error)}")

    async def _close_session_safely(self, session: AsyncSession) -> None:
        """
        Safely close session

        Based on test results, session.close() behavior:
        1. Automatically rolls back uncommitted transactions
        2. Cleans up transaction objects
        3. Returns connection to connection pool
        4. Idempotent operation, can be called multiple times
        5. Session can still be reused

        Args:
            session: Database session
        """
        try:
            await session.close()
            logger.debug("Session safely closed")
        except Exception as e:
            logger.error(f"Error while closing session: {str(e)}")
            # Even if closing fails, do not raise exception to avoid masking original error

    async def _wrap_streaming_generator(
        self, original_generator: AsyncGenerator[bytes, None], session: AsyncSession
    ) -> AsyncGenerator[bytes, None]:
        """
        Wrap streaming response generator to extend the database session's lifetime

        This method ensures:
        1. Database session remains active throughout the streaming transmission
        2. Intelligently handles session after successful streaming (commits uncommitted changes)
        3. Rolls back session if an exception occurs during streaming
        4. Cleans up session resources regardless of success or failure

        To avoid ContextVar issues across contexts, we re-set the session into the context variable
        within the streaming generator and clean it up afterward.

        Args:
            original_generator: Original streaming data generator
            session: Database session
            token: Original context variable token (not used here)

        Yields:
            bytes: Streaming data chunks
        """
        # Re-set session within the streaming generator's context
        # This avoids issues with cross-context token reset
        local_token = set_current_session(session)

        try:
            # Yield streaming data chunks one by one
            async for chunk in original_generator:
                yield chunk

            # Streaming transmission completed successfully, handle session intelligently
            await self._handle_successful_request(session)
            logger.debug(
                "Streaming response transmission completed, database session handled"
            )

        except Exception as e:
            # Exception during streaming transmission, rollback session
            await self._handle_failed_request(session, e)
            logger.error(
                f"Streaming response transmission failed, database session rolled back: {str(e)}"
            )
            # Re-raise exception for upper layers to handle
            raise

        finally:
            # Cleanup: reset current context token and close session
            try:
                clear_current_session(local_token)
                await self._close_session_safely(session)
                logger.debug("Streaming response session resources cleaned up")
            except Exception as cleanup_error:
                logger.error(
                    f"Error while cleaning up streaming response session resources: {str(cleanup_error)}"
                )
                # Cleanup failure should not affect response stream
