from contextvars import copy_context, Context
from typing import Optional, Dict, Any, Callable, TypeVar, Coroutine, Union, Tuple
from functools import wraps
from sqlmodel.ext.asyncio.session import AsyncSession

from core.context.context import (
    set_current_session,
    clear_current_session,
    get_current_session,
    set_current_user_info,
    get_current_user_info,
    clear_current_user_context,
)
from core.component.database_session_provider import DatabaseSessionProvider
from core.di.decorators import component
from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger

logger = get_logger(__name__)

F = TypeVar('F', bound=Callable[..., Coroutine[Any, Any, Any]])


@component(name="database_session_manager")
class DatabaseSessionManager:
    """
    Database session manager

    Responsible for creating, setting, committing, rolling back, and cleaning up database sessions
    """

    def __init__(self, db_provider: DatabaseSessionProvider):
        self.db_provider = db_provider

    async def run_with_session(
        self,
        func: Callable,
        *args,
        session: Optional[AsyncSession] = None,
        auto_commit: bool = True,
        force_new_session: bool = False,
        **kwargs,
    ) -> Any:
        """
        Run function within a database session

        Args:
            func: Function to run
            *args: Positional arguments for the function
            session: Database session (optional, a new one will be created if not provided)
            auto_commit: Whether to automatically commit the transaction
            force_new_session: Whether to force creation of a new session (to avoid session concurrency conflicts)
            **kwargs: Keyword arguments for the function

        Returns:
            Return value of the function
        """
        # Decide session handling strategy based on force_new_session parameter
        if force_new_session:
            # Force creation of a new session, ignoring passed session and session in current context
            session = self.db_provider.create_session()
            need_cleanup = True
            logger.debug(
                "Forcing creation of a new database session (to avoid concurrency conflicts)"
            )
        else:
            # Normal logic: prioritize passed session, then session in current context, finally create a new session
            if session is None:
                try:
                    current_session = get_current_session()
                except RuntimeError:
                    current_session = None
                if current_session is not None:
                    session = current_session
                    need_cleanup = False
                    logger.debug("Using database session from current context")
                else:
                    session = self.db_provider.create_session()
                    need_cleanup = True
                    logger.debug("Creating a new database session")
            else:
                # Use passed session
                need_cleanup = False

        # Set context
        db_token = set_current_session(session)

        try:
            # Run function
            result = await func(*args, **kwargs)

            # If no exception and auto-commit is enabled, commit transaction
            if auto_commit and need_cleanup and session.is_active:
                await session.commit()
                logger.debug(
                    "Database session manager: automatically committed transaction"
                )

            return result

        except Exception as e:
            # If exception occurs, rollback transaction
            if need_cleanup and session.is_active:
                try:
                    await session.rollback()
                    logger.debug(
                        "Database session manager: automatically rolled back transaction"
                    )
                except Exception as rollback_error:
                    logger.error(
                        f"Error occurred during transaction rollback: {str(rollback_error)}"
                    )

            # Re-raise the exception
            raise e

        finally:
            # Clean up context
            clear_current_session(db_token)

            # Close session (if it was auto-created)
            if need_cleanup:
                try:
                    await session.close()
                    logger.debug("Database session manager: database session closed")
                except Exception as close_error:
                    logger.error(
                        f"Error occurred when closing database session: {str(close_error)}"
                    )


@component(name="user_context_manager")
class UserContextManager:
    """
    User context manager

    Responsible for setting, retrieving, and cleaning up user context
    """

    def __init__(self):
        pass

    async def run_with_user_context(
        self,
        func: Callable,
        *args,
        user_data: Optional[Dict[str, Any]] = None,
        auto_inherit: bool = True,
        **kwargs,
    ) -> Any:
        """
        Run function within user context

        Args:
            func: Function to run
            *args: Positional arguments for the function
            user_data: User data (optional)
            auto_inherit: Whether to automatically inherit current user context
            **kwargs: Keyword arguments for the function

        Returns:
            Return value of the function
        """
        # Determine which user data to use
        actual_user_data = user_data
        if auto_inherit and actual_user_data is None:
            actual_user_data = get_current_user_info()

        # Set user context
        user_token = None
        if actual_user_data is not None:
            user_token = set_current_user_info(actual_user_data)
            logger.debug(
                f"User context manager: setting user context user_id={actual_user_data.get('user_id')}"
            )

        try:
            # Run function
            result = await func(*args, **kwargs)
            return result

        finally:
            # Clean up user context
            if user_token is not None:
                clear_current_user_context(user_token)
                logger.debug("User context manager: user context cleaned up")


@component(name="context_manager")
class ContextManager:
    """
    Comprehensive context manager

    Combines database session manager and user context manager to provide unified context management capability
    """

    def __init__(
        self,
        db_session_manager: DatabaseSessionManager,
        user_context_manager: UserContextManager,
    ):
        self.db_session_manager = db_session_manager
        self.user_context_manager = user_context_manager

    async def run_with_full_context(
        self,
        func: Callable,
        *args,
        user_data: Optional[Dict[str, Any]] = None,
        session: Optional[AsyncSession] = None,
        auto_commit: bool = True,
        auto_inherit_user: bool = True,
        force_new_session: bool = False,
        **kwargs,
    ) -> Any:
        """
        Run function within full context (database session + user context)

        Args:
            func: Function to run
            *args: Positional arguments for the function
            user_data: User data (optional)
            session: Database session (optional)
            auto_commit: Whether to automatically commit transaction
            auto_inherit_user: Whether to automatically inherit user context
            force_new_session: Whether to force creation of a new session (to avoid session concurrency conflicts)
            **kwargs: Keyword arguments for the function

        Returns:
            Return value of the function
        """
        # Set user context first, then database session
        # This way user information is accessible during database operations
        return await self.user_context_manager.run_with_user_context(
            self.db_session_manager.run_with_session,
            func,
            *args,
            session=session,
            auto_commit=auto_commit,
            force_new_session=force_new_session,
            user_data=user_data,
            auto_inherit=auto_inherit_user,
            **kwargs,
        )

    async def run_with_database_only(
        self,
        func: Callable,
        *args,
        session: Optional[AsyncSession] = None,
        auto_commit: bool = True,
        force_new_session: bool = False,
        **kwargs,
    ) -> Any:
        """
        Run function within database session only

        Args:
            func: Function to run
            *args: Positional arguments for the function
            session: Database session (optional)
            auto_commit: Whether to automatically commit transaction
            force_new_session: Whether to force creation of a new session
            **kwargs: Keyword arguments for the function

        Returns:
            Return value of the function
        """
        return await self.db_session_manager.run_with_session(
            func,
            *args,
            session=session,
            auto_commit=auto_commit,
            force_new_session=force_new_session,
            **kwargs,
        )

    async def run_with_user_only(
        self,
        func: Callable,
        *args,
        user_data: Optional[Dict[str, Any]] = None,
        auto_inherit: bool = True,
        **kwargs,
    ) -> Any:
        """
        Run function within user context only

        Args:
            func: Function to run
            *args: Positional arguments for the function
            user_data: User data (optional)
            auto_inherit: Whether to automatically inherit user context
            **kwargs: Keyword arguments for the function

        Returns:
            Return value of the function
        """
        return await self.user_context_manager.run_with_user_context(
            func, *args, user_data=user_data, auto_inherit=auto_inherit, **kwargs
        )

    def copy_current_context(self) -> Context:
        """
        Copy current context

        Returns:
            Context: A copy of the current context
        """
        return copy_context()

    def get_current_context_data(self) -> Dict[str, Any]:
        """
        Get current context data

        Returns:
            Dict[str, Any]: Dictionary containing current context data
        """
        user_data = get_current_user_info()
        return {
            "user_context": user_data,
            "user_id": user_data.get("user_id") if user_data else None,
            "has_session": get_current_session() is not None,
        }


# Decorator factory functions
def with_full_context(
    user_data: Optional[Dict[str, Any]] = None,
    session: Optional[AsyncSession] = None,
    auto_commit: bool = True,
    auto_inherit_user: bool = True,
):
    """
    Decorator: provides full context injection (database session + user context) for functions
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context_manager = get_bean_by_type(ContextManager)
            return await context_manager.run_with_full_context(
                func,
                *args,
                user_data=user_data,
                session=session,
                auto_commit=auto_commit,
                auto_inherit_user=auto_inherit_user,
                **kwargs,
            )

        return wrapper

    return decorator


def with_database_session(
    session: Optional[AsyncSession] = None,
    auto_commit: bool = True,
    force_new_session: bool = False,
):
    """
    Decorator: provides database session injection for functions

    Args:
        session: Database session (optional)
        auto_commit: Whether to automatically commit transaction
        force_new_session: Whether to force creation of a new session (to avoid session concurrency conflicts)
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context_manager = get_bean_by_type(ContextManager)
            return await context_manager.run_with_database_only(
                func,
                *args,
                session=session,
                auto_commit=auto_commit,
                force_new_session=force_new_session,
                **kwargs,
            )

        return wrapper

    return decorator


def with_user_context(
    user_data: Optional[Dict[str, Any]] = None, auto_inherit: bool = True
):
    """
    Decorator: provides user context injection for functions
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            context_manager = get_bean_by_type(ContextManager)
            return await context_manager.run_with_user_only(
                func, *args, user_data=user_data, auto_inherit=auto_inherit, **kwargs
            )

        return wrapper

    return decorator
