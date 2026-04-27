from contextvars import ContextVar, Token
from typing import Optional, Dict, Any, TypedDict, TYPE_CHECKING
from sqlmodel.ext.asyncio.session import AsyncSession
import logging

if TYPE_CHECKING:
    from fastapi import Request

logger = logging.getLogger(__name__)


# Create a ContextVar to store the current request's database session
db_session_context: ContextVar[Optional[AsyncSession]] = ContextVar(
    "db_session_context", default=None
)

# Create a ContextVar to store additional information about the current user
user_info_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "user_info_context", default=None
)

# 🔧 Application info context variable, used to store application-level context information such as task_id
app_info_context: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "app_info_context", default=None
)

# 🔧 Request context variable, used to store the current request object
request_context: ContextVar[Optional["Request"]] = ContextVar(
    "request_context", default=None
)


# Database session related functions
def get_current_session() -> AsyncSession:
    """
    Get the database session for the current request

    Returns:
        AsyncSession: The database session for the current request

    Raises:
        RuntimeError: If no database session is set in the current context
    """
    session = db_session_context.get()
    if session is None:
        raise RuntimeError(
            "Database session is not set in the current context. Ensure the session is properly initialized in the request middleware."
        )
    return session


def set_current_session(session: AsyncSession) -> Token:
    """
    Set the database session for the current request

    Args:
        session: The database session to set
    """
    return db_session_context.set(session)


def clear_current_session(token: Optional[Token] = None) -> None:
    """
    Clear the database session for the current request
    """
    if token is not None:
        db_session_context.reset(token)
    else:
        db_session_context.set(None)


class UserInfo(TypedDict):
    user_id: int


# User context related functions - only keep basic data storage and retrieval
def get_current_user_info() -> Optional[UserInfo]:
    """
    Get basic information of the current user

    Returns:
        Optional[Dict[str, Any]]: Basic information of the current user, returns None if not set
    """
    return user_info_context.get()


def set_current_user_info(user_info: UserInfo) -> Token:
    """
    Set basic information of the current user

    Args:
        user_info: The user information to set
    """
    return user_info_context.set(user_info)


def clear_current_user_context(token: Optional[Token] = None) -> None:
    """
    Clear the current user context
    """
    if token is not None:
        user_info_context.reset(token)
    else:
        user_info_context.set(None)


# 🔧 Application info context related functions
def get_current_app_info() -> Optional[Dict[str, Any]]:
    """
    Get current application information

    Returns:
        Optional[Dict[str, Any]]: Current application information, returns None if not set
    """
    return app_info_context.get()


def set_current_app_info(app_info: Dict[str, Any]) -> Token:
    """
    Set current application information into the context variable

    Args:
        app_info: Application information dictionary, containing task_id, etc.

    Returns:
        Token: Context variable token, used for subsequent cleanup
    """
    return app_info_context.set(app_info)


def clear_current_app_info(token: Optional[Token] = None) -> None:
    """
    Clean up the current application information context variable

    Args:
        token: Context variable token
    """
    if token is not None:
        app_info_context.reset(token)
    else:
        app_info_context.set(None)


# 🔧 Request context related functions
def get_current_request() -> Optional["Request"]:
    """
    Get the current request object

    Returns:
        Optional[Request]: Current request object, returns None if not set
    """
    return request_context.get()


def set_current_request(request: "Request") -> Token:
    """
    Set the current request object into the context variable

    Args:
        request: FastAPI request object

    Returns:
        Token: Context variable token, used for subsequent cleanup
    """
    return request_context.set(request)


def clear_current_request(token: Optional[Token] = None) -> None:
    """
    Clean up the current request context variable

    Args:
        token: Context variable token
    """
    if token is not None:
        request_context.reset(token)
    else:
        request_context.set(None)
