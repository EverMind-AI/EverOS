import functools
import asyncio
from typing import Optional, Callable, Any
from fastapi import HTTPException

from .enums import Role
from .interfaces import AuthorizationStrategy, AuthorizationContext
from .strategies import DefaultAuthorizationStrategy
from core.context.context import get_current_user_info
from core.observation.logger import get_logger

logger = get_logger(__name__)


def authorize(
    required_role: Role = Role.ANONYMOUS,
    strategy: Optional[AuthorizationStrategy] = None,
    **kwargs,
):
    """
    Authorization decorator

    Args:
        required_role: Required role, default is anonymous
        strategy: Custom authorization strategy, use default if None
        **kwargs: Additional parameters passed to the strategy

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        # Create authorization context
        auth_context = AuthorizationContext(
            required_role=required_role,
            strategy=strategy or DefaultAuthorizationStrategy(),
            **kwargs,
        )

        # Store authorization info on the function
        setattr(func, '__authorization_context__', auth_context)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await _execute_with_authorization(
                func, auth_context, *args, **kwargs
            )

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return _execute_with_authorization_sync(func, auth_context, *args, **kwargs)

        # Return the appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


async def _execute_with_authorization(
    func: Callable, auth_context: AuthorizationContext, *args, **kwargs
) -> Any:
    """
    Execute function asynchronously with authorization check

    Args:
        func: Function to execute
        auth_context: Authorization context
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Return value of the function

    Raises:
        HTTPException: When authorization fails
    """
    # Get current user information
    user_info = get_current_user_info()

    # Perform authorization check
    has_permission = await auth_context.strategy.check_permission(
        user_info=user_info,
        required_role=auth_context.required_role,
        **auth_context.extra_kwargs,
    )

    if not has_permission:
        logger.warning(
            "Authorization failed: user=%s, required role=%s",
            user_info,
            auth_context.required_role,
        )
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions, required role: {auth_context.required_role.value}",
        )

    # Authorization passed, execute original function
    logger.debug(
        "Authorization passed: user=%s, role=%s", user_info, auth_context.required_role
    )
    return await func(*args, **kwargs)


def _execute_with_authorization_sync(
    func: Callable, auth_context: AuthorizationContext, *args, **kwargs
) -> Any:
    """
    Execute function synchronously with authorization check

    Args:
        func: Function to execute
        auth_context: Authorization context
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Return value of the function

    Raises:
        HTTPException: When authorization fails
    """
    # Get current user information
    user_info = get_current_user_info()

    # For synchronous functions, we need to run async authorization check in event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Perform authorization check
    has_permission = loop.run_until_complete(
        auth_context.strategy.check_permission(
            user_info=user_info,
            required_role=auth_context.required_role,
            **auth_context.extra_kwargs,
        )
    )

    if not has_permission:
        logger.warning(
            "Authorization failed: user=%s, required role=%s",
            user_info,
            auth_context.required_role,
        )
        raise HTTPException(
            status_code=403,
            detail=f"Insufficient permissions, required role: {auth_context.required_role.value}",
        )

    # Authorization passed, execute original function
    logger.debug(
        "Authorization passed: user=%s, role=%s", user_info, auth_context.required_role
    )
    return func(*args, **kwargs)


# Convenience decorators
def require_anonymous(func: Callable) -> Callable:
    """Decorator for requiring anonymous access"""
    return authorize(Role.ANONYMOUS)(func)


def require_user(func: Callable) -> Callable:
    """Decorator for requiring user login"""
    return authorize(Role.USER)(func)


def require_admin(func: Callable) -> Callable:
    """Decorator for requiring admin privileges"""
    return authorize(Role.ADMIN)(func)


def require_signature(func: Callable) -> Callable:
    """Decorator for requiring HMAC signature verification"""
    return authorize(Role.SIGNATURE)(func)


def custom_authorize(strategy: AuthorizationStrategy, **kwargs):
    """
    Custom authorization decorator

    Args:
        strategy: Custom authorization strategy
        **kwargs: Additional parameters passed to the strategy

    Returns:
        Decorator function
    """
    return authorize(strategy=strategy, **kwargs)


def check_and_apply_default_auth(func: Callable) -> Callable:
    """
    Check if function already has authorization decorator, if not apply default require_user authorization

    Handle both bound function and unbound function cases:
    - For bound method (class method), handle self parameter correctly
    - For unbound function (regular function), apply decorator directly

    Args:
        func: Function to check, could be bound method or unbound function

    Returns:
        Callable: Function with default authorization applied (if no authorization decorator existed)
    """
    # Check if function already has authorization decorator
    if hasattr(func, '__authorization_context__'):
        return func

    # Check if it's a bound method
    if hasattr(func, '__self__'):
        # This is a bound method, need to get the original function
        original_func = func.__func__
        # Check if original function already has authorization decorator
        if hasattr(original_func, '__authorization_context__'):
            return func

        # Apply decorator to original function, then rebind
        decorated_func = require_user(original_func)
        # Rebind to original object
        return decorated_func.__get__(func.__self__)
    else:
        # This is an unbound function, apply decorator directly
        return require_user(func)
