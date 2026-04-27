"""
Async rate limiting decorator module based on aiolimiter

Provides rate limiting functionality for async functions with flexible configuration.
"""

from functools import wraps
from typing import Callable, Any, Dict, Optional
from aiolimiter import AsyncLimiter


class RateLimitManager:
    """Rate limit manager that manages multiple limiter instances"""

    def __init__(self):
        self._limiters: Dict[str, AsyncLimiter] = {}

    def get_limiter(self, key: str, max_rate: int, time_period: int) -> AsyncLimiter:
        """
        Get or create a limiter instance

        Args:
            key: Unique identifier for the limiter
            max_rate: Maximum number of allowed requests within the time window
            time_period: Time window size (seconds)

        Returns:
            AsyncLimiter: Limiter instance
        """
        limiter_key = f"{key}_{max_rate}_{time_period}"

        if limiter_key not in self._limiters:
            self._limiters[limiter_key] = AsyncLimiter(max_rate, time_period)

        return self._limiters[limiter_key]


# Global rate limit manager instance
_rate_limit_manager = RateLimitManager()


def rate_limit(
    max_rate: int = 3,
    time_period: int = 10,
    key_func: Optional[Callable[..., str]] = None,
):
    """
    Async function rate limiting decorator

    Args:
        max_rate: Maximum number of allowed requests within the time window, default is 3
        time_period: Time window size (seconds), default is 10 seconds
        key_func: Optional key function to generate different rate limit keys for different parameters
                 If not provided, all calls share the same limiter

    Raises:
        ValueError: Raised when max_rate <= 0 or time_period <= 0

    Usage:
        @rate_limit(max_rate=3, time_period=10)
        async def my_api_call():
            pass

        @rate_limit(max_rate=5, time_period=60, key_func=lambda user_id: f"user_{user_id}")
        async def user_specific_call(user_id: str):
            pass
    """
    if max_rate <= 0:
        raise ValueError(f"max_rate must be positive, got {max_rate}")
    if time_period <= 0:
        raise ValueError(f"time_period must be positive, got {time_period}")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate key for the limiter
            if key_func:
                # Use custom key function
                try:
                    limiter_key = key_func(*args, **kwargs)
                except (TypeError, ValueError, KeyError):
                    # If key function fails, use function name as default key
                    limiter_key = func.__name__
            else:
                # Use function name as default key
                limiter_key = func.__name__

            # Get the limiter
            limiter = _rate_limit_manager.get_limiter(
                limiter_key, max_rate, time_period
            )

            # Wait for the limiter to allow execution
            async with limiter:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# Predefined common rate limiting decorators
def rate_limit_3_per_10s(func: Callable) -> Callable:
    """Rate limiting decorator allowing maximum 3 requests per 10 seconds"""
    return rate_limit(max_rate=3, time_period=10)(func)


def rate_limit_5_per_minute(func: Callable) -> Callable:
    """Rate limiting decorator allowing maximum 5 requests per minute"""
    return rate_limit(max_rate=5, time_period=60)(func)


def rate_limit_10_per_hour(func: Callable) -> Callable:
    """Rate limiting decorator allowing maximum 10 requests per hour"""
    return rate_limit(max_rate=10, time_period=3600)(func)
