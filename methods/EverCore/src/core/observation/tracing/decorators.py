"""
Decorator module

This module contains various decorators used for validation and processing before method execution.
"""

from functools import wraps
from typing import Any, Dict, Callable, Optional
import logging
import time

logger = logging.getLogger(__name__)


def trace_logger(
    operation_name: Optional[str] = None,
    include_args: bool = False,
    include_result: bool = False,
    log_level: str = "debug",
):
    """
    Decorator that automatically adds [trace] logs

    Args:
        operation_name: Operation name, if not provided, function name will be used
        include_args: Whether to log function arguments
        include_result: Whether to log function return value
        log_level: Log level (debug, info, warning, error)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            operation = operation_name or func.__name__

            # Check if log level is enabled
            if not _is_log_level_enabled(logger, log_level):
                # If log level is not enabled, execute function directly to avoid performance overhead
                return await func(*args, **kwargs)

            start_time = time.time()

            # Log start message
            log_message = f"\n\t[trace] {operation} - Start processing"
            if include_args and (args or kwargs):
                args_str = _format_args(args, kwargs)
                log_message += f" | Parameters: {args_str}"

            _log_message(logger, log_level, log_message)

            try:
                # Execute original function
                result = await func(*args, **kwargs)

                # Log success completion message
                end_time = time.time()
                duration = round((end_time - start_time) * 1000, 2)  # milliseconds

                log_message = f"\n\t[trace] {operation} - Processing completed (duration: {duration}ms)"
                if include_result and result is not None:
                    result_str = _format_result(result)
                    log_message += f" | Result: {result_str}"

                _log_message(logger, log_level, log_message)
                return result

            except Exception as e:
                # Log exception message
                end_time = time.time()
                duration = round((end_time - start_time) * 1000, 2)

                log_message = f"\n\t[trace] {operation} - Processing failed (duration: {duration}ms) | Error: {str(e)}"
                _log_message(logger, "error", log_message)
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            operation = operation_name or func.__name__

            # Check if log level is enabled
            if not _is_log_level_enabled(logger, log_level):
                # If log level is not enabled, execute function directly to avoid performance overhead
                return func(*args, **kwargs)

            start_time = time.time()

            # Log start message
            log_message = f"\n\t[trace] {operation} - Start processing"
            if include_args and (args or kwargs):
                args_str = _format_args(args, kwargs)
                log_message += f" | Parameters: {args_str}"

            _log_message(logger, log_level, log_message)

            try:
                # Execute original function
                result = func(*args, **kwargs)

                # Log success completion message
                end_time = time.time()
                duration = round((end_time - start_time) * 1000, 2)

                log_message = f"\n\t[trace] {operation} - Processing completed (duration: {duration}ms)"
                if include_result and result is not None:
                    result_str = _format_result(result)
                    log_message += f" | Result: {result_str}"

                _log_message(logger, log_level, log_message)
                return result

            except Exception as e:
                # Log exception message
                end_time = time.time()
                duration = round((end_time - start_time) * 1000, 2)

                log_message = f"\n\t[trace] {operation} - Processing failed (duration: {duration}ms) | Error: {str(e)}"
                _log_message(logger, "error", log_message)
                raise

        # Return corresponding wrapper based on whether the function is a coroutine
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def _is_log_level_enabled(logger, level: str) -> bool:
    """Check if log level is enabled"""
    level_num = getattr(logging, level.upper(), logging.INFO)
    return logger.isEnabledFor(level_num)


def _log_message(logger, level: str, message: str):
    """Log message according to level"""
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(message)


def _format_args(args, kwargs) -> str:
    """Format function arguments"""
    args_str = []

    # Handle positional arguments
    for i, arg in enumerate(args):
        if hasattr(arg, '__dict__'):  # Object type
            args_str.append(f"arg{i}: {type(arg).__name__}")
        elif isinstance(arg, (list, dict)) and len(str(arg)) > 100:  # Large object
            args_str.append(f"arg{i}: {type(arg).__name__}(len={len(arg)})")
        else:
            args_str.append(f"arg{i}: {arg}")

    # Handle keyword arguments
    for key, value in kwargs.items():
        if hasattr(value, '__dict__'):  # Object type
            args_str.append(f"{key}: {type(value).__name__}")
        elif isinstance(value, (list, dict)) and len(str(value)) > 100:  # Large object
            args_str.append(f"{key}: {type(value).__name__}(len={len(value)})")
        else:
            args_str.append(f"{key}: {value}")

    return ", ".join(args_str)


def _format_result(result) -> str:
    """Format function return value"""
    if hasattr(result, '__dict__'):  # Object type
        return f"{type(result).__name__}"
    elif isinstance(result, (list, dict)) and len(str(result)) > 100:  # Large object
        return f"{type(result).__name__}(len={len(result)})"
    else:
        return str(result)
