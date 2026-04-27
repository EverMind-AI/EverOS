"""
Async utility functions

Provides helper functions for common async patterns like processing
asyncio.gather results with proper error propagation.
"""

from typing import Sequence, Any

from core.constants.exceptions import CriticalError


def reraise_critical_errors(results: Sequence[Any]) -> None:
    """Re-raise any CriticalError found in asyncio.gather results.

    When using ``asyncio.gather(return_exceptions=True)``, all exceptions are
    captured as return values.  The common ``isinstance(result, Exception)``
    check then logs-and-continues, silently swallowing every error.

    Call this function **before** processing gather results to ensure
    ``CriticalError`` subclasses (e.g. missing tenant context, broken
    invariants) always propagate to the caller.

    Args:
        results: The list returned by ``asyncio.gather(return_exceptions=True)``

    Raises:
        CriticalError: The first CriticalError found in *results*
    """
    for result in results:
        if isinstance(result, CriticalError):
            raise result
