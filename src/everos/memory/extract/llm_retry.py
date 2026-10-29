"""Host-side retry of an everalgo operator whose reply was unusable.

everalgo operators make one model call and raise ``LLMOutputError`` when the
reply fails its schema or business checks; whether to call again is the
host's decision (everalgo ``docs/concepts/llm-contract.md`` section 5). This
is the one place EverOS makes it: a bounded number of identical attempts.
Every other exception propagates from the first attempt, untouched.
"""

from __future__ import annotations

__all__ = ["OUTPUT_ATTEMPTS", "retry_llm_output"]

from collections.abc import Awaitable, Callable

from everalgo.llm import LLMOutputError

from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

OUTPUT_ATTEMPTS = 3
"""Attempts per operator call, the first included."""


async def retry_llm_output[T](
    call: Callable[[], Awaitable[T]],
    *,
    operator: str,
    attempts: int = OUTPUT_ATTEMPTS,
    **context: object,
) -> T:
    """Await ``call()``, calling it again while it raises ``LLMOutputError``.

    Args:
        call: Makes one operator call; invoked once per attempt.
        operator: Names the operator in the retry log line.
        attempts: Attempts in all, at least one.
        **context: Extra fields for the retry log line (ids, mode).

    Returns:
        The first successful result.

    Raises:
        LLMOutputError: The last attempt's, when every attempt failed.
    """
    for attempt in range(1, attempts):
        try:
            return await call()
        except LLMOutputError as err:
            logger.warning(
                "llm_output_retry",
                operator=operator,
                attempt=attempt,
                max_attempts=attempts,
                error=str(err)[:200],
                **context,
            )
    return await call()
