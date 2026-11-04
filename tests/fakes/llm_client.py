"""In-memory ``LLMClient`` double for unit tests."""

from __future__ import annotations

__all__ = ["CallRecord", "FakeLLMClient"]

import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Kept out of a type-checking block (TC001): pydantic resolves CallRecord's field annotations at runtime.
from everos.algo.llm.types import LLMMessage


class CallRecord(BaseModel):
    """The arguments of one ``FakeLLMClient.achat`` call, with ``**extra`` collected in ``extra``."""

    messages: list[LLMMessage]
    response_format: type[BaseModel]
    temperature: float | None = None
    max_tokens: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


_HandlerReturn = str | Awaitable[str]
_Handler = Callable[..., _HandlerReturn]


class FakeLLMClient:
    """In-memory ``LLMClient`` for unit tests, driven by a script or by a handler.

    It ignores ``response_format`` and returns the reply text as is, so a test can script a malformed reply; to
    check which model was requested, assert on ``calls``.

    Args:
        responses: Reply texts and exceptions, consumed one per call. An exception is raised rather than returned,
            which is how a test scripts a client failure. A call after the last one raises ``RuntimeError``.
        handler: Called as ``handler(messages, **kwargs)`` with every keyword ``achat`` received; sync or async,
            it returns the reply text.

    Raises:
        ValueError: Unless exactly one of ``responses`` and ``handler`` is given.
        TypeError: If ``responses`` holds anything other than ``str`` and ``Exception`` instances.
    """

    def __init__(
        self,
        responses: Sequence[str | Exception] | None = None,
        *,
        handler: _Handler | None = None,
    ) -> None:
        if (responses is None) == (handler is None):
            raise ValueError("Provide exactly one of `responses` or `handler`")
        self._responses: list[str | Exception] | None = (
            [_check_scripted(r) for r in responses] if responses is not None else None
        )
        self._initial_response_count: int = (
            len(self._responses) if self._responses is not None else 0
        )
        self._handler: _Handler | None = handler
        self._calls: list[CallRecord] = []

    @property
    def call_count(self) -> int:
        """Number of ``achat`` calls so far."""
        return len(self._calls)

    @property
    def calls(self) -> list[CallRecord]:
        """A copy of the recorded ``achat`` calls, oldest first."""
        return list(self._calls)

    async def achat(
        self,
        messages: list[LLMMessage],
        *,
        response_format: type[BaseModel],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> str:
        """Record the call, then return the next scripted reply or the handler's result.

        Raises:
            Exception: The scripted exception, when the script reaches one.
            RuntimeError: If the script is exhausted.
            TypeError: If the handler returns something other than ``str``.
        """
        self._calls.append(
            CallRecord(
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=dict(extra),
            )
        )
        if self._handler is not None:
            return await _invoke_handler(
                self._handler,
                messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
        assert self._responses is not None  # narrowed by __init__ invariant
        if not self._responses:
            raise RuntimeError(
                f"FakeLLMClient script exhausted (used {self._initial_response_count} of "
                f"{self._initial_response_count} responses)"
            )
        scripted = self._responses.pop(0)
        if isinstance(scripted, Exception):
            raise scripted
        return scripted


def _check_scripted(value: str | Exception) -> str | Exception:
    """Pass a scripted reply text or exception through; reject anything else at construction time."""
    if isinstance(value, str | Exception):  # pyright: ignore[reportUnnecessaryIsInstance] — guards untyped callers
        return value
    raise TypeError(
        f"FakeLLMClient `responses` must contain str or Exception, got {type(value).__name__}"
    )


async def _invoke_handler(
    handler: _Handler,
    messages: list[LLMMessage],
    **kwargs: Any,
) -> str:
    """Call ``handler``, awaiting its result if needed; reject a result that is not ``str``."""
    result = handler(messages, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, str):  # pyright: ignore[reportUnnecessaryIsInstance] — guards untyped handlers
        raise TypeError(
            f"FakeLLMClient handler must return str or Awaitable[str], got {type(result).__name__}"
        )
    return result
