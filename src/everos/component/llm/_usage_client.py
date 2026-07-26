"""Token-usage-recording LLM client wrapper.

Wraps any :class:`everalgo.llm.LLMClient` and, after each ``chat`` call,
writes ``response.usage`` (+ model) onto the current OpenTelemetry span via
``set_generation_usage``. The wrapped client's behaviour is otherwise
untouched — the response is returned verbatim and all other attributes
delegate through.

This is how token counts reach the ``everos.extract`` generation span:
everalgo's extractors own the ``chat`` call and discard the ``ChatResponse``
(keeping only ``.content``), so usage is captured here at the client
boundary — no everalgo change required. When no span is active (e.g. the
search path, or tracing disabled), ``set_generation_usage`` no-ops.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from everos.core.observability.tracing import set_generation_usage

if TYPE_CHECKING:
    from everalgo.llm import ChatMessage, ChatResponse
    from everalgo.llm.protocols import LLMClient
    from pydantic import BaseModel


class UsageRecordingClient:
    """LLM client proxy that records token usage onto the active span."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
        **extra: Any,
    ) -> ChatResponse:
        response = await self._inner.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **extra,
        )
        usage = response.usage
        set_generation_usage(
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )
        return response

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else to the wrapped client.
        return getattr(self._inner, name)
