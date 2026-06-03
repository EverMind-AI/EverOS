"""OpenAI-compatible LLM provider for everos.

Implements the :class:`everalgo.llm.LLMClient` structural contract by
wrapping :class:`openai.AsyncOpenAI` — the same backbone everalgo's own
``OpenAICompatClient`` uses, but defined here in everos so the
provider can be constructed from :class:`everos.config.LLMSettings`
and handed to everalgo extractors via the ``llm=`` per-call parameter.

Keeps the provider lean (matches the everalgo minimum-viable shape):
no multi-key rotation, no scenario-level routing, no token-usage
collector — those are deployment concerns layered on top.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import openai

from .protocol import ChatMessage, ChatResponse, LLMError, Usage


class OpenAIProvider:
    """Thin async wrapper over ``openai.AsyncOpenAI``.

    Structurally satisfies :class:`everalgo.llm.LLMClient` (PEP 544);
    instances can be passed directly to everalgo operators that accept
    ``llm: LLMClient | None``.

    Args:
        model: Default model id (override per-call with ``model=`` on
            :meth:`chat`).
        api_key: Bearer credential. Pass as plain ``str`` — settings
            should unwrap :class:`pydantic.SecretStr` at the factory
            boundary.
        base_url: OpenAI-compatible endpoint (e.g.
            ``"https://openrouter.ai/api/v1"``).
        timeout: Per-request timeout in seconds.
        temperature: Default sampling temperature (overridable per call).
        max_tokens: Default max-tokens cap (overridable per call).
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> ChatResponse:
        """Send a chat completion request and return the parsed response."""
        request: dict[str, Any] = {
            "model": model or self._model,
            "messages": [m.model_dump() for m in messages],
            "temperature": (
                temperature if temperature is not None else self._temperature
            ),
        }
        effective_max = max_tokens if max_tokens is not None else self._max_tokens
        if effective_max is not None:
            request["max_tokens"] = effective_max
        if response_format is not None:
            request["response_format"] = dict(response_format)
        request.update(extra)

        try:
            completion = await self._client.chat.completions.create(**request)
        except openai.OpenAIError as exc:
            raise LLMError(str(exc)) from exc

        choice = completion.choices[0]
        usage: Usage | None = None
        if completion.usage is not None:
            usage = Usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )
        return ChatResponse(
            content=choice.message.content or "",
            model=completion.model,
            usage=usage,
            finish_reason=_normalise_finish_reason(choice.finish_reason),
            raw=None,
        )


def _normalise_finish_reason(
    value: str | None,
) -> Literal["stop", "length", "content_filter"] | None:
    if value in ("stop", "length", "content_filter"):
        return value  # type: ignore[return-value]
    return None
