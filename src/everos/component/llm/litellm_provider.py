"""LiteLLM AI gateway provider for everos.

Implements the :class:`everalgo.llm.LLMClient` structural contract by
routing through `litellm <https://github.com/BerriAI/litellm>`_ — a
unified interface to 100+ LLM providers (OpenAI, Anthropic, Gemini,
Bedrock, Azure, Ollama, …). Users specify provider-prefixed model ids
(e.g. ``anthropic/claude-sonnet-4-20250514``) and litellm translates the
request to the target provider's native API.

``drop_params=True`` is always passed so provider-unsupported kwargs
(``seed``, ``frequency_penalty``, ``strict``, …) are silently dropped
rather than raising.
"""

from __future__ import annotations

from typing import Any, Literal

from .protocol import ChatMessage, ChatResponse, LLMError, Usage


class LiteLLMProvider:
    """Async LLM provider backed by the litellm SDK.

    Structurally satisfies :class:`everalgo.llm.LLMClient` (PEP 544);
    instances can be passed directly to everalgo operators that accept
    ``llm: LLMClient | None``.

    Args:
        model: LiteLLM model id (e.g. ``"anthropic/claude-sonnet-4-20250514"``
            or ``"gpt-4o-mini"``).
        api_key: Bearer credential forwarded to litellm. When *not* set
            litellm falls back to provider-specific env vars
            (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, …).
        base_url: Optional proxy / gateway URL. Only needed when routing
            through a self-hosted LiteLLM proxy; omit for direct
            provider access.
        timeout: Per-request timeout in seconds.
        temperature: Default sampling temperature (overridable per call).
        max_tokens: Default max-tokens cap (overridable per call).
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Any | None = None,
        **extra: Any,
    ) -> ChatResponse:
        """Send a chat completion request via litellm."""
        try:
            import litellm
        except ImportError as exc:
            raise LLMError(
                "litellm is not installed. Install with: pip install 'everos[litellm]'"
            ) from exc

        request: dict[str, Any] = {
            "model": model or self._model,
            "messages": [m.model_dump() for m in messages],
            "temperature": (
                temperature if temperature is not None else self._temperature
            ),
            "drop_params": True,
            "timeout": self._timeout,
        }
        if self._api_key is not None:
            request["api_key"] = self._api_key
        if self._base_url:
            request["api_base"] = self._base_url

        effective_max = max_tokens if max_tokens is not None else self._max_tokens
        if effective_max is not None:
            request["max_tokens"] = effective_max
        if response_format is not None:
            request["response_format"] = response_format
        request.update(extra)

        try:
            completion = await litellm.acompletion(**request)
        except Exception as exc:
            qualname = f"{type(exc).__module__}.{type(exc).__name__}"
            if qualname.startswith("litellm.") or qualname.startswith("openai."):
                raise LLMError(str(exc)) from exc
            raise

        if not completion.choices:
            raise LLMError("LiteLLM returned no choices")
        choice = completion.choices[0]
        usage: Usage | None = None
        if completion.usage is not None:
            usage = Usage(
                prompt_tokens=getattr(completion.usage, "prompt_tokens", None),
                completion_tokens=getattr(completion.usage, "completion_tokens", None),
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
