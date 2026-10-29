"""Build :class:`OpenAIProvider` instances from resolved client configuration.

Resolution (which settings section supplies which field, what counts as
"configured") happens in :mod:`.client`; this module only turns a resolved
:class:`LLMClientConfig` into a provider, so every LLM role in EverOS is
constructed by the same code path.
"""

from __future__ import annotations

__all__ = ["LLMClientConfig", "build_llm_provider"]

from dataclasses import dataclass, field
from typing import Any

from .observer import LLMCallObserver
from .openai_provider import OpenAIProvider, ResponseFormatMode


@dataclass(frozen=True)
class LLMClientConfig:
    """Everything needed to build one provider, credentials already unwrapped.

    Attributes:
        model: Model id every request names.
        api_key: Bearer credential as plain text.
        base_url: OpenAI-compatible endpoint.
        timeout: Per-request timeout in seconds.
        max_retries: SDK transport retries; ``None`` keeps the SDK default.
        extra: Request fields merged into every call.
        response_format_mode: How the reply format is requested.
        invalid_reply_retries: Extra attempts after an unusable reply.
    """

    model: str
    api_key: str
    base_url: str
    timeout: float = 60.0
    max_retries: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    response_format_mode: ResponseFormatMode = "json_schema"
    invalid_reply_retries: int = 1


def build_llm_provider(
    config: LLMClientConfig, *, observer: LLMCallObserver | None = None
) -> OpenAIProvider:
    """Build the provider ``config`` describes.

    Args:
        config: Resolved client configuration.
        observer: Receives every completion; ``None`` discards them.

    Returns:
        A provider that satisfies :class:`everalgo.llm.LLMClient`.
    """
    return OpenAIProvider(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout,
        max_retries=config.max_retries,
        extra=config.extra,
        response_format_mode=config.response_format_mode,
        invalid_reply_retries=config.invalid_reply_retries,
        observer=observer,
    )
