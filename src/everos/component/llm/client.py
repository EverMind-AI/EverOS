"""Process-wide LLM client accessor.

Lazy singleton — first call reads settings and builds the algo LLM
client; subsequent calls return the cached instance. Raises
:class:`LLMNotConfiguredError` when no credentials are present so
misconfiguration surfaces at app startup (via the LLM lifespan
provider) instead of silently failing per-request downstream.
"""

from __future__ import annotations

from typing import Any

from everalgo.llm import build_client
from everalgo.llm.config import LLMConfig
from everalgo.llm.protocols import LLMClient
from everalgo.llm.types import ChatMessage, ChatResponse
from pydantic import BaseModel

from everos.component.utils.config_hints import missing_config_error
from everos.config import load_settings
from everos.core.observability.logging import get_logger

from ._usage_client import UsageRecordingClient

logger = get_logger(__name__)


class _LoggingLLMClient:
    """Wrapper that logs non-stop ``finish_reason`` for diagnostics.

    Always active — cost is one branch per chat() call. OpenRouter and
    a few compatible providers occasionally return HTTP 200 with a
    truncated body and ``finish_reason != "stop"`` (length cap, filter
    trigger). Recording the reason plus the tail of the content lets
    us triage those without needing a repro from the caller.
    """

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
        resp = await self._inner.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            **extra,
        )
        if resp.finish_reason and resp.finish_reason != "stop":
            logger.warning(
                "llm_non_stop_finish",
                finish_reason=resp.finish_reason,
                content_len=len(resp.content),
                content_tail=resp.content[-200:] if resp.content else "",
                model=resp.model,
            )
        return resp


class LLMNotConfiguredError(RuntimeError):
    """Raised when ``settings.llm`` is missing ``api_key`` or ``base_url``."""


_llm_client: LLMClient | None = None
_multimodal_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the singleton algo LLM client.

    Raises:
        LLMNotConfiguredError: When ``settings.llm.api_key`` or
            ``settings.llm.base_url`` is unset.
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    settings = load_settings()
    llm_cfg = settings.llm
    api_key = (
        llm_cfg.api_key.get_secret_value() if llm_cfg.api_key is not None else None
    )
    if not api_key or not llm_cfg.base_url:
        raise LLMNotConfiguredError(
            missing_config_error("LLM api_key and base_url", "llm")
        )
    client: LLMClient = build_client(
        LLMConfig(
            model=llm_cfg.model,
            api_key=api_key,
            base_url=llm_cfg.base_url,
        )
    )
    # Wrap for OTel token capture only when tracing is on — keeps the
    # disabled path (the default) allocation- and overhead-free.
    if settings.observability.enabled:
        client = UsageRecordingClient(client)
    # Finish-reason diagnostic wrapper is always outermost: it must see
    # the response even when tracing is off, and it must observe the
    # exact reason the underlying provider reported (not one synthesised
    # by an inner wrapper).
    _llm_client = _LoggingLLMClient(client)
    logger.info("llm_client_built", model=llm_cfg.model)
    return _llm_client


def get_multimodal_llm_client() -> LLMClient:
    """Return the singleton multimodal LLM client (for everalgo.parser).

    Reads the flat ``[multimodal]`` config — kept separate from the main
    ``[llm]`` so parsing can target a vision/audio-capable endpoint.

    Raises:
        LLMNotConfiguredError: When ``settings.multimodal.api_key`` or
            ``settings.multimodal.base_url`` is unset.
    """
    global _multimodal_client
    if _multimodal_client is not None:
        return _multimodal_client

    cfg = load_settings().multimodal
    api_key = cfg.api_key.get_secret_value() if cfg.api_key is not None else None
    if not api_key or not cfg.base_url:
        raise LLMNotConfiguredError(
            missing_config_error("Multimodal LLM api_key and base_url", "multimodal")
        )
    _multimodal_client = build_client(
        LLMConfig(
            model=cfg.model,
            api_key=api_key,
            base_url=cfg.base_url,
        )
    )
    logger.info("multimodal_llm_client_built", model=cfg.model)
    return _multimodal_client
