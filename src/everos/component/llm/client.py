"""Process-wide LLM client accessor.

Lazy singleton — first call reads settings and builds the algo LLM
client; subsequent calls return the cached instance. Raises
:class:`LLMNotConfiguredError` when no credentials are present so
misconfiguration surfaces at app startup (via the LLM lifespan
provider) instead of silently failing per-request downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from everalgo.llm import build_client
from everalgo.llm.config import LLMConfig
from everalgo.llm.protocols import LLMClient
from everalgo.llm.types import ChatMessage, ChatResponse
from pydantic import BaseModel

from everos.component.utils.config_hints import missing_config_error
from everos.config import Settings, load_settings
from everos.core.observability.logging import get_logger

from ._usage_client import UsageRecordingClient
from .openai_provider import OpenAIProvider

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
_decider_client: LLMClient | None = None


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
            timeout=llm_cfg.timeout_seconds,
            extra=dict(llm_cfg.extra),
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


@dataclass(frozen=True)
class DeciderClientConfig:
    """What the decider actually talks to, after ``[decider]`` -> ``[llm]`` inheritance.

    One resolution, two consumers: :func:`get_decider_llm_client` builds the client
    from it, and a caller that wants to *check* the decider before using it (the
    benchmark harness probes it before the first question) must ask the same question
    the same way. Re-deriving the rule at the check site is how a config that runs fine
    -- ``[decider].model`` set, endpoint inherited from ``[llm]`` -- ended up skipping
    the probe entirely, which is the one case the probe exists for.
    """

    model: str
    api_key: str
    base_url: str
    timeout: float
    max_retries: int | None
    extra: dict[str, Any] = field(default_factory=dict)
    inherits_model: bool = False
    """True when ``[decider]`` names no model and the main ``[llm]`` model decides."""


def resolve_decider_config(settings: Settings | None = None) -> DeciderClientConfig:
    """Resolve the decider's effective client settings.

    ``model``, ``api_key`` and ``base_url`` each fall back to ``[llm]`` -- so pointing
    the decider at another hosted model needs only ``[decider].model``, and a config
    that predates the section keeps the main model as its decider. ``timeout`` and
    ``max_retries`` always come from ``[decider]``: they bound one round inside a live
    search, which is a different deadline from a background extraction's.

    Raises:
        LLMNotConfiguredError: When neither section yields a usable key and base URL.
    """
    settings = settings or load_settings()
    cfg, llm_cfg = settings.decider, settings.llm
    api_key = cfg.api_key.get_secret_value() if cfg.api_key is not None else None
    api_key = api_key or (
        llm_cfg.api_key.get_secret_value() if llm_cfg.api_key is not None else None
    )
    base_url = cfg.base_url or llm_cfg.base_url
    model = cfg.model or llm_cfg.model
    if not api_key or not base_url:
        raise LLMNotConfiguredError(
            missing_config_error("decider api_key and base_url", "decider")
        )
    # `extra` is endpoint vocabulary, so it travels with whichever section supplied the
    # endpoint: a decider on its own gateway must not inherit a field that gateway
    # rejects, and a decider inheriting `[llm]` must not lose the field that endpoint
    # requires -- dropping `[llm].extra` there would silently re-enable thinking on a
    # Qwen deployment that configured it off once, for both roles.
    return DeciderClientConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=cfg.timeout_seconds,
        max_retries=cfg.sdk_max_retries,
        extra=dict(cfg.extra if cfg.model else llm_cfg.extra),
        inherits_model=not cfg.model,
    )


def get_decider_llm_client() -> LLMClient:
    """Return the singleton retrieval-decider client.

    Built from :func:`resolve_decider_config`, and always its own instance -- even when
    every field is inherited from ``[llm]``. Sharing the extraction client looks
    equivalent and is not: the SDK's own ``max_retries`` is fixed at construction, so a
    shared client cannot have the decider's ``sdk_max_retries`` applied to it, and the
    SDK retries then multiply with the decider's own. Four decider attempts times three
    SDK requests against a 60s deadline is the 728s round that this exists to stop.

    Raises:
        LLMNotConfiguredError: When neither ``[decider]`` nor ``[llm]`` yields a usable
            ``api_key`` and ``base_url``.
    """
    global _decider_client
    if _decider_client is not None:
        return _decider_client
    settings = load_settings()
    cfg = resolve_decider_config(settings)
    # everalgo's `LLMConfig` has no `max_retries` field, so `build_client` cannot carry
    # one; everos' own provider can, and satisfies the same `LLMClient` protocol with
    # the same response type.
    client: LLMClient = OpenAIProvider(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
        extra=cfg.extra,
    )
    if settings.observability.enabled:
        client = UsageRecordingClient(client)
    _decider_client = _LoggingLLMClient(client)
    logger.info(
        "decider_client_built",
        model=cfg.model,
        base_url=cfg.base_url,
        inherits_model=cfg.inherits_model,
        sdk_max_retries=cfg.max_retries,
    )
    return _decider_client


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
