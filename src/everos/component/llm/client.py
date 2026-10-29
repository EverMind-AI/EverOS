"""Process-wide LLM client accessors.

Three lazy singletons, one per LLM role: extraction (``[llm]``), the
multi-round retrieval decider (``[decider]``, falling back to ``[llm]``) and
multimodal parsing (``[multimodal]``). Each resolves its section into an
:class:`LLMClientConfig` and builds the provider through
:func:`build_llm_provider`, so all three are the same :class:`OpenAIProvider`.
A missing key or base URL raises :class:`LLMNotConfiguredError`, so
misconfiguration surfaces at startup (via the LLM lifespan provider) instead
of per request.

With ``[observability] enabled`` each provider reports its completions to a
:class:`TracingObserver`; otherwise to a :class:`NullObserver`.
"""

from __future__ import annotations

__all__ = [
    "DeciderClientConfig",
    "LLMNotConfiguredError",
    "get_decider_llm_client",
    "get_llm_client",
    "get_multimodal_llm_client",
    "resolve_decider_config",
]

from dataclasses import dataclass

from everalgo.llm import LLMClient

from everos.component.utils.config_hints import missing_config_error
from everos.config import Settings, load_settings
from everos.core.observability.logging import get_logger

from ._tracing_observer import TracingObserver
from .factory import LLMClientConfig, build_llm_provider
from .observer import LLMCallObserver, NullObserver

logger = get_logger(__name__)


class LLMNotConfiguredError(RuntimeError):
    """Raised when an LLM section is missing ``api_key`` or ``base_url``."""


_llm_client: LLMClient | None = None
_multimodal_client: LLMClient | None = None
_decider_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return the singleton extraction client.

    Raises:
        LLMNotConfiguredError: When ``settings.llm.api_key`` or
            ``settings.llm.base_url`` is unset.
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    settings = load_settings()
    config = _resolve_llm_config(settings)
    _llm_client = build_llm_provider(config, observer=_observer_for(settings))
    logger.info("llm_client_built", model=config.model)
    return _llm_client


@dataclass(frozen=True)
class DeciderClientConfig(LLMClientConfig):
    """What the decider actually talks to, after ``[decider]`` -> ``[llm]`` inheritance.

    One resolution, two consumers: :func:`get_decider_llm_client` builds the client
    from it, and a caller that wants to *check* the decider before using it (the
    benchmark harness probes it before the first question) must ask the same question
    the same way. Re-deriving the rule at the check site is how a config that runs fine
    -- ``[decider].model`` set, endpoint inherited from ``[llm]`` -- ended up skipping
    the probe entirely, which is the one case the probe exists for.

    Attributes:
        inherits_model: True when ``[decider]`` names no model and the main
            ``[llm]`` model decides.
    """

    inherits_model: bool = False


def resolve_decider_config(settings: Settings | None = None) -> DeciderClientConfig:
    """Resolve the decider's effective client settings.

    ``model``, ``api_key`` and ``base_url`` each fall back to ``[llm]`` -- so pointing
    the decider at another hosted model needs only ``[decider].model``, and a config
    that predates the section keeps the main model as its decider. ``timeout``,
    ``max_retries`` and ``invalid_reply_retries`` always come from ``[decider]``: they
    bound one round inside a live search, which is a different deadline from a
    background extraction's.

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
    # `extra` is endpoint vocabulary and `response_format_mode` a model capability,
    # so both travel with whichever section supplied the model: a decider on its own
    # gateway must not inherit a field that gateway rejects, and a decider inheriting
    # `[llm]` must not lose the field that endpoint requires -- dropping `[llm].extra`
    # there would silently re-enable thinking on a Qwen deployment that configured it
    # off once, for both roles.
    model_section = cfg if cfg.model else llm_cfg
    return DeciderClientConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout=cfg.timeout_seconds,
        max_retries=cfg.sdk_max_retries,
        extra=dict(model_section.extra),
        response_format_mode=model_section.response_format_mode,
        invalid_reply_retries=cfg.invalid_reply_retries,
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
    config = resolve_decider_config(settings)
    _decider_client = build_llm_provider(config, observer=_observer_for(settings))
    logger.info(
        "decider_client_built",
        model=config.model,
        base_url=config.base_url,
        inherits_model=config.inherits_model,
        sdk_max_retries=config.max_retries,
    )
    return _decider_client


def get_multimodal_llm_client() -> LLMClient:
    """Return the singleton multimodal LLM client (for everalgo.parser).

    Reads the flat ``[multimodal]`` config -- kept separate from the main
    ``[llm]`` so parsing can target a vision/audio-capable endpoint.

    Raises:
        LLMNotConfiguredError: When ``settings.multimodal.api_key`` or
            ``settings.multimodal.base_url`` is unset.
    """
    global _multimodal_client
    if _multimodal_client is not None:
        return _multimodal_client
    settings = load_settings()
    config = _resolve_multimodal_config(settings)
    _multimodal_client = build_llm_provider(config, observer=_observer_for(settings))
    logger.info("multimodal_llm_client_built", model=config.model)
    return _multimodal_client


def _resolve_llm_config(settings: Settings) -> LLMClientConfig:
    """Resolve ``[llm]``; raise :class:`LLMNotConfiguredError` if unusable."""
    cfg = settings.llm
    api_key = cfg.api_key.get_secret_value() if cfg.api_key is not None else None
    if not api_key or not cfg.base_url:
        raise LLMNotConfiguredError(
            missing_config_error("LLM api_key and base_url", "llm")
        )
    return LLMClientConfig(
        model=cfg.model,
        api_key=api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout_seconds,
        extra=dict(cfg.extra),
        response_format_mode=cfg.response_format_mode,
        invalid_reply_retries=cfg.invalid_reply_retries,
    )


def _resolve_multimodal_config(settings: Settings) -> LLMClientConfig:
    """Resolve ``[multimodal]``; raise :class:`LLMNotConfiguredError` if unusable."""
    cfg = settings.multimodal
    api_key = cfg.api_key.get_secret_value() if cfg.api_key is not None else None
    if not api_key or not cfg.base_url:
        raise LLMNotConfiguredError(
            missing_config_error("Multimodal LLM api_key and base_url", "multimodal")
        )
    return LLMClientConfig(
        model=cfg.model,
        api_key=api_key,
        base_url=cfg.base_url,
        response_format_mode=cfg.response_format_mode,
        invalid_reply_retries=cfg.invalid_reply_retries,
    )


def _observer_for(settings: Settings) -> LLMCallObserver:
    return TracingObserver() if settings.observability.enabled else NullObserver()
