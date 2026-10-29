"""Build :class:`OpenAIProvider` instances from settings, one per LLM role.

Three roles share one construction path: extraction (``[llm]``), the
multi-round retrieval decider (``[decider]``, falling back to ``[llm]``) and
multimodal parsing (``[multimodal]``). ``resolve_*_config`` turns a section
into an :class:`LLMClientConfig` -- raising :class:`LLMNotConfiguredError`
when the key or base URL is missing -- and :func:`build_llm_provider` turns
that into a provider. The application bootstrap decides what a missing role
means.

With ``[observability] enabled`` each provider reports its completions to a
:class:`TracingObserver` (see :func:`observer_for`); otherwise to a
:class:`NullObserver`.
"""

from __future__ import annotations

__all__ = [
    "DeciderClientConfig",
    "LLMClientConfig",
    "LLMNotConfiguredError",
    "build_llm_provider",
    "observer_for",
    "resolve_decider_config",
    "resolve_llm_config",
    "resolve_multimodal_config",
]

from dataclasses import dataclass, field
from typing import Any

from everos.config import Settings, missing_config_error
from everos.core.errors import LLMNotConfiguredError

from ._tracing_observer import TracingObserver
from .observer import LLMCallObserver, NullObserver
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


@dataclass(frozen=True)
class DeciderClientConfig(LLMClientConfig):
    """What the decider actually talks to, after ``[decider]`` -> ``[llm]`` inheritance.

    One resolution, two consumers: the application bootstrap builds the client
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


def resolve_decider_config(settings: Settings) -> DeciderClientConfig:
    """Resolve the decider's effective client settings.

    ``model``, ``api_key`` and ``base_url`` each fall back to ``[llm]`` -- so pointing
    the decider at another hosted model needs only ``[decider].model``, and a config
    that predates the section keeps the main model as its decider. ``timeout``,
    ``max_retries`` and ``invalid_reply_retries`` always come from ``[decider]``: they
    bound one round inside a live search, which is a different deadline from a
    background extraction's.

    The decider is always its own provider instance, even when every field is
    inherited from ``[llm]``. Sharing the extraction provider looks equivalent
    and is not: the SDK's own ``max_retries`` is fixed at construction, so a
    shared client cannot have the decider's ``sdk_max_retries`` applied to it,
    and the SDK retries then multiply with the decider's own. Four decider
    attempts times three SDK requests against a 60s deadline is the 728s round
    that this exists to stop.

    Raises:
        LLMNotConfiguredError: When neither section yields a usable key and base URL.
    """
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


def resolve_llm_config(settings: Settings) -> LLMClientConfig:
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


def resolve_multimodal_config(settings: Settings) -> LLMClientConfig:
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


def observer_for(settings: Settings) -> LLMCallObserver:
    """The completion observer every LLM role reports to under ``settings``."""
    return TracingObserver() if settings.observability.enabled else NullObserver()
