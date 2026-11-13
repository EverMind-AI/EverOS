"""Build :class:`OpenAIProvider` instances from settings.

Every place the application calls an LLM is a task
(:class:`~everos.config.LLMTask`) whose client is ``[llm]`` with that task's
``[llm.<task>]`` section laid over it. :func:`resolve_llm_task` is the one
place that merge happens: each field is taken from the first section that sets
it -- the task's own section, then (decider only) the deprecated ``[decider]``
fields, then the task's own defaults (:data:`~everos.config.LLM_TASK_DEFAULTS`),
then ``[llm]`` -- and the section that supplied it is recorded, for the
startup log and ``everos config show``. ``extra`` is one value, never merged
key by key. :func:`llm_client_config` checks the result is usable, raising
:class:`LLMNotConfiguredError` when the key or base URL is missing, and
:func:`build_llm_task_providers` builds one provider per distinct
configuration. Multimodal parsing has its own section, read by
:func:`resolve_multimodal_config`.

With ``[observability] enabled`` each provider reports its completions to a
:class:`TracingObserver` (see :func:`observer_for`); otherwise to a
:class:`NullObserver`.
"""

from __future__ import annotations

__all__ = [
    "LLMClientConfig",
    "LLMNotConfiguredError",
    "LLMTaskResolution",
    "build_llm_provider",
    "build_llm_task_providers",
    "llm_client_config",
    "observer_for",
    "resolve_llm_task",
    "resolve_llm_tasks",
    "resolve_multimodal_config",
]

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final
from urllib.parse import urlsplit

from pydantic import BaseModel

from everos.config import (
    LLM_TASK_DEFAULTS,
    DeciderSettings,
    LLMClientSettings,
    LLMTask,
    Settings,
    missing_config_error,
)
from everos.core.errors import LLMNotConfiguredError

from ._tracing_observer import TracingObserver
from .observer import LLMCallObserver, NullObserver
from .openai_provider import OpenAIProvider, ResponseFormatMode

_DEFAULT_SOURCE: Final = "default"
_LEGACY_DECIDER_SOURCE: Final = "decider"

_DEDICATED_TASKS: Final = frozenset({LLMTask.DECIDER})
"""Tasks that always get a provider of their own, even when their resolved
configuration matches another task's.

The decider runs inside a search request with its own SDK retry count; a
provider fixes that count at construction, so sharing one would hand the
decider's retry budget to extraction or the reverse."""

_LEGACY_DECIDER_FIELDS: Final = tuple(
    name
    for name in LLMClientSettings.model_fields
    if name in DeciderSettings.model_fields
)
"""The LLM fields the deprecated ``[decider]`` section still carries."""


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
        temperature: Sent when a call names none; ``None`` leaves it out.
        max_tokens: Sent when a call names none; ``None`` leaves it out.
    """

    model: str
    api_key: str
    base_url: str
    timeout: float = 60.0
    max_retries: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    response_format_mode: ResponseFormatMode = "json_schema"
    invalid_reply_retries: int = 1
    temperature: float | None = None
    max_tokens: int | None = None


@dataclass(frozen=True)
class LLMTaskResolution:
    """One task's effective LLM settings and the section each field came from.

    Attributes:
        task: The task resolved.
        settings: Every field's effective value.
        sources: Field name -> the section that supplied it: ``"llm.<task>"``,
            ``"decider"`` (deprecated section, decider only), ``"llm"`` or
            ``"default"``.
    """

    task: LLMTask
    settings: LLMClientSettings
    sources: Mapping[str, str]

    @property
    def host(self) -> str | None:
        """The endpoint's host name; ``None`` when no base URL is set."""
        if not self.settings.base_url:
            return None
        return urlsplit(self.settings.base_url).hostname

    @property
    def deprecated_fields(self) -> tuple[str, ...]:
        """The fields taken from the deprecated ``[decider]`` section."""
        return tuple(
            name
            for name, source in self.sources.items()
            if source == _LEGACY_DECIDER_SOURCE
        )


def resolve_llm_task(settings: Settings, task: LLMTask) -> LLMTaskResolution:
    """Merge ``[llm]`` and ``task``'s sections into the task's effective settings.

    No credential check: a resolution of an unconfigured deployment is still
    worth showing. :func:`llm_client_config` checks.
    """
    layers = _layers(settings, task)
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for name in LLMClientSettings.model_fields:
        sources[name], values[name] = next(
            (source, layer[name]) for source, layer in layers if name in layer
        )
    return LLMTaskResolution(
        task=task,
        settings=LLMClientSettings(**values),
        sources=MappingProxyType(sources),
    )


def resolve_llm_tasks(settings: Settings) -> dict[LLMTask, LLMTaskResolution]:
    """:func:`resolve_llm_task` for every task, in :class:`LLMTask` order."""
    return {task: resolve_llm_task(settings, task) for task in LLMTask}


def _layers(settings: Settings, task: LLMTask) -> list[tuple[str, Mapping[str, Any]]]:
    """``task``'s field sources, highest priority first; the last sets every field."""
    base = settings.llm
    names = LLMClientSettings.model_fields
    everything = {name: getattr(base, name) for name in names}
    layers: list[tuple[str, Mapping[str, Any]]] = [
        (f"llm.{task}", _values_set(getattr(base, task.value), names))
    ]
    if task is LLMTask.DECIDER:
        layers.append(
            (
                _LEGACY_DECIDER_SOURCE,
                _values_set(settings.decider, _LEGACY_DECIDER_FIELDS),
            )
        )
    layers += [
        (_DEFAULT_SOURCE, LLM_TASK_DEFAULTS.get(task, {})),
        ("llm", {k: v for k, v in everything.items() if k in base.model_fields_set}),
        (_DEFAULT_SOURCE, everything),
    ]
    return layers


def _values_set(section: BaseModel, names: Iterable[str]) -> dict[str, Any]:
    """The fields among ``names`` that ``section`` sets; ``None`` means unset."""
    values = {name: getattr(section, name) for name in names}
    return {name: value for name, value in values.items() if value is not None}


def llm_client_config(resolution: LLMTaskResolution) -> LLMClientConfig:
    """Turn a resolution into a buildable configuration.

    Raises:
        LLMNotConfiguredError: The resolved key or base URL is empty; the
            message names the task and the ``[llm]`` section.
    """
    cfg = resolution.settings
    api_key = cfg.api_key.get_secret_value() if cfg.api_key is not None else ""
    if not api_key or not cfg.base_url:
        raise LLMNotConfiguredError(
            missing_config_error(
                f"LLM api_key and base_url (task {resolution.task})", "llm"
            )
        )
    return LLMClientConfig(
        model=cfg.model,
        api_key=api_key,
        base_url=cfg.base_url,
        timeout=cfg.timeout_seconds,
        max_retries=cfg.sdk_max_retries,
        extra=dict(cfg.extra),
        response_format_mode=cfg.response_format_mode,
        invalid_reply_retries=cfg.invalid_reply_retries,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


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
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        max_retries=config.max_retries,
        extra=config.extra,
        response_format_mode=config.response_format_mode,
        invalid_reply_retries=config.invalid_reply_retries,
        observer=observer,
    )


def build_llm_task_providers(
    configs: Mapping[LLMTask, LLMClientConfig],
    *,
    observer: LLMCallObserver | None = None,
) -> dict[LLMTask, OpenAIProvider]:
    """One provider per task; tasks with equal configurations share one.

    Sharing keeps a single-model deployment on one connection pool, as before
    tasks existed. The decider always gets its own (see ``_DEDICATED_TASKS``).

    Args:
        configs: Each task's buildable configuration.
        observer: Receives every provider's completions.

    Returns:
        The provider for each task in ``configs``.
    """
    shared: list[tuple[LLMClientConfig, OpenAIProvider]] = []
    providers: dict[LLMTask, OpenAIProvider] = {}
    for task, config in configs.items():
        if task in _DEDICATED_TASKS:
            providers[task] = build_llm_provider(config, observer=observer)
            continue
        provider = next((p for c, p in shared if c == config), None)
        if provider is None:
            provider = build_llm_provider(config, observer=observer)
            shared.append((config, provider))
        providers[task] = provider
    return providers


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
