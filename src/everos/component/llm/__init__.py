"""EverOS's implementation of the everalgo ``LLMClient`` contract.

The contract and its types (``LLMClient``, ``LLMMessage``, ``LLMOutputError``,
``StrictOutputModel``) belong to everalgo; import them from ``everalgo.llm``.
This package exports only what EverOS itself defines:

- :class:`OpenAIProvider` -- the one ``LLMClient`` implementation, over any
  OpenAI-compatible Chat Completions endpoint.
- :class:`LLMCallError` -- what a failed call raises, with a ``reason``.
- :class:`LLMCallObserver` / :class:`TokenUsage` / :class:`NullObserver` --
  the hook that sees each completion's usage and finish reason.
- :class:`LLMClientConfig` / :func:`build_llm_provider` -- the factory.
- :func:`resolve_llm_config` / :func:`resolve_decider_config` /
  :func:`resolve_multimodal_config` / :func:`observer_for` -- what the
  application bootstrap feeds the factory, one per LLM role; a role whose
  credentials are missing raises :class:`LLMNotConfiguredError`.

There is no process-wide accessor: the application bootstrap builds each
provider once and injects it.

External usage::

    from everos.component.llm import (
        build_llm_provider, observer_for, resolve_llm_config,
    )

    llm = build_llm_provider(
        resolve_llm_config(settings), observer=observer_for(settings)
    )
"""

from .errors import LLMCallError as LLMCallError
from .factory import DeciderClientConfig as DeciderClientConfig
from .factory import LLMClientConfig as LLMClientConfig
from .factory import LLMNotConfiguredError as LLMNotConfiguredError
from .factory import build_llm_provider as build_llm_provider
from .factory import observer_for as observer_for
from .factory import resolve_decider_config as resolve_decider_config
from .factory import resolve_llm_config as resolve_llm_config
from .factory import resolve_multimodal_config as resolve_multimodal_config
from .observer import LLMCallObserver as LLMCallObserver
from .observer import NullObserver as NullObserver
from .observer import TokenUsage as TokenUsage
from .openai_provider import OpenAIProvider as OpenAIProvider

__all__ = [
    "DeciderClientConfig",
    "LLMCallError",
    "LLMCallObserver",
    "LLMClientConfig",
    "LLMNotConfiguredError",
    "NullObserver",
    "OpenAIProvider",
    "TokenUsage",
    "build_llm_provider",
    "observer_for",
    "resolve_decider_config",
    "resolve_llm_config",
    "resolve_multimodal_config",
]
