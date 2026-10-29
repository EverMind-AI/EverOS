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
- :func:`get_llm_client` / :func:`get_decider_llm_client` /
  :func:`get_multimodal_llm_client` -- process-wide accessors, raising
  :class:`LLMNotConfiguredError` when credentials are missing.

External usage::

    from everalgo.llm import LLMMessage
    from everos.component.llm import get_llm_client

    text = await get_llm_client().achat(
        [LLMMessage(role="user", content=prompt)], response_format=Reply
    )
"""

from .client import LLMNotConfiguredError as LLMNotConfiguredError
from .client import get_decider_llm_client as get_decider_llm_client
from .client import get_llm_client as get_llm_client
from .client import get_multimodal_llm_client as get_multimodal_llm_client
from .errors import LLMCallError as LLMCallError
from .factory import LLMClientConfig as LLMClientConfig
from .factory import build_llm_provider as build_llm_provider
from .observer import LLMCallObserver as LLMCallObserver
from .observer import NullObserver as NullObserver
from .observer import TokenUsage as TokenUsage
from .openai_provider import OpenAIProvider as OpenAIProvider

__all__ = [
    "LLMCallError",
    "LLMCallObserver",
    "LLMClientConfig",
    "LLMNotConfiguredError",
    "NullObserver",
    "OpenAIProvider",
    "TokenUsage",
    "build_llm_provider",
    "get_decider_llm_client",
    "get_llm_client",
    "get_multimodal_llm_client",
]
