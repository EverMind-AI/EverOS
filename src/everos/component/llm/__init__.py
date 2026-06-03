"""LLM provider adapters (one provider per file, mem0-style).

Public surface:

- :class:`LLMClient` — Protocol every provider satisfies (re-exported
  from :mod:`everalgo.llm`; same shape so everos providers can be
  handed to everalgo operators).
- :class:`ChatMessage` / :class:`ChatResponse` / :class:`Usage` — chat
  payload types (re-exported from :mod:`everalgo.llm`).
- :class:`LLMError` — provider-side failure (re-exported).
- :class:`LLMNotConfiguredError` — raised when credentials are missing.
- :class:`OpenAIProvider` — concrete provider wrapping
  ``openai.AsyncOpenAI`` against any OpenAI-compatible endpoint.
- :func:`build_llm_provider` — settings-driven factory.
- :func:`get_llm_client` — process-wide lazy singleton accessor.

External usage::

    from everos.component.llm import build_llm_provider, LLMClient
    provider = build_llm_provider(settings.llm)
"""

from .client import LLMNotConfiguredError as LLMNotConfiguredError
from .client import get_llm_client as get_llm_client
from .client import get_multimodal_llm_client as get_multimodal_llm_client
from .factory import build_llm_provider as build_llm_provider
from .openai_provider import OpenAIProvider as OpenAIProvider
from .protocol import ChatMessage as ChatMessage
from .protocol import ChatResponse as ChatResponse
from .protocol import LLMClient as LLMClient
from .protocol import LLMError as LLMError
from .protocol import Usage as Usage

__all__ = [
    "ChatMessage",
    "ChatResponse",
    "LLMClient",
    "LLMError",
    "LLMNotConfiguredError",
    "OpenAIProvider",
    "Usage",
    "build_llm_provider",
    "get_llm_client",
    "get_multimodal_llm_client",
]
