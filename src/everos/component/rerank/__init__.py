"""Rerank provider adapters (one provider per file).

Public surface:

- :class:`RerankProvider` — Protocol every provider satisfies.
- :class:`RerankResult` / :class:`RerankError` — value type + error.
- :class:`DeepInfraRerankProvider` — DeepInfra inference-API rerank.
- :class:`VllmRerankProvider` — OpenAI-compat ``/v1/rerank`` (vLLM,
  self-hosted, other compatible servers).
- :func:`build_rerank_provider` — settings-driven factory that picks
  the concrete provider via ``settings.rerank.provider``.

External usage::

    from everos.component.rerank import build_rerank_provider
    provider = build_rerank_provider(settings.rerank)
    scored = await provider.rerank("how to file a claim", documents)
"""

from .deepinfra_provider import DeepInfraRerankProvider as DeepInfraRerankProvider
from .factory import build_rerank_provider as build_rerank_provider
from .protocol import RerankError as RerankError
from .protocol import RerankProvider as RerankProvider
from .protocol import RerankResult as RerankResult
from .vllm_provider import VllmRerankProvider as VllmRerankProvider

__all__ = [
    "DeepInfraRerankProvider",
    "RerankError",
    "RerankProvider",
    "RerankResult",
    "VllmRerankProvider",
    "build_rerank_provider",
]
