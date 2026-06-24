"""Rerank provider adapters (one provider per file).

Public surface:

- :class:`RerankProvider` — Protocol every provider satisfies.
- :class:`RerankResult` / :class:`RerankServiceError` — value type + error.
- :class:`RerankError` — backward-compat alias for :class:`RerankServiceError`.
- :class:`DeepInfraRerankProvider` — DeepInfra inference-API rerank.
- :class:`DashScopeRerankProvider` — Aliyun Bailian / DashScope rerank.
- :class:`VllmRerankProvider` — OpenAI-compat ``/v1/rerank`` (vLLM,
  self-hosted, other compatible servers).
- :func:`build_rerank_provider` — settings-driven factory that picks
  the concrete provider via ``settings.rerank.provider``.

External usage::

    from everos.component.rerank import build_rerank_provider
    provider = build_rerank_provider(settings.rerank)
    scored = await provider.rerank("how to file a claim", documents)
"""

from everos.core.errors import RerankServiceError as RerankServiceError

from .dashscope_provider import DashScopeRerankProvider as DashScopeRerankProvider
from .deepinfra_provider import DeepInfraRerankProvider as DeepInfraRerankProvider
from .factory import build_rerank_provider as build_rerank_provider
from .protocol import RerankError as RerankError
from .protocol import RerankProvider as RerankProvider
from .protocol import RerankResult as RerankResult
from .vllm_provider import VllmRerankProvider as VllmRerankProvider

__all__ = [
    "DashScopeRerankProvider",
    "DeepInfraRerankProvider",
    "RerankError",
    "RerankProvider",
    "RerankResult",
    "RerankServiceError",
    "VllmRerankProvider",
    "build_rerank_provider",
]
