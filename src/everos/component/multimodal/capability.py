"""MultimodalLLMCapability — soft-dependency wrapper for multimodal LLM client.

Parallel structure to :class:`everos.component.rerank.RerankCapability`, but
for the multimodal LLM used by ``everalgo.parser``. A caller either requires
multimodal (raising :class:`ProviderNotConfiguredError` -> HTTP 422 when
missing) or checks ``available`` to skip parsing when unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from everos.core.errors import ProviderNotConfiguredError

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


@dataclass(frozen=True)
class MultimodalLLMCapability:
    """Wraps an optional multimodal LLMClient with a hard-require API."""

    provider: LLMClient | None

    @property
    def available(self) -> bool:
        """True iff a provider was successfully constructed."""
        return self.provider is not None

    def require(self) -> LLMClient:
        """Return the provider, or raise :class:`ProviderNotConfiguredError`.

        Caller supplies its own ``feature``/``alternative_hint`` context by
        catching and re-raising if needed; this base call only identifies
        the missing provider as ``"multimodal_llm"``.
        """
        if self.provider is None:
            raise ProviderNotConfiguredError(provider="multimodal_llm")
        return self.provider
