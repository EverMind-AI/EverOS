"""Rerank provider protocol.

The contract every rerank provider satisfies: given a query and a list
of candidate documents, return a re-ordered list of ``(index, score)``
pairs (highest relevance first). The provider does **not** filter —
that's the caller's job (e.g. drop scores below a threshold, take
``top_k``). Returning every input pair keeps the contract stable
across providers whose backends may not natively support ``top_n``.

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, Protocol, runtime_checkable

from everos.core.errors import RerankServiceError as RerankServiceError

# Backward compat alias.
RerankError = RerankServiceError


class RerankResult(NamedTuple):
    """One scored document from a rerank call.

    ``index`` is the position of the document in the *input* list (so
    callers can map back to the original document text). ``score`` is
    provider-defined; higher = more relevant.
    """

    index: int
    score: float


@runtime_checkable
class RerankProvider(Protocol):
    """Async rerank provider contract."""

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str | None = None,
    ) -> list[RerankResult]:
        """Score and re-order ``documents`` against ``query``.

        Args:
            query: The search query.
            documents: Passage texts to score against ``query``.
            instruction: Task instruction for instruction-tuned rerankers
                (e.g. Qwen3-Reranker). Providers that wrap the model's chat
                template fold this into the prompt; providers backed by a
                dedicated rerank endpoint that handles templating server-side
                may ignore it. ``None`` defers to the provider's default.

        Returns:
            One :class:`RerankResult` per input document, sorted by
            ``score`` descending. The returned list length equals
            ``len(documents)``.
        """

        ...
