"""Rerank callback factory handed to ``everalgo.rank`` by knowledge search.

* :func:`build_fetched_rerank_fn` — the passage is fetched by candidate id
  (knowledge topics keep their body in the catalog, not in the index) and
  scored by the configured :class:`RerankProvider`. Returns every reranked
  candidate; the caller is responsible for truncation.

Memory search has no cross-encoder stage: multi-round retrieval ranks by RRF
and the decider alone.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from everalgo.rank.protocols import RerankFn
from everalgo.types import Candidate

from everos.component.rerank import RerankProvider
from everos.core.observability.tracing import memory_span


def build_fetched_rerank_fn(
    provider: RerankProvider,
    *,
    passages: Callable[[Sequence[str]], Awaitable[Sequence[str]]],
    instruction: str | None = None,
) -> RerankFn:
    """``RerankFn`` whose passages come from ``passages(candidate_ids)``.

    ``passages`` returns one text per id, in order. Used where the passage is
    not carried by the recall candidate -- knowledge topics keep their body
    in the catalog, so it is read only for the pool actually reranked.
    """

    async def _rerank(
        query: str,
        candidates: Sequence[Candidate],
    ) -> list[Candidate]:
        items = list(candidates)
        if not items:
            return []
        texts = list(await passages([c.id for c in items]))
        with memory_span(
            "everos.search.rank",
            observation_type="span",
            metadata={"phase": "cross_encoder"},
        ):
            results = await provider.rerank(query, texts, instruction=instruction)
        return [
            items[r.index].model_copy(update={"score": float(r.score)})
            for r in results
            if 0 <= r.index < len(items)
        ]

    return _rerank
