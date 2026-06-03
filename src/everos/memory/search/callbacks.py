"""Callback factories handed to ``everalgo.rank.arank``.

Three callbacks the rank pipeline expects:

* :func:`build_rerank_fn` — cross-encoder scorer used by ``agentic``
  Round-1 + final rerank, and by ``rrf`` / ``lr`` when LLM rerank is
  enabled. Pulls the display text out of ``Candidate.metadata`` and
  drives the configured :class:`RerankProvider`. Returns all reranked
  candidates; the caller is responsible for truncation.
* :func:`build_skill_rerank_fn` — skill-shaped variant: composes a
  ``"Agent Skill: {name} - {description}"`` passage (the multi-field
  shape doesn't fit the single-``text_field`` contract above) and uses
  a skill-specific instruction. Mirrors memsys_opensource
  ``_rerank_skill_items``.
* :func:`build_retrieve_fn` — Round-2 recall callback for ``agentic``.
  Re-runs the sparse + dense recall path for a refined query and fuses
  the two routes with RRF (``k=60``) before handing back to the agentic
  loop.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from everalgo.rank.fusion import rrf
from everalgo.rank.protocols import RerankFn, RetrieveFn
from everalgo.types import Candidate

from everos.component.rerank import RerankProvider

if TYPE_CHECKING:
    from .recall import KindRecaller


def build_rerank_fn(
    provider: RerankProvider,
    *,
    text_field: str,
    instruction: str | None = None,
) -> RerankFn:
    """Build an everalgo ``RerankFn`` over the configured rerank provider.

    Returns a 2-arg ``(query, candidates) -> list[Candidate]`` async callable
    matching ``everalgo.rank.protocols.RerankFn``. All reranked candidates are
    returned without truncation — the caller (``aagentic_retrieve``) is
    responsible for slicing via ``round1_rerank_top_n``.

    ``text_field`` decides which ``Candidate.metadata`` key carries the
    passage text — ``"episode"`` for episodes, ``"task_intent"`` for cases.
    Missing fields fall back to the empty string so the rerank call never
    throws on a malformed row.

    ``instruction`` is the task instruction for instruction-tuned rerankers
    (e.g. Qwen3-Reranker); it is forwarded to the provider verbatim. ``None``
    defers to the provider's default instruction.
    """

    async def _rerank(
        query: str,
        candidates: Sequence[Candidate],
    ) -> list[Candidate]:
        items = list(candidates)
        if not items:
            return []
        passages = [str(c.metadata.get(text_field, "")) for c in items]
        results = await provider.rerank(query, passages, instruction=instruction)
        out: list[Candidate] = []
        for r in results:
            if not 0 <= r.index < len(items):
                continue
            out.append(items[r.index].model_copy(update={"score": float(r.score)}))
        return out

    return _rerank


# Biases the reranker toward methodology / domain match rather than
# generic Q-A relevance (memsys_opensource ``_rerank_skill_items``).
_SKILL_RERANK_INSTRUCTION = (
    "Determine whether the skill's methodology and domain "
    "are applicable to the query, preferring same-domain "
    "skills with directly relevant steps."
)


def _format_skill_passage(candidate: Candidate) -> str:
    """``"Agent Skill: {name}"`` + ``" - {description}"`` when present.
    Mirrors opensource ``extract_text_from_hit`` for AGENT_SKILL.
    """
    meta = candidate.metadata
    name = str(meta.get("name", "") or "")
    description = str(meta.get("description", "") or "")
    if not name:
        return description
    if description:
        return f"Agent Skill: {name} - {description}"
    return f"Agent Skill: {name}"


def build_skill_rerank_fn(provider: RerankProvider) -> RerankFn:
    """Skill-shaped ``RerankFn``: multi-field passage +
    :data:`_SKILL_RERANK_INSTRUCTION`. Output stays score-comparable
    with the memsys_opensource ``_rerank_skill_items`` baseline.
    """

    async def _rerank(
        query: str,
        candidates: Sequence[Candidate],
    ) -> list[Candidate]:
        items = list(candidates)
        if not items:
            return []
        passages = [_format_skill_passage(c) for c in items]
        results = await provider.rerank(
            query, passages, instruction=_SKILL_RERANK_INSTRUCTION
        )
        out: list[Candidate] = []
        for r in results:
            if not 0 <= r.index < len(items):
                continue
            out.append(items[r.index].model_copy(update={"score": float(r.score)}))
        return out

    return _rerank


def build_retrieve_fn(
    recaller: KindRecaller,
    *,
    where: str,
    embed_query_fn: Callable[[str], Awaitable[list[float]]],
    rrf_k: int = 60,
) -> RetrieveFn:
    """Build an everalgo ``RetrieveFn`` that fuses fresh sparse + dense recall.

    ``embed_query_fn`` is an async ``(str) -> list[float]`` that produces
    a 1024-d vector for an arbitrary query — typically the project's
    :class:`EmbeddingProvider.embed`. We re-embed the refined queries
    that the agentic loop emits in Round 2.
    """

    async def _retrieve(query: str, top_n: int) -> list[Candidate]:
        recall_limit = top_n * 5
        vector = await embed_query_fn(query)
        sparse = await recaller.sparse_recall(query, where, limit=recall_limit)
        dense = (
            await recaller.dense_recall(vector, where, limit=recall_limit)
            if vector
            else []
        )
        if not sparse and not dense:
            return []
        fused = rrf(dense, sparse, k=rrf_k)
        return fused[:top_n]

    return _retrieve
