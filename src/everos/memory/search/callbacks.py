"""Callback factories handed to ``everalgo.rank.arank``.

Four callbacks the rank pipeline expects:

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
* :func:`build_case_rerank_fn` — case-shaped variant, mirrors
  :func:`build_skill_rerank_fn` for ``"Agent Case: {task_intent} -
  {approach}"`` passages.
* :func:`build_retrieve_fn` — Round-2 recall callback for ``agentic``.
  Re-runs the sparse + dense recall path for a refined query and fuses
  the two routes with RRF (``k=60``) before handing back to the agentic
  loop.

``_format_skill_passage_from_metadata`` / ``_format_case_passage_from_metadata``
take the raw ``Candidate.metadata`` dict rather than a ``Candidate`` so the
metadata bridge in ``agentic_agent.py`` (which formats a passage before a
``Candidate`` wrapping it exists) can reuse the exact same formatting logic
the rerank step uses — one implementation, no drift between what the LLM
sufficiency check sees and what the reranker sees.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING

from everalgo.rank.fusion import rrf
from everalgo.rank.protocols import RerankFn, RetrieveFn
from everalgo.types import Candidate

from everos.component.rerank import RerankProvider
from everos.core.observability.tracing import memory_span

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
        with memory_span(
            "everos.search.rank",
            observation_type="span",
            metadata={"phase": "cross_encoder"},
        ):
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


def _format_skill_passage_from_metadata(meta: dict[str, object]) -> str:
    """``"Agent Skill: {name}"`` + ``" - {description}"`` when present.
    Mirrors opensource ``extract_text_from_hit`` for AGENT_SKILL.
    """
    name = str(meta.get("name", "") or "")
    description = str(meta.get("description", "") or "")
    if not name:
        return description
    if description:
        return f"Agent Skill: {name} - {description}"
    return f"Agent Skill: {name}"


def _format_skill_passage(candidate: Candidate) -> str:
    """``Candidate``-shaped wrapper over :func:`_format_skill_passage_from_metadata`."""
    return _format_skill_passage_from_metadata(candidate.metadata)


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
        with memory_span(
            "everos.search.rank",
            observation_type="span",
            metadata={"phase": "cross_encoder_skill"},
        ):
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


# Mirrors _SKILL_RERANK_INSTRUCTION: biases the reranker toward methodology /
# domain match for agent cases rather than generic Q-A relevance.
_CASE_RERANK_INSTRUCTION = (
    "Determine whether the case's task and approach are applicable to the "
    "query, preferring same-domain cases with directly relevant methodology."
)


def _format_case_passage_from_metadata(meta: dict[str, object]) -> str:
    """``"Agent Case: {task_intent}"`` + ``" - {approach}"`` when present.

    Mirrors ``_format_skill_passage_from_metadata``. Falls back to
    ``task_intent`` alone when ``approach`` is empty (which is legal per the
    cascade handler — no non-empty guard on ``approach``).
    """
    task_intent = str(meta.get("task_intent", "") or "")
    approach = str(meta.get("approach", "") or "")
    if not task_intent:
        return approach
    if approach:
        return f"Agent Case: {task_intent} - {approach}"
    return f"Agent Case: {task_intent}"


def _format_case_passage(candidate: Candidate) -> str:
    """``Candidate``-shaped wrapper over :func:`_format_case_passage_from_metadata`."""
    return _format_case_passage_from_metadata(candidate.metadata)


def build_case_rerank_fn(provider: RerankProvider) -> RerankFn:
    """Case-shaped ``RerankFn``: multi-field passage + :data:`_CASE_RERANK_INSTRUCTION`.
    Mirrors :func:`build_skill_rerank_fn`.
    """

    async def _rerank(
        query: str,
        candidates: Sequence[Candidate],
    ) -> list[Candidate]:
        items = list(candidates)
        if not items:
            return []
        passages = [_format_case_passage(c) for c in items]
        with memory_span(
            "everos.search.rank",
            observation_type="span",
            metadata={"phase": "cross_encoder_case"},
        ):
            results = await provider.rerank(
                query, passages, instruction=_CASE_RERANK_INSTRUCTION
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
