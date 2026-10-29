"""Agent-kind AGENTIC search — flat hybrid path (no cluster, no MaxSim).

Implements the flat agentic path for ``agent_case`` and ``agent_skill``
memory kinds.  Unlike the episode AGENTIC path (which uses cluster + MaxSim),
agent memory is retrieved via a plain hybrid (RRF) recall straight into
``aagentic_retrieve``.

Hyperparameters are aligned to the memsys_opensource ``AgenticConfig`` defaults
(``agentic_utils.py``):

* ``_ROUND1_TOP_N = 20``       — ``round1_top_n``
* ``_ROUND2_CAP = 40``         — ``combined_total``
* ``_HYBRID_RRF_K = 60``       — ``rrf_k`` default in ``retrieval_utils.py:347``
* ``_DENSE_CANDIDATES = 50``   — ``round1_emb_top_n``
* ``_SPARSE_CANDIDATES = 50``  — ``round1_bm25_top_n``
* ``_ROUND1_RERANK_TOP_N = 10`` — ``round1_rerank_top_n``
* ``_MULTI_QUERY_COUNT = 3``   — ``num_queries``

"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal

from everalgo.rank.agentic import aagentic_retrieve
from everalgo.rank.hybrid import ahybrid_retrieve
from everalgo.types import Candidate

from everos.core.observability.tracing import memory_span
from everos.memory.ports import Filter, Stored
from everos.memory.search.callbacks import (
    _format_case_passage_from_metadata,
    _format_skill_passage_from_metadata,
    build_case_rerank_fn,
    build_skill_rerank_fn,
)
from everos.memory.search.recall import RecallResult
from everos.memory.search.shaper import shape_agent_cases, shape_agent_skills

from .dto import SearchAgentCaseItem, SearchAgentSkillItem

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

    from everos.component.rerank import RerankProvider
    from everos.memory.search.recall.agent_case import AgentCaseRecaller
    from everos.memory.search.recall.agent_skill import AgentSkillRecaller

# ── Hyperparameters aligned to memsys_opensource AgenticConfig defaults ─────
# Source: memsys_opensource/src/agentic_layer/agentic_utils.py (AgenticConfig)
#         and retrieval_utils.py:347 (rrf_k default).
_DENSE_CANDIDATES: int = 50  # round1_emb_top_n
_SPARSE_CANDIDATES: int = 50  # round1_bm25_top_n
_HYBRID_RRF_K: int = 60  # retrieval_utils.py:347 default rrf_k
_ROUND1_TOP_N: int = 20  # round1_top_n (was 50, aligned to opensource 20)
_ROUND1_RERANK_TOP_N: int = 10  # round1_rerank_top_n
_ROUND2_CAP: int = 40  # combined_total
_MULTI_QUERY_COUNT: int = 3  # num_queries
_REFINEMENT_STRATEGY: str = "multi_query"


_EMPTY_PASSAGE = "(empty)"
"""Stand-in body for a row whose every passage field is blank. Keeps
``everalgo.rank.agentic._format_docs`` from raising on an empty string."""


def _with_doc_metadata(
    candidate: Candidate, *, format_passage: Callable[[dict[str, Any]], str]
) -> Candidate:
    """Bridge an agent candidate to the everalgo ``_format_docs`` contract.

    ``aagentic_retrieve`` renders round-1 candidates into the sufficiency /
    multi-query LLM prompt via ``everalgo.rank.agentic._format_docs``, which
    reads ``metadata["episode"]`` as a dict with ``subject`` + ``content`` and
    a ms-epoch ``metadata["timestamp"]``. Agent-kind records carry their body
    across several fields (``name``/``description`` for skills,
    ``task_intent``/``approach`` for cases), so ``format_passage`` is the
    same kind-shaped formatter the reranker uses -- the passage the LLM
    sufficiency check sees is the passage the reranker scores.
    ``content`` falls back to ``_EMPTY_PASSAGE`` when the formatter yields
    nothing (both source fields empty, reachable on the case side): without
    it ``_format_docs`` raises on the empty string and one malformed record
    500s the whole request. Skills have no timestamp; ``_format_docs``
    renders that as ``N/A``.
    """
    metadata = dict(candidate.metadata)
    metadata["episode"] = {
        "subject": metadata.get("subject", ""),
        "content": format_passage(metadata) or _EMPTY_PASSAGE,
    }
    return candidate.model_copy(update={"metadata": metadata})


async def search_agent_cases_agentic(
    query: str,
    *,
    where: Filter,
    case_recaller: AgentCaseRecaller,
    embed_query_fn: Callable[[str], Awaitable[list[float]]],
    reranker: RerankProvider,
    llm: LLMClient,
    top_k: int,
) -> list[SearchAgentCaseItem]:
    """Agent-case AGENTIC search via flat hybrid retrieve + aagentic_retrieve.

    Args:
        query: User search query.
        where: Pre-compiled LanceDB filter string (owner + any request filters).
        case_recaller: AgentCase-table sparse + dense recall callbacks.
        embed_query_fn: Async ``(str) -> list[float]`` query embedder.
        reranker: Cross-encoder rerank provider.
        llm: LLM client for sufficiency check + multi-query generation.
        top_k: Maximum cases to return.

    Returns:
        Ranked list of at most ``top_k`` ``SearchAgentCaseItem`` objects.
    """
    candidates, stored = await _run_agentic_retrieve(
        query=query,
        where=where,
        recaller=case_recaller,
        embed_query_fn=embed_query_fn,
        reranker=reranker,
        llm=llm,
        top_k=top_k,
        kind="case",
    )
    return shape_agent_cases(candidates, stored)


async def search_agent_skills_agentic(
    query: str,
    *,
    where: Filter,
    skill_recaller: AgentSkillRecaller,
    embed_query_fn: Callable[[str], Awaitable[list[float]]],
    reranker: RerankProvider,
    llm: LLMClient,
    top_k: int,
) -> list[SearchAgentSkillItem]:
    """Agent-skill AGENTIC search via flat hybrid retrieve + aagentic_retrieve.

    Args:
        query: User search query.
        where: Pre-compiled LanceDB filter string (owner + any request filters).
        skill_recaller: AgentSkill-table sparse + dense recall callbacks.
        embed_query_fn: Async ``(str) -> list[float]`` query embedder.
        reranker: Cross-encoder rerank provider.
        llm: LLM client for sufficiency check + multi-query generation.
        top_k: Maximum skills to return.

    Returns:
        Ranked list of at most ``top_k`` ``SearchAgentSkillItem`` objects.
    """
    candidates, stored = await _run_agentic_retrieve(
        query=query,
        where=where,
        recaller=skill_recaller,
        embed_query_fn=embed_query_fn,
        reranker=reranker,
        llm=llm,
        top_k=top_k,
        kind="skill",
    )
    return shape_agent_skills(candidates, stored)


async def _run_agentic_retrieve(
    query: str,
    *,
    where: Filter,
    recaller: AgentCaseRecaller | AgentSkillRecaller,
    embed_query_fn: Callable[[str], Awaitable[list[float]]],
    reranker: RerankProvider,
    llm: LLMClient,
    top_k: int,
    kind: Literal["case", "skill"],
) -> tuple[list[Candidate], Mapping[str, Stored[Any]]]:
    """Shared flat agentic retrieve pipeline for agent memory kinds.

    Builds a hybrid_full retrieve closure over the given recaller and
    hands it to ``aagentic_retrieve`` with hyperparameters aligned to the
    memsys_opensource ``AgenticConfig`` defaults.
    No cluster or MaxSim step: agent memory is small enough for a flat pass.

    ``kind`` selects the passage formatter and the rerank fn together — the
    passage the LLM sufficiency check sees and the passage the cross-encoder
    scores must be the same shape, or the two stages silently disagree on
    what "relevant" means.
    """
    if kind == "case":
        passage_formatter = _format_case_passage_from_metadata
        rerank_fn = build_case_rerank_fn(reranker)
    else:
        passage_formatter = _format_skill_passage_from_metadata
        rerank_fn = build_skill_rerank_fn(reranker)

    stored: dict[str, Stored[Any]] = {}

    def _keep(result: RecallResult[Any]) -> list[Candidate]:
        stored.update(result.stored_by_id)
        return result.candidates

    async def _dense(q: str, k: int) -> list[Candidate]:
        vec = await embed_query_fn(q)
        if not vec:
            return []
        return _keep(await recaller.dense_recall(vec, where, limit=k))

    async def _sparse(q: str, k: int) -> list[Candidate]:
        return _keep(await recaller.sparse_recall(q, where, limit=k))

    async def hybrid_full(q: str, k: int) -> list[Candidate]:
        with memory_span(
            "everos.search.recall",
            observation_type="retriever",
            metadata={"phase": "agentic_hybrid"},
        ):
            hits = await ahybrid_retrieve(
                q,
                dense_retrieve=_dense,
                sparse_retrieve=_sparse,
                top_n=k,
                dense_candidates=_DENSE_CANDIDATES,
                sparse_candidates=_SPARSE_CANDIDATES,
                rrf_k=_HYBRID_RRF_K,
            )
        # Bridge to the everalgo doc contract so ``_format_docs`` (the LLM
        # sufficiency / multi-query prompt) sees an episode dict + ms
        # timestamp; agent-kind rows otherwise lack it and _format_docs raises.
        return [_with_doc_metadata(c, format_passage=passage_formatter) for c in hits]

    candidates, _decision = await aagentic_retrieve(
        query,
        base_retrieve=hybrid_full,
        round2_retrieve=None,
        round2_cap=_ROUND2_CAP,
        rerank_fn=rerank_fn,
        llm=llm,
        top_n=top_k,
        round1_top_n=_ROUND1_TOP_N,
        round1_rerank_top_n=_ROUND1_RERANK_TOP_N,
        refinement_strategy=_REFINEMENT_STRATEGY,
        multi_query_count=_MULTI_QUERY_COUNT,
        rrf_k=_HYBRID_RRF_K,
    )
    return candidates, stored
