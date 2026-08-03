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

import datetime as _dt
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from everalgo.rank.agentic import aagentic_retrieve
from everalgo.rank.hybrid import ahybrid_retrieve
from everalgo.types import Candidate

from everos.component.utils.datetime import from_timestamp, to_timestamp_ms
from everos.core.observability.tracing import memory_span
from everos.memory.search.callbacks import build_rerank_fn
from everos.memory.search.shaper import (
    shape_agent_case_from_candidate,
    shape_agent_skill_from_candidate,
)

from .dto import SearchAgentCaseItem, SearchAgentSkillItem

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

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


def _to_everalgo_doc_metadata(
    metadata: dict[str, Any], *, text_field: str
) -> dict[str, Any]:
    """Bridge agent recall metadata to the everalgo ``_format_docs`` contract.

    ``aagentic_retrieve`` renders round-1 candidates into the sufficiency /
    multi-query LLM prompt via ``everalgo.rank.agentic._format_docs``, which
    reads ``metadata["episode"]`` as a dict with ``subject`` + ``content`` and
    a ms-epoch ``metadata["timestamp"]``. Agent-kind rows carry their body in
    ``text_field`` (``task_intent`` / ``skill``) and the time in ``timestamp``
    (datetime); without this bridge ``_format_docs`` raises ``TypeError``.
    Mirrors the episode path's bridge in ``agentic.py``.
    ``_restore_shaper_metadata`` reverts it before DTO shaping.
    """
    bridged = dict(metadata)
    content = metadata.get(text_field)
    if isinstance(content, str):
        bridged["episode"] = {
            "subject": metadata.get("subject", ""),
            "content": content,
        }
    timestamp = metadata.get("timestamp")
    if isinstance(timestamp, _dt.datetime):
        bridged["timestamp"] = to_timestamp_ms(timestamp)
    return bridged


def _restore_shaper_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Revert ``_to_everalgo_doc_metadata`` before agent-DTO shaping.

    The shaper reads ``timestamp`` as a ``datetime``; the bridged ms-epoch
    int must be reverted. The injected ``episode`` dict is inert for the
    agent shapers (they read case/skill fields), so it is simply dropped.

    Deliberately wider than an exact inverse of the bridge: a row whose
    ``timestamp`` was *natively* numeric (agent tables store datetimes
    today, so this does not occur) would also be converted — correct
    either way, because the shaper contract requires a ``datetime``.
    """
    reverted = dict(metadata)
    timestamp = metadata.get("timestamp")
    if isinstance(timestamp, (int, float)):
        reverted["timestamp"] = from_timestamp(timestamp)
    reverted.pop("episode", None)
    return reverted


async def search_agent_cases_agentic(
    query: str,
    *,
    where: str,
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
    candidates = await _run_agentic_retrieve(
        query=query,
        where=where,
        recaller=case_recaller,
        embed_query_fn=embed_query_fn,
        reranker=reranker,
        llm=llm,
        top_k=top_k,
    )
    return [
        item
        for c in candidates
        for item in [shape_agent_case_from_candidate(c)]
        if item is not None
    ]


async def search_agent_skills_agentic(
    query: str,
    *,
    where: str,
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
    candidates = await _run_agentic_retrieve(
        query=query,
        where=where,
        recaller=skill_recaller,
        embed_query_fn=embed_query_fn,
        reranker=reranker,
        llm=llm,
        top_k=top_k,
    )
    return [
        item
        for c in candidates
        for item in [shape_agent_skill_from_candidate(c)]
        if item is not None
    ]


async def _run_agentic_retrieve(
    query: str,
    *,
    where: str,
    recaller: AgentCaseRecaller | AgentSkillRecaller,
    embed_query_fn: Callable[[str], Awaitable[list[float]]],
    reranker: RerankProvider,
    llm: LLMClient,
    top_k: int,
) -> list[Candidate]:
    """Shared flat agentic retrieve pipeline for agent memory kinds.

    Builds a hybrid_full retrieve closure over the given recaller and
    hands it to ``aagentic_retrieve`` with hyperparameters aligned to the
    memsys_opensource ``AgenticConfig`` defaults.
    No cluster or MaxSim step: agent memory is small enough for a flat pass.
    """

    async def _dense(q: str, k: int) -> list[Candidate]:
        vec = await embed_query_fn(q)
        if not vec:
            return []
        return await recaller.dense_recall(vec, where, limit=k)

    async def _sparse(q: str, k: int) -> list[Candidate]:
        return await recaller.sparse_recall(q, where, limit=k)

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
        return [
            c.model_copy(
                update={
                    "metadata": _to_everalgo_doc_metadata(
                        c.metadata, text_field=recaller.text_field
                    )
                }
            )
            for c in hits
        ]

    rerank_fn = build_rerank_fn(reranker, text_field=recaller.text_field)

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
    # Revert the doc-contract bridge so the agent DTO shapers see the
    # original metadata shape (timestamp as datetime, no ``episode`` dict).
    return [
        c.model_copy(update={"metadata": _restore_shaper_metadata(c.metadata)})
        for c in candidates
    ]
