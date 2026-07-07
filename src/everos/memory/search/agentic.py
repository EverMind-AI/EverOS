"""Episode AGENTIC cluster-path orchestration — 1:1 with everalgo benchmark.

Implements the cluster main path from ``benchmarks/common/stages/search.py``
(``enable_cluster_retrieval=True``):

    fact-MaxSim (dense + sparse)
        -> ahybrid_retrieve  (hybrid_full)
        -> acluster_retrieve (cluster_scoped, base=hybrid_full)
        -> aagentic_retrieve (base=cluster_scoped, round2=hybrid_full)

Hyperparameters match benchmark ``config.py`` defaults and are frozen as
module-level constants — no env/TOML knobs at this layer.

id contract: candidates flowing through the pipeline carry
``id=memcell_id`` (regular episodes, parent_type=memcell) or
``id=entry_id`` (merged episodes, parent_type=cluster). The final
shaping step remaps to ``id=episode_id`` via ``metadata["episode_id"]``
before calling ``shape_episode_from_candidate``.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from everalgo.rank.agentic import aagentic_retrieve
from everalgo.rank.cluster import acluster_retrieve
from everalgo.rank.hybrid import ahybrid_retrieve
from everalgo.rank.maxsim import amaxsim_retrieve
from everalgo.types import Candidate

from everos.component.utils.datetime import from_timestamp, to_timestamp_ms
from everos.core.observability.logging import get_logger
from everos.infra.persistence.sqlite import cluster_repo
from everos.memory.search.callbacks import build_rerank_fn
from everos.memory.search.shaper import shape_episode_from_candidate

from .dto import SearchEpisodeItem

logger = get_logger(__name__)

if TYPE_CHECKING:
    from everalgo.clustering import Cluster
    from everalgo.llm.protocols import LLMClient

    from everos.component.rerank import RerankProvider
    from everos.memory.search.recall.atomic_fact import AtomicFactRecaller
    from everos.memory.search.recall.episode import EpisodeRecaller

# ── Benchmark hyperparameters (config.py defaults) ──────────────────────────
_DENSE_CANDIDATES: int = 50
_SPARSE_CANDIDATES: int = 50
_HYBRID_RRF_K: int = 40
_CLUSTER_BASE_CANDIDATES: int | None = None
_CLUSTER_TOP_K: int = 10
_ROUND1_TOP_N: int = 50
_ROUND1_RERANK_TOP_N: int = 10
_ROUND2_CAP: int = 40
_MULTI_QUERY_COUNT: int = 3
_REFINEMENT_STRATEGY: str = "multi_query"

# Child-pool sizing for amaxsim_retrieve. The benchmark passes
# len(full_fact_corpus); EverOS doesn't know the corpus size upfront,
# so we pass a large sentinel and let the LanceDB limit clamp naturally.
_FACT_CHILD_CANDIDATES: int = 100_000

# Qwen3-Reranker task instruction for the search scene (benchmark
# ``config.reranker_instruction``). Steers the cross-encoder toward fact /
# entity / detail relevance rather than topical similarity.
_RERANK_INSTRUCTION: str = (
    "Determine if the passage contains specific facts, entities "
    "(names, dates, locations), or details that directly answer the question."
)


async def search_episodes_agentic(
    query: str,
    *,
    owner_id: str,
    where: str,
    app_id: str = "default",
    project_id: str = "default",
    episode_recaller: EpisodeRecaller,
    atomic_fact_recaller: AtomicFactRecaller,
    embed_query_fn: Callable[[str], Awaitable[list[float]]],
    reranker: RerankProvider,
    llm: LLMClient,
    top_k: int,
) -> list[SearchEpisodeItem]:
    """Episode AGENTIC search via cluster-scoped MaxSim — 1:1 with benchmark.

    Args:
        query: User search query.
        owner_id: Owner whose memories are searched.
        where: Pre-compiled LanceDB filter string (owner + any request filters).
        episode_recaller: Episode-table sparse + dense + fetch callbacks.
        atomic_fact_recaller: AtomicFact-table sparse + dense callbacks.
        embed_query_fn: Async ``(str) -> list[float]`` query embedder.
        reranker: Cross-encoder rerank provider.
        llm: LLM client for sufficiency check + multi-query generation.
        top_k: Maximum episodes to return (maps to ``top_n`` in aagentic_retrieve).

    Returns:
        Ranked list of at most ``top_k`` ``SearchEpisodeItem`` objects.
        Empty when no clusters exist or retrieval returns nothing.
    """

    # 1. Fact-level child retrieve closures (dense + sparse via both tables).
    async def _fact_dense(q: str, k: int) -> list[Candidate]:
        vec = await embed_query_fn(q)
        if not vec:
            return []
        fact_results, ep_results = await asyncio.gather(
            atomic_fact_recaller.dense_recall(vec, where, limit=k),
            episode_recaller.dense_recall_subject_as_child(vec, where, limit=k),
        )
        return fact_results + ep_results

    async def _fact_sparse(q: str, k: int) -> list[Candidate]:
        fact_results, ep_results = await asyncio.gather(
            atomic_fact_recaller.sparse_recall(q, where, limit=k),
            episode_recaller.sparse_recall_as_child(q, where, limit=k),
        )
        return fact_results + ep_results

    # 2. parent_fetch: maps entry_ids (from atomic_fact.parent_id) to episodes.
    #    Atomic facts always point to episodes via entry_id regardless of
    #    whether the episode is memcell-based or cluster-merged.
    async def _parent_fetch(parent_ids: list[str]) -> list[Candidate]:
        episodes = await episode_recaller.fetch_by_entry_ids(parent_ids, where)
        result: list[Candidate] = []
        for c in episodes:
            entry_id = c.metadata.get("entry_id")
            if not isinstance(entry_id, str):
                continue
            result.append(
                Candidate(
                    id=entry_id,
                    score=0.0,
                    source=c.source,
                    metadata=_to_everalgo_doc_metadata(
                        {**c.metadata, "episode_id": c.id}
                    ),
                )
            )
        return result

    # 3. MaxSim RetrieveFns: fact vectors/BM25 -> max-pool by memcell -> candidates.
    async def _dense(q: str, k: int) -> list[Candidate]:
        return await amaxsim_retrieve(
            q,
            child_retrieve=_fact_dense,
            parent_fetch=_parent_fetch,
            top_n=k,
            child_candidates=_FACT_CHILD_CANDIDATES,
        )

    async def _sparse(q: str, k: int) -> list[Candidate]:
        return await amaxsim_retrieve(
            q,
            child_retrieve=_fact_sparse,
            parent_fetch=_parent_fetch,
            top_n=k,
            child_candidates=_FACT_CHILD_CANDIDATES,
        )

    # 4. hybrid_full: RRF fusion of dense + sparse MaxSim.
    async def hybrid_full(q: str, k: int) -> list[Candidate]:
        return await ahybrid_retrieve(
            q,
            dense_retrieve=_dense,
            sparse_retrieve=_sparse,
            top_n=k,
            dense_candidates=_DENSE_CANDIDATES,
            sparse_candidates=_SPARSE_CANDIDATES,
            rrf_k=_HYBRID_RRF_K,
        )

    # 5. Load cluster snapshot + full-corpus all_docs (memcell-keyed).
    #    Reshape metadata to the everalgo doc contract so the sufficiency /
    #    multi-query LLM prompt (rendered by ``_format_docs``) sees the episode
    #    body and a ms-epoch date instead of the memcell id.
    clusters: list[Cluster] = await cluster_repo.list_for_owner(
        owner_id,
        "user_memory",
        app_id=app_id,
        project_id=project_id,
    )
    raw_all_docs = await episode_recaller.fetch_all_for_owner(where)
    all_docs: list[Candidate] = [
        c.model_copy(update={"metadata": _to_everalgo_doc_metadata(c.metadata)})
        for c in raw_all_docs
    ]

    # 6. cluster_scoped: narrows hybrid_full to top-K cluster member expansions.
    async def cluster_scoped(q: str, _k: int) -> list[Candidate]:
        return await acluster_retrieve(
            q,
            base_retrieve=hybrid_full,
            base_candidates=_CLUSTER_BASE_CANDIDATES,
            clusters=clusters,
            all_docs=all_docs,
            cluster_top_k=_CLUSTER_TOP_K,
        )

    # 7. Cross-encoder rerank fn (2-arg RerankFn, no internal truncation).
    rerank_fn = build_rerank_fn(
        reranker, text_field="episode", instruction=_RERANK_INSTRUCTION
    )

    # 8. aagentic_retrieve — benchmark cluster main path.
    candidates, decision = await aagentic_retrieve(
        query,
        base_retrieve=cluster_scoped,
        round2_retrieve=hybrid_full,
        llm=llm,
        rerank_fn=rerank_fn,
        round2_cap=_ROUND2_CAP,
        top_n=top_k,
        round1_top_n=_ROUND1_TOP_N,
        round1_rerank_top_n=_ROUND1_RERANK_TOP_N,
        refinement_strategy=_REFINEMENT_STRATEGY,
        multi_query_count=_MULTI_QUERY_COUNT,
        rrf_k=_HYBRID_RRF_K,
    )

    round_tag = "round1" if decision.is_sufficient else "round2"
    logger.info("agentic_search_decision", round=round_tag, query=query[:80])

    # 9. Shape: remap id from memcell_id -> episode_id, then build DTO.
    return _shape_results(candidates)


def _to_everalgo_doc_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Bridge recall metadata to the everalgo ``_format_docs`` doc contract.

    ``aagentic_retrieve`` renders Round-1 candidates into the sufficiency /
    multi-query LLM prompt via ``everalgo.rank.agentic._format_docs``, which
    reads ``metadata["episode"]`` as a dict with ``subject`` + ``content``
    keys and the date from a ms-epoch ``metadata["timestamp"]``. everos
    episode rows carry the body in ``episode`` (str) and the time in
    ``timestamp`` (datetime); without this bridge ``_format_docs`` raises
    ``TypeError``.

    The flat ``episode`` string is also kept as ``text`` for the reranker
    (which reads a plain string). ``_restore_shaper_metadata`` reverts
    the restructured metadata before DTO shaping.
    """
    bridged = dict(metadata)
    episode = metadata.get("episode")
    if isinstance(episode, str):
        bridged["text"] = episode
        bridged["episode"] = {
            "subject": metadata.get("subject", ""),
            "content": episode,
        }
    timestamp = metadata.get("timestamp")
    if isinstance(timestamp, _dt.datetime):
        bridged["timestamp"] = to_timestamp_ms(timestamp)
    return bridged


def _restore_shaper_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Revert bridged metadata fields before DTO shaping.

    Undoes two transforms from ``_to_everalgo_doc_metadata``:
    1. ``timestamp``: ms-epoch (int) → ``datetime`` (shaper requires it).
    2. ``episode``: dict ``{"subject", "content"}`` → flat str (shaper
       reads ``metadata["episode"]`` as a plain string).
    """
    reverted = dict(metadata)
    timestamp = metadata.get("timestamp")
    if isinstance(timestamp, (int, float)):
        reverted["timestamp"] = from_timestamp(timestamp)
    episode = metadata.get("episode")
    if isinstance(episode, dict):
        reverted["episode"] = episode.get("content", "")
    return reverted


def _shape_results(candidates: list[Candidate]) -> list[SearchEpisodeItem]:
    """Remap candidate id from memcell_id -> episode_id; build the DTO list."""
    result: list[SearchEpisodeItem] = []
    for c in candidates:
        ep_id = c.metadata.get("episode_id")
        if not isinstance(ep_id, str):
            continue
        ep_cand = Candidate(
            id=ep_id,
            score=c.score,
            source=c.source,
            metadata=_restore_shaper_metadata(c.metadata),
        )
        item = shape_episode_from_candidate(ep_cand)
        if item is not None:
            result.append(item)
    return result
