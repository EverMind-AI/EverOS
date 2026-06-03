"""Skill HYBRID cross-encoder lane: rrf → cross-encoder rerank → shape.

Used when ``enable_llm_rerank`` is off. The LLM-rerank lane lives in
:meth:`SearchManager._search_agent_skills` so the flag stays wired only
at the orchestration layer. Passage shape + skill instruction live in
:func:`memory.search.callbacks.build_skill_rerank_fn`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everalgo.rank.fusion import rrf
from everalgo.types import Candidate

from everos.core.observability.logging import get_logger
from everos.memory.search.callbacks import build_skill_rerank_fn
from everos.memory.search.shaper import shape_agent_skill_from_candidate

from .dto import SearchAgentSkillItem

if TYPE_CHECKING:
    from everos.component.rerank import RerankProvider

logger = get_logger(__name__)


async def search_agent_skills_hybrid(
    query: str,
    *,
    sparse: list[Candidate],
    dense: list[Candidate],
    reranker: RerankProvider,
    top_k: int,
) -> list[SearchAgentSkillItem]:
    """Skill HYBRID retrieval: rrf → cross-encoder rerank → shape.

    Args:
        query: User search query.
        sparse: BM25 recall candidates for the skill table.
        dense: Vector ANN recall candidates for the skill table.
        reranker: Cross-encoder rerank provider (not LLM-based).
        top_k: Maximum skills to return after rerank.

    Returns:
        Ranked list of at most ``top_k`` ``SearchAgentSkillItem`` objects.
    """
    fused = _fuse(sparse, dense)
    reranked = await _cross_encoder_rerank(query, fused, reranker, top_k)
    return _shape_results(reranked)


# ── Pipeline steps ────────────────────────────────────────────────────────


def _fuse(sparse: list[Candidate], dense: list[Candidate]) -> list[Candidate]:
    """RRF fusion of sparse and dense candidates."""
    return rrf(sparse, dense)


async def _cross_encoder_rerank(
    query: str,
    candidates: list[Candidate],
    reranker: RerankProvider,
    top_k: int,
) -> list[Candidate]:
    """Cross-encoder rerank via the skill-shaped factory, then slice to top_k."""
    if not candidates:
        return []
    rerank_fn = build_skill_rerank_fn(reranker)
    reranked = await rerank_fn(query, candidates)
    return reranked[:top_k]


def _shape_results(candidates: list[Candidate]) -> list[SearchAgentSkillItem]:
    """Shape each Candidate into a SearchAgentSkillItem, dropping malformed rows."""
    return [
        item
        for c in candidates
        for item in [shape_agent_skill_from_candidate(c)]
        if item is not None
    ]
