"""SearchManager — top-level orchestrator for ``POST /api/v2/memory/search``.

Hard partition by ``owner_type``:

* ``user``  → ``episodes`` (+ ``profiles`` when ``include_profile=true``)
* ``agent`` → ``agent_cases`` + ``agent_skills``

Per kind, :func:`memory.search.adapter.resolve_pipeline` decides whether
the path is "single-route recall, no fusion" (``KEYWORD`` / ``VECTOR``)
or "sparse + dense → everalgo.rank" (``HYBRID`` / ``AGENTIC``). Component
guards (embedding / cross-encoder / LLM) raise early when a method is
selected without its prerequisites.

``HYBRID`` defaults to **no LLM rerank** — the response comes back
straight after the heap-expand pipeline (RRF-ordered expansion → LR-calibrated
global top-N competition with fact eviction). ``enable_llm_rerank`` is
**ignored** for the hierarchy path. ``AGENTIC`` keeps its own
internal cross-encoder rerank loop; the flag is ignored there.

``SearchEpisodeItem.atomic_facts`` is populated **only** when the HYBRID
pipeline runs over episodes. The other methods leave it empty: there is
no query-relevance score we can assign to a fact pulled by parent_id
alone, and emitting ``score=0.0`` facts would muddy the contract.

The manager never writes to storage; it only reads LanceDB + markdown.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

from everalgo.rank import DEFAULT_RANK_CONFIG, RankConfig, arank
from everalgo.rank.fusion import rrf
from everalgo.types import Candidate, RankInput

from everos.component.embedding import get_embedding_capability
from everos.component.rerank import get_rerank_capability
from everos.component.utils.datetime import to_display_tz
from everos.config import load_settings
from everos.core.context import resolve_request_id
from everos.core.errors import ConfigurationError, ProviderNotConfiguredError
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import (
    capture_input,
    capture_output,
    current_trace_ids,
    emit_recall_scores,
    memory_span,
)
from everos.infra.persistence.sqlite import (
    UnprocessedBuffer,
    unprocessed_buffer_repo,
)

from .adapter import resolve_pipeline
from .agentic import search_episodes_agentic
from .agentic_agent import search_agent_cases_agentic, search_agent_skills_agentic
from .dto import (
    FilterNode,
    SearchAgentCaseItem,
    SearchAgentSkillItem,
    SearchData,
    SearchEpisodeItem,
    SearchMethod,
    SearchProfileItem,
    SearchRequest,
    SearchResponse,
    UnprocessedMessageDTO,
)
from .filters import compile_filters
from .hierarchy import build_ep_to_fact_parents, heap_expand
from .shaper import (
    reshape_hybrid_output,
    shape_agent_case_from_candidate,
    shape_agent_skill_from_candidate,
    shape_episode_from_candidate,
)
from .skill_hybrid import search_agent_skills_hybrid

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

    from everos.component.embedding import EmbeddingProvider
    from everos.component.rerank import RerankProvider
    from everos.component.tokenizer import Tokenizer

    from .recall import (
        AgentCaseRecaller,
        AgentSkillRecaller,
        AtomicFactRecaller,
        EpisodeRecaller,
        ProfileRecaller,
    )

logger = get_logger(__name__)

# Recall pool sizing — matches the legacy enterprise constants
# ``DEFAULT_RECALL_MULTIPLIER`` / ``DEFAULT_TOPK_LIMIT``.
# Multiplier kicks in for ``top_k > 0``; ``top_k = -1`` (unlimited) is capped
# at the fixed top-k limit (100) rather than ``100 * multiplier`` — that way
# the recall pool never balloons to 500 in unlimited mode.
_DEFAULT_RECALL_MULTIPLIER = 2
_DEFAULT_TOP_K_CAP = 100

# Agent cases / skills carry heavy per-row payloads (``approach``,
# full ``content``); cap unlimited mode at 10 to keep rerank context
# bounded. Positive ``top_k`` from the caller bypasses this.
_AGENT_TOP_K_CAP = 10

# Vector ``radius`` (cosine similarity threshold) default for **unlimited
# mode only**. In ``top_k > 0`` mode we trust the truncation cap to ditch
# low-quality tail; in ``top_k = -1`` mode we would otherwise return up to
# 100 candidates with no quality floor, so we layer a default 0.5
# similarity threshold the way enterprise does (enterprise uses 0.6 — we
# pick 0.5 slightly looser because LanceDB cosine vs Milvus cosine score
# distributions can drift a bit on the same model).
_DEFAULT_UNLIMITED_RADIUS = 0.5

# ``maxsim_atomic`` recall pool sizing — atomic facts are ~28× denser than
# episodes (one memcell → 1 episode + ~28 atomic facts), so the fact pool
# is sized as ``top_k_episode * 20`` to consistently cover enough distinct
# parent memcells before the max-pool reduction. Capped to keep the ANN
# scan bounded on very large top_k requests.
_MAXSIM_FACT_MULTIPLIER = 20
_MAXSIM_FACT_POOL_CAP = 2000

# Mirror of ``service._boundary._TRACK``. The unprocessed buffer is a single
# shared track because boundary detection is single-pass — switching mode
# requires a fresh process. Hard-coded here (instead of importing) to keep
# the memory layer free of service-layer imports per the DDD direction rule.
_UNPROCESSED_TRACK = "memorize"


def _top_score(data: SearchData) -> float:
    """Max relevance score across scored result items (0.0 when empty).

    Profiles are excluded — they are a KV fetch with no query-relevance score.
    """
    items = [*data.episodes, *data.agent_cases, *data.agent_skills]
    return max((item.score for item in items), default=0.0)


# Methods whose top score is calibrated to a comparable [0, 1] scale
# (HYBRID → LR sigmoid, AGENTIC → cross-encoder), so the recall_hit
# threshold is meaningful. KEYWORD (unbounded BM25) and single-route
# VECTOR are excluded — a fixed threshold there yields a misleading
# near-constant "hit" that inflates cross-method dashboards. Their top
# score is also reported under a separate score name (see ``scores``).
_CALIBRATED_METHODS = frozenset({SearchMethod.HYBRID, SearchMethod.AGENTIC})


class SearchManager:
    """Orchestrates per-kind recall, fusion, and shape into the public DTO."""

    def __init__(
        self,
        *,
        episode_recaller: EpisodeRecaller,
        atomic_fact_recaller: AtomicFactRecaller,
        agent_case_recaller: AgentCaseRecaller,
        agent_skill_recaller: AgentSkillRecaller,
        profile_recaller: ProfileRecaller,
        embedding: EmbeddingProvider | None,
        reranker: RerankProvider | None,
        llm_client: LLMClient | None,
        search_tokenizer: Tokenizer | None = None,
    ) -> None:
        self._ep = episode_recaller
        self._fact = atomic_fact_recaller
        self._case = agent_case_recaller
        self._skill = agent_skill_recaller
        self._profile = profile_recaller
        self._embedding = embedding
        self._reranker = reranker
        self._llm = llm_client
        self._search_tokenizer = search_tokenizer

    # ── Public entry ────────────────────────────────────────────────

    async def search(self, req: SearchRequest) -> SearchResponse:
        request_id = resolve_request_id()
        with memory_span(
            "everos.memory.search",
            observation_type="retriever",
            session_id=_extract_top_level_session_id(req.filters),
            user_id=req.user_id,
            metadata={
                "request_id": request_id,
                "app_id": req.app_id,
                "project_id": req.project_id,
                "agent_id": req.agent_id,
                "owner_type": req.owner_type,
                "method": req.method.value,
            },
        ) as span:
            # Content (query text) only when capture_content is on.
            capture_input(
                span,
                {"query": req.query, "top_k": req.top_k, "method": req.method.value},
            )
            # Compile filters first: a malformed `filters` payload is a user
            # input error (422) and should surface before the server-side
            # component guard (500). The two steps are independent.
            where = compile_filters(
                req.filters,
                owner_id=req.owner_id,
                owner_type=req.owner_type,
                app_id=req.app_id,
                project_id=req.project_id,
            )
            self._validate_components(req)

            if req.owner_type == "user":
                episodes, profiles, unprocessed = await asyncio.gather(
                    self._search_episodes(req, where),
                    self._fetch_profile(req),
                    self._load_unprocessed(req),
                )
                data = SearchData(
                    episodes=episodes,
                    profiles=profiles,
                    unprocessed_messages=unprocessed,
                )
            else:  # "agent"
                (cases, skills), unprocessed = await asyncio.gather(
                    self._search_cases_and_skills(req, where),
                    self._load_unprocessed(req),
                )
                data = SearchData(
                    agent_cases=cases,
                    agent_skills=skills,
                    unprocessed_messages=unprocessed,
                )

            # Returned hits (ids only) — content, so only when capture_content
            # is on. This is the point of a search trace: what came back.
            capture_output(
                span,
                {
                    "episodes": [e.id for e in data.episodes],
                    "agent_cases": [c.id for c in data.agent_cases],
                    "agent_skills": [s.id for s in data.agent_skills],
                },
            )

            # Recall-quality signal on the span (always on; the Langfuse
            # scores push is separate and gated on creds — see the score sink).
            # `hit` is only meaningful for calibrated-score methods; leave it
            # unset for KEYWORD/VECTOR so an unbounded score isn't forced
            # through a fixed threshold into a misleading always-hit verdict.
            top_score = _top_score(data)
            span.set_attribute("everos.search.top_score", top_score)
            hit: bool | None = None
            if req.method in _CALIBRATED_METHODS:
                threshold = load_settings().observability.recall_hit_threshold
                hit = top_score >= threshold
                span.set_attribute("everos.search.hit", hit)

            # Push recall-quality scores to Langfuse out-of-band (no-op unless
            # a score sink is configured); attach to this retriever span.
            ids = current_trace_ids()
            if ids is not None:
                emit_recall_scores(
                    trace_id=ids[0],
                    observation_id=ids[1],
                    top_score=top_score,
                    hit=hit,
                    method=req.method.value,
                )

            return SearchResponse(request_id=request_id, data=data)

    # ── Unprocessed buffer ──────────────────────────────────────────

    async def _load_unprocessed(
        self, req: SearchRequest
    ) -> list[UnprocessedMessageDTO]:
        """Load in-flight buffer rows for ``filters.session_id`` (if present).

        Returns ``[]`` unless ``filters`` carries a top-level ``session_id``
        eq scalar — buffer rows have no ``user_id`` / ``agent_id`` attribution
        (boundary detection runs before owner inference), so session is the
        only meaningful query dimension.
        """
        session_id = _extract_top_level_session_id(req.filters)
        if session_id is None:
            return []
        rows = await unprocessed_buffer_repo.list_for_track(
            session_id,
            _UNPROCESSED_TRACK,
            app_id=req.app_id,
            project_id=req.project_id,
        )
        return [_unprocessed_buffer_to_dto(r) for r in rows]

    # ── Agent partition ─────────────────────────────────────────────

    async def _search_cases_and_skills(
        self, req: SearchRequest, where: str
    ) -> tuple[list[SearchAgentCaseItem], list[SearchAgentSkillItem]]:
        """Cases + skills, serial when bridging.

        HYBRID + LLM rerank runs serially: reranked cases feed the
        skill bridge. Every other method runs the two kinds in parallel
        with no bridge — the bridge only pays off after rerank has
        produced high-quality case scores to inherit.
        """
        if _effective_llm_rerank(req):
            cases = await self._search_agent_cases(req, where)
            bridge_cases = [
                Candidate(id=c.id, score=c.score, source="vector", metadata={})
                for c in cases
            ]
            skills = await self._search_agent_skills(
                req, where, bridge_cases=bridge_cases
            )
            return cases, skills

        cases, skills = await asyncio.gather(
            self._search_agent_cases(req, where),
            self._search_agent_skills(req, where),
        )
        return cases, skills

    # ── Episodes ────────────────────────────────────────────────────

    async def _search_episodes(
        self, req: SearchRequest, where: str
    ) -> list[SearchEpisodeItem]:
        if req.method == SearchMethod.AGENTIC:
            return await search_episodes_agentic(
                req.query,
                owner_id=req.owner_id,
                where=where,
                app_id=req.app_id,
                project_id=req.project_id,
                episode_recaller=self._ep,
                atomic_fact_recaller=self._fact,
                embed_query_fn=self._embedding.embed,  # type: ignore[union-attr]
                reranker=self._reranker,  # type: ignore[arg-type]
                llm=self._llm,  # type: ignore[arg-type]
                top_k=self._top_k(req.top_k),
            )

        fusion_mode, _ = resolve_pipeline(req.method, "episode")
        enable_rerank = _effective_llm_rerank(req)
        top_k = self._top_k(req.top_k)

        # ── KEYWORD / VECTOR: single-route recall ──
        if fusion_mode is None:
            with memory_span(
                "everos.search.recall",
                observation_type="retriever",
                metadata={"phase": "single_route", "method": req.method.value},
            ):
                if req.method == SearchMethod.KEYWORD:
                    cands = await self._ep.sparse_recall(
                        req.query, where, limit=self._recall_limit(req.top_k)
                    )
                else:
                    cands = await self._maxsim_atomic_recall(req, where, top_k)
            # ``atomic_facts`` stays empty: facts come back only when the HYBRID
            # pipeline surfaces them with a score (see ``reshape_hybrid_output``).
            # Single-route recall has no per-fact score against the query, so
            # we do not back-fill — that would emit ``score=0.0`` facts whose
            # semantics are ambiguous.
            return [
                ep
                for ep in (shape_episode_from_candidate(c) for c in cands[:top_k])
                if ep is not None
            ]

        # ── HYBRID: parallel sparse + dense recall ──
        sparse, dense, query_vector = await self._recall_sparse_dense(
            self._ep, req, where, top_k
        )

        if fusion_mode == "hierarchy":
            with memory_span(
                "everos.search.rank",
                observation_type="span",
                metadata={"phase": "hierarchy"},
            ):
                rrf_candidates = rrf(sparse, dense)
                ep_to_parents = build_ep_to_fact_parents(rrf_candidates)
                episode_to_facts = await self._fact.facts_for_episodes(
                    ep_to_parents,
                    where,
                    per_episode=max(top_k * 2, 20),
                    query_vector=query_vector,
                )
                scored = heap_expand(
                    sparse=sparse,
                    dense=dense,
                    episode_to_facts=episode_to_facts,
                    top_k=top_k,
                )
                episode_pool = {c.id: c for c in (*sparse, *dense)}
                shaped = reshape_hybrid_output(scored, episode_pool=episode_pool)
                if req.min_score is not None:
                    shaped = [s for s in shaped if s.score >= req.min_score]
                return shaped

        # rrf / lr: standard everalgo fusion path (fallback).
        with memory_span(
            "everos.search.rank",
            observation_type="span",
            metadata={"phase": fusion_mode},
        ):
            output = await arank(
                RankInput(
                    query=req.query,
                    memory_type=self._ep.everalgo_memory_type,  # type: ignore[arg-type]
                    sparse_candidates=sparse,
                    dense_candidates=dense,
                    top_k=top_k,
                    radius=_effective_radius(req),
                ),
                config=RankConfig(fusion_mode=fusion_mode)
                if fusion_mode != "rrf"
                else DEFAULT_RANK_CONFIG,
                llm=self._llm,
                enable_rerank=enable_rerank,
                rerank_top_k=top_k,
            )
        ep_candidates = (_scored_as_candidate(s) for s in output.items)
        return [
            ep
            for ep in (shape_episode_from_candidate(c) for c in ep_candidates)
            if ep is not None
        ]

    # ── Agent cases ─────────────────────────────────────────────────

    async def _search_agent_cases(
        self, req: SearchRequest, where: str
    ) -> list[SearchAgentCaseItem]:
        if req.method == SearchMethod.AGENTIC:
            return await search_agent_cases_agentic(
                req.query,
                where=where,
                case_recaller=self._case,
                embed_query_fn=self._embedding.embed,  # type: ignore[union-attr]
                reranker=self._reranker,  # type: ignore[arg-type]
                llm=self._llm,  # type: ignore[arg-type]
                top_k=self._top_k(req.top_k),
            )
        fusion_mode, _ = resolve_pipeline(req.method, "agent_case")
        enable_rerank = _effective_llm_rerank(req)
        top_k = self._top_k(req.top_k, cap=_AGENT_TOP_K_CAP)

        if fusion_mode is None:
            cands = await self._single_route_recall(
                self._case, req, where, top_k, cap=_AGENT_TOP_K_CAP
            )
            shaped = (shape_agent_case_from_candidate(c) for c in cands[:top_k])
            return [item for item in shaped if item is not None]

        sparse, dense, _ = await self._recall_sparse_dense(
            self._case, req, where, top_k, cap=_AGENT_TOP_K_CAP
        )
        with memory_span(
            "everos.search.rank",
            observation_type="span",
            metadata={"phase": fusion_mode, "kind": "agent_case"},
        ):
            output = await arank(
                RankInput(
                    query=req.query,
                    memory_type=self._case.everalgo_memory_type,  # type: ignore[arg-type]
                    sparse_candidates=sparse,
                    dense_candidates=dense,
                    top_k=top_k,
                    radius=_effective_radius(req),
                ),
                config=RankConfig(fusion_mode=fusion_mode)
                if fusion_mode != "rrf"
                else DEFAULT_RANK_CONFIG,
                llm=self._llm,
                enable_rerank=enable_rerank,
                rerank_top_k=top_k,
            )
        case_candidates = (_scored_as_candidate(s) for s in output.items)
        shaped = (shape_agent_case_from_candidate(c) for c in case_candidates)
        return [item for item in shaped if item is not None]

    # ── Agent skills ────────────────────────────────────────────────

    async def _search_agent_skills(
        self,
        req: SearchRequest,
        where: str,
        *,
        bridge_cases: list[Candidate] | None = None,
    ) -> list[SearchAgentSkillItem]:
        """Rank agent skills. ``bridge_cases`` (reranked case id+score) is
        supplied only on HYBRID + LLM-rerank to feed the case→skill bridge;
        ``None`` everywhere else.
        """
        if req.method == SearchMethod.AGENTIC:
            return await search_agent_skills_agentic(
                req.query,
                where=where,
                skill_recaller=self._skill,
                embed_query_fn=self._embedding.embed,  # type: ignore[union-attr]
                reranker=self._reranker,  # type: ignore[arg-type]
                llm=self._llm,  # type: ignore[arg-type]
                top_k=self._top_k(req.top_k, cap=_AGENT_TOP_K_CAP),
            )
        fusion_mode, _ = resolve_pipeline(req.method, "agent_skill")
        top_k = self._top_k(req.top_k, cap=_AGENT_TOP_K_CAP)

        if fusion_mode is None:
            cands = await self._single_route_recall(
                self._skill, req, where, top_k, cap=_AGENT_TOP_K_CAP
            )
            shaped = (shape_agent_skill_from_candidate(c) for c in cands[:top_k])
            return [item for item in shaped if item is not None]

        sparse, dense, _ = await self._recall_sparse_dense(
            self._skill, req, where, top_k, cap=_AGENT_TOP_K_CAP
        )

        # Case→skill bridge: union skills surfaced via lineage cases into
        # the dense pool with their max-pooled source-case score.
        bridged = await self._case_bridged_skills(bridge_cases, where, top_k)
        dense = _merge_by_id_max(dense, bridged)

        # Lane selection lives here so ``skill_hybrid`` stays single-purpose
        # (cross-encoder) and symmetry with the case path is preserved.
        if _effective_llm_rerank(req):
            # LLM lane: generic ``arank`` dispatches by ``memory_type="skill"``
            # to the skill facade (adds the skill-only 0.4 relevance gate).
            # Config is ``rrf`` — ``skill_hybrid`` is an everos routing
            # label, not an everalgo fusion mode.
            with memory_span(
                "everos.search.rank",
                observation_type="span",
                metadata={"phase": "skill_llm", "kind": "agent_skill"},
            ):
                output = await arank(
                    RankInput(
                        query=req.query,
                        memory_type=self._skill.everalgo_memory_type,  # type: ignore[arg-type]
                        sparse_candidates=sparse,
                        dense_candidates=dense,
                        top_k=top_k,
                        radius=_effective_radius(req),
                    ),
                    config=DEFAULT_RANK_CONFIG,
                    llm=self._llm,
                    enable_rerank=True,
                    rerank_top_k=top_k,
                )
            skill_candidates = (_scored_as_candidate(s) for s in output.items)
            shaped = (shape_agent_skill_from_candidate(c) for c in skill_candidates)
            return [item for item in shaped if item is not None]

        # Cross-encoder lane (default): rrf + skill-shaped cross-encoder rerank.
        # The rank span is emitted inside build_skill_rerank_fn (callbacks),
        # so the cross-encoder rerank is covered uniformly with the agentic
        # path rather than double-wrapped here.
        return await search_agent_skills_hybrid(
            req.query,
            sparse=sparse,
            dense=dense,
            reranker=self._reranker,  # type: ignore[arg-type]
            top_k=top_k,
        )

    # ── Profile ─────────────────────────────────────────────────────

    async def _fetch_profile(self, req: SearchRequest) -> list[SearchProfileItem]:
        if not req.include_profile or req.owner_type != "user":
            return []
        return await self._profile.fetch(req.owner_id)

    # ── Recall helpers ──────────────────────────────────────────────

    async def _single_route_recall(
        self,
        recaller: EpisodeRecaller | AgentCaseRecaller | AgentSkillRecaller,
        req: SearchRequest,
        where: str,
        top_k: int,
        *,
        cap: int = _DEFAULT_TOP_K_CAP,
    ) -> list[Candidate]:
        with memory_span(
            "everos.search.recall",
            observation_type="retriever",
            metadata={"phase": "single_route", "method": req.method.value},
        ):
            if req.method == SearchMethod.KEYWORD:
                return await recaller.sparse_recall(
                    req.query, where, limit=self._recall_limit(req.top_k, cap=cap)
                )
            vector = await self._embed_query(req.query)
            cands = await recaller.dense_recall(
                vector, where, limit=self._recall_limit(req.top_k, cap=cap)
            )
            return self._apply_radius(cands, _effective_radius(req))

    async def _recall_sparse_dense(
        self,
        recaller: EpisodeRecaller | AgentCaseRecaller | AgentSkillRecaller,
        req: SearchRequest,
        where: str,
        top_k: int,
        *,
        cap: int = _DEFAULT_TOP_K_CAP,
    ) -> tuple[list[Candidate], list[Candidate], list[float]]:
        """Fan out keyword + vector recall in parallel.

        The third return is the query embedding itself — the HYBRID
        pipeline passes it into ``facts_for_episodes`` so per-fact
        cosine scoring reuses the same vector instead of re-embedding
        the query. Returns
        ``[]`` for ``vector`` when no embedding provider is configured.
        """
        with memory_span(
            "everos.search.recall",
            observation_type="retriever",
            metadata={"phase": "sparse_dense", "method": req.method.value},
        ):
            vector = await self._embed_query(req.query)
            limit = self._recall_limit(req.top_k, cap=cap)
            sparse, dense = await asyncio.gather(
                recaller.sparse_recall(req.query, where, limit=limit),
                recaller.dense_recall(vector, where, limit=limit)
                if vector
                else _empty_candidates(),
            )
            dense = self._apply_radius(dense, _effective_radius(req))
            return sparse, dense, vector

    async def _maxsim_atomic_recall(
        self, req: SearchRequest, where: str, top_k: int
    ) -> list[Candidate]:
        """MaxSim-style: ANN atomic_facts → max-pool by memcell → batch fetch episodes.

        Trades one extra LanceDB ANN scan (over the ~28× denser
        ``atomic_fact`` table) for finer-grained semantic match — long
        episodes whose single mean-pooled vector dilutes a specific topic
        recover via the matching atomic fact's own embedding. Mirrors
        EverOS/EverAlgo's MaxSim retrieval pattern.
        """
        vector = await self._embed_query(req.query)
        if not vector:
            return []
        fact_limit = min(top_k * _MAXSIM_FACT_MULTIPLIER, _MAXSIM_FACT_POOL_CAP)
        fact_cands = await self._fact.dense_recall(vector, where, limit=fact_limit)
        # Max-pool fact scores by parent episode entry_id.
        ep_score: dict[str, float] = {}
        for fc in fact_cands:
            pid = fc.metadata.get("parent_id")
            if not isinstance(pid, str) or not pid:
                continue
            if fc.score > ep_score.get(pid, -1.0):
                ep_score[pid] = fc.score
        if not ep_score:
            return []
        ranked = sorted(ep_score.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        top_entry_ids = [eid for eid, _ in ranked]
        score_by_entry = dict(ranked)
        ep_cands = await self._ep.fetch_by_entry_ids(top_entry_ids, where)
        rescored: list[Candidate] = []
        for c in ep_cands:
            eid = c.metadata.get("entry_id")
            s = score_by_entry.get(eid, 0.0) if isinstance(eid, str) else 0.0
            rescored.append(
                Candidate(id=c.id, score=s, source="vector", metadata=c.metadata)
            )
        rescored.sort(key=lambda c: c.score, reverse=True)
        return self._apply_radius(rescored, _effective_radius(req))

    async def _case_bridged_skills(
        self,
        bridge_cases: list[Candidate] | None,
        where: str,
        top_k: int,
    ) -> list[Candidate]:
        """Reverse-resolve lineage skills and max-pool their source-case
        scores.

        Reuses ``bridge_cases`` (already-reranked id+score) so ``agent_case``
        is never scanned twice. Scores must be LLM-rerank relevance in
        ``[0, 1]`` to stay comparable with the direct dense pool — never
        feed BM25 / fusion scores in here. Mirrors the ``maxsim_atomic``
        fact→episode pooling. Empty input ⇒ no bridge.
        """
        if not bridge_cases:
            return []
        case_score = {c.id: c.score for c in bridge_cases}
        # Bound the reverse fetch by the matched-case count; one case can map
        # to several skills, so allow a small fan-out per case.
        skill_cands = await self._skill.fetch_by_case_ids(
            list(case_score), where, limit=max(top_k, len(case_score) * 4)
        )
        bridged: list[Candidate] = []
        for sc in skill_cands:
            raw_ids = sc.metadata.get("source_case_ids")
            src_ids = raw_ids if isinstance(raw_ids, list) else []
            best = max(
                (case_score[cid] for cid in src_ids if cid in case_score),
                default=0.0,
            )
            bridged.append(
                Candidate(id=sc.id, score=best, source="vector", metadata=sc.metadata)
            )
        return bridged

    async def _embed_query(self, query: str) -> list[float]:
        if self._embedding is None:
            return []
        vector = await self._embedding.embed(query)
        expected = getattr(self._embedding, "dim", None)
        if expected and len(vector) != expected:
            # Fail here, not inside LanceDB. A mismatched query vector reaches
            # the engine as an opaque `ValueError: Invalid input, No vector
            # column found to match...` after the query has already been set
            # up — measured at 13-14s per request in a soak run, versus
            # microseconds here, and it surfaces as an unhandled 500 with a
            # ~6k-line traceback instead of a named error.
            #
            # ConfigurationError (not InvalidInputError): the caller only ever
            # sends query *text*; the vector is produced by our own provider,
            # so a width that disagrees with the provider's declared ``dim``
            # is a server-side configuration or provider-implementation fault.
            raise ConfigurationError(
                f"embedding provider returned a {len(vector)}-dimension query "
                f"vector but declares dim={expected}; the vector index cannot "
                f"be searched with a mismatched width"
            )
        return vector

    # ── Limits / filters ────────────────────────────────────────────

    @staticmethod
    def _top_k(top_k: int, *, cap: int = _DEFAULT_TOP_K_CAP) -> int:
        """Resolve ``-1`` to ``cap``; pass others through unchanged.

        ``cap`` defaults to the episode/atomic_fact upper bound; agent
        cases / skills pass :data:`_AGENT_TOP_K_CAP` so an unbounded
        request still returns a tight, rerank-friendly result set.
        """
        return cap if top_k == -1 else top_k

    @staticmethod
    def _recall_limit(top_k_request: int, *, cap: int = _DEFAULT_TOP_K_CAP) -> int:
        """Effective recall pool size — branches on the *raw* request value.

        Mirrors enterprise:

        - ``top_k == -1`` (unlimited)  → fixed ``cap``
        - ``top_k > 0``                → ``top_k * DEFAULT_RECALL_MULTIPLIER``

        ``cap`` aligns the unlimited-mode pool with each kind's
        :meth:`_top_k` ceiling (e.g. agent kinds use the tighter
        :data:`_AGENT_TOP_K_CAP`).
        """
        if top_k_request == -1:
            return cap
        return max(
            top_k_request * _DEFAULT_RECALL_MULTIPLIER, _DEFAULT_RECALL_MULTIPLIER
        )

    @staticmethod
    def _apply_radius(cands: list[Candidate], radius: float | None) -> list[Candidate]:
        if radius is None:
            return cands
        return [c for c in cands if c.score >= radius]

    # ── Component guards ────────────────────────────────────────────

    def _validate_components(self, req: SearchRequest) -> None:
        """Fail fast when the chosen method needs a provider that is missing.

        Every provider branch (embedding, rerank, LLM) raises the same
        :class:`ProviderNotConfiguredError` -> HTTP 422 with a toml-pointing
        message; vocabulary is uniform across the three.

        Embedding and rerank checks read BOTH the process-wide capability
        singletons AND this manager's own injected providers. Either side
        being "missing" is enough to fail the guard: accessor
        says-unavailable means the tier never had it; ``self._embedding``
        is ``None`` means whoever built this manager didn't wire one in.
        Because dispatch reads ``self._embedding`` directly, letting an
        accessor-says-available-but-self-is-None case through would
        produce an opaque ``AttributeError('NoneType')`` deep in recall
        instead of a clean 422. Single source of truth = both must agree
        the provider exists.

        Agent HYBRID has two rerank lanes selected by ``enable_llm_rerank``:
        the LLM lane (``True``) reranks via the LLM and needs no rerank
        provider; the cross-encoder lane (``False``, default) does.

        Design contract (PR #361 round-3 review #6): this manager is
        instantiated by ``service.search`` — a single caller that always
        sources its providers from the same capability accessors this
        guard reads. On that path DI-injected provider and accessor
        state cannot disagree, so the belt-and-suspenders check is a
        near-no-op; it only fires on pathological drift (a subclass or
        a test that constructs a manager with mismatched inputs).
        Library consumers that embed :class:`SearchManager` directly
        and hand it their OWN provider will hit 422 even when their
        provider is valid, because the accessor reflects the
        process-wide toml. This is an accepted limitation of the
        current DI shape — the guard prefers false-positives
        (over-refuse) to false-negatives (dispatch with a ``None``
        provider) because the latter surfaces as an opaque
        ``AttributeError('NoneType')`` deep in recall.
        """
        method = req.method
        is_agent_hybrid = method == SearchMethod.HYBRID and req.owner_type == "agent"

        needs_embedding = method in (
            SearchMethod.VECTOR,
            SearchMethod.HYBRID,
            SearchMethod.AGENTIC,
        )
        if needs_embedding and (
            not get_embedding_capability().available or self._embedding is None
        ):
            raise ProviderNotConfiguredError(
                provider="embedding",
                feature=_feature_name(method, req.owner_type),
            )

        # agent HYBRID cross-encoder lane (enable_llm_rerank=False, the
        # default) reaches ``search_agent_skills_hybrid``, which needs a
        # real rerank provider; the LLM lane reranks via the LLM instead
        # and is exempt.
        if (
            is_agent_hybrid
            and not req.enable_llm_rerank
            and (not get_rerank_capability().available or self._reranker is None)
        ):
            raise ProviderNotConfiguredError(
                provider="rerank",
                feature="agent_hybrid",
                alternative_hint=(
                    "Set enable_llm_rerank=true to route through the "
                    "LLM rerank lane instead of the cross-encoder "
                    "provider."
                ),
            )

        if method == SearchMethod.AGENTIC and (
            not get_rerank_capability().available or self._reranker is None
        ):
            raise ProviderNotConfiguredError(
                provider="rerank", feature="agentic_search"
            )

        # LLM has no Capability wrapper yet; guard on the injected client
        # directly. Belt-and-suspenders with the embedding/rerank branches
        # above -- same 422 vocabulary via ProviderNotConfiguredError.
        is_hybrid_llm_lane = method == SearchMethod.HYBRID and req.enable_llm_rerank
        if is_hybrid_llm_lane and self._llm is None:
            raise ProviderNotConfiguredError(
                provider="llm",
                feature=_feature_name(method, req.owner_type),
                alternative_hint=(
                    "drop the enable_llm_rerank flag to use the cross-encoder "
                    "rerank lane (needs a configured [rerank] provider)"
                ),
            )
        if method == SearchMethod.AGENTIC and self._llm is None:
            raise ProviderNotConfiguredError(
                provider="llm",
                feature=_feature_name(method, req.owner_type),
            )


def _scored_as_candidate(scored) -> Candidate:  # type: ignore[no-untyped-def]
    """Adapt a single-type ``ScoredItem`` back to a ``Candidate``.

    Adapts ``ScoredItem`` back to ``Candidate`` so the existing
    Candidate-based shapers apply.
    """
    return Candidate(
        id=scored.id,
        score=scored.score,
        source="other",
        metadata=dict(scored.metadata),
    )


def _feature_name(method: SearchMethod, owner_type: str) -> str:
    """Map a search method (+ owner partition) to its 422 ``feature`` tag.

    HYBRID splits by ``owner_type`` because the episode path (embed only)
    and the agent path (embed, plus a rerank lane) have different
    prerequisites and deserve distinct log/response tags.

    Note: this vocabulary (``"vector"``, ``"user_hybrid"``, ``"agent_hybrid"``,
    ``"agentic_search"``) is separate from the capability-level vocabulary
    in :mod:`everos.component.capabilities`. See that module's docstring
    for the rationale and contract.
    """
    if method == SearchMethod.VECTOR:
        return "vector"
    if method == SearchMethod.HYBRID:
        return "agent_hybrid" if owner_type == "agent" else "user_hybrid"
    if method == SearchMethod.AGENTIC:
        return "agentic_search"
    return method.value


def _effective_llm_rerank(req: SearchRequest) -> bool:
    """LLM Phase-5 rerank only kicks in for ``HYBRID`` and only when the
    caller opts in. ``AGENTIC`` runs its own cross-encoder rerank loop
    (via ``rerank_fn``) and intentionally skips Phase-5.
    """
    return req.method == SearchMethod.HYBRID and req.enable_llm_rerank


def _effective_radius(req: SearchRequest) -> float | None:
    """Resolve the cosine-similarity threshold actually applied to dense hits.

    Priority:

    1. Caller-supplied ``req.radius`` always wins (including ``0.0`` when
       they explicitly want everything).
    2. Otherwise, ``top_k == -1`` (unlimited) defaults to
       ``_DEFAULT_UNLIMITED_RADIUS`` so the response keeps a quality
       floor — matches enterprise's auto-default behaviour.
    3. Otherwise (normal ``top_k > 0`` mode), return ``None`` and trust
       truncation to handle tail quality.
    """
    if req.radius is not None:
        return req.radius
    if req.top_k == -1:
        return _DEFAULT_UNLIMITED_RADIUS
    return None


async def _empty_candidates() -> list[Candidate]:
    return []


def _extract_top_level_session_id(filters: FilterNode | None) -> str | None:
    """Return the literal value of a top-level ``session_id`` eq scalar.

    The unprocessed-buffer trigger only fires for the simple shape
    ``filters = {"session_id": "<sid>"}``. Anything wrapped in ``AND`` /
    ``OR``, nested deeper, or expressed via an operator map (``{"eq":
    ...}``, ``{"in": ...}``) is treated as "session not pinned" — there
    is no defensible buffer-scope mapping for those compound predicates.
    """
    if filters is None:
        return None
    extra = filters.__pydantic_extra__ or {}
    value = extra.get("session_id")
    return value if isinstance(value, str) and value else None


def _unprocessed_buffer_to_dto(row: UnprocessedBuffer) -> UnprocessedMessageDTO:
    """Render one ``unprocessed_buffer`` row as its public DTO.

    Mirrors :class:`MessageItemDTO`'s ``content`` shorthand: a single-item
    ``[{"type":"text","text":...}]`` payload collapses to the inner string;
    every other shape stays as the opaque ``list[dict]`` so multimodal
    payloads round-trip without lossy flattening.
    """
    content_items = json.loads(row.content_items_json)
    if (
        isinstance(content_items, list)
        and len(content_items) == 1
        and isinstance(content_items[0], dict)
        and content_items[0].get("type") == "text"
        and isinstance(content_items[0].get("text"), str)
    ):
        content: str | list[dict[str, object]] = content_items[0]["text"]
    else:
        content = content_items
    tool_calls = (
        json.loads(row.tool_calls_json) if row.tool_calls_json is not None else None
    )
    return UnprocessedMessageDTO(
        id=row.message_id,
        app_id=row.app_id,
        project_id=row.project_id,
        session_id=row.session_id,
        sender_id=row.sender_id,
        sender_name=row.sender_name,
        role=row.role,  # type: ignore[arg-type]
        content=content,
        timestamp=to_display_tz(row.timestamp),
        tool_calls=tool_calls,
        tool_call_id=row.tool_call_id,
    )


def _merge_by_id_max(
    primary: list[Candidate], extra: list[Candidate]
) -> list[Candidate]:
    """Union by id, keep higher score. Folds bridged skills into the dense
    pool so downstream fusion doesn't double-count overlap.
    """
    by_id: dict[str, Candidate] = {c.id: c for c in primary}
    for c in extra:
        existing = by_id.get(c.id)
        if existing is None or c.score > existing.score:
            by_id[c.id] = c
    return list(by_id.values())


_ = Sequence  # quiet unused-import for the typing-only annotation above
