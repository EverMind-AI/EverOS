"""SearchManager — top-level orchestrator for ``POST /api/v2/memory/search``.

Hard partition by ``owner_type``:

* ``user``  → ``episodes`` (+ ``profiles`` when ``include_profile=true``)
* ``agent`` → ``agent_cases`` + ``agent_skills``

Per kind, :func:`memory.search.adapter.resolve_pipeline` decides whether
the path is "single-route recall, no fusion" (``KEYWORD`` / ``VECTOR``)
or "sparse + dense → everalgo.rank" (``HYBRID`` / ``AGENTIC``). Component
guards (embedding / cross-encoder / LLM) raise early when a method is
selected without its prerequisites.

Episode ``HYBRID`` recalls three lanes — BM25, episode vectors, and the
atomic-fact MaxSim lane — and ranks them with :mod:`memory.search.hierarchy`
(two RRF fusions, then LR-calibrated scoring); ``top_k`` counts episodes. It
has **no LLM rerank**: ``enable_llm_rerank`` is **ignored** there and only
applies to agent fusion. ``AGENTIC`` keeps its own internal cross-encoder
rerank loop; the flag is ignored there.

``SearchEpisodeItem.atomic_facts`` is populated **only** when the HYBRID
pipeline runs over episodes. The other methods leave it empty: there is
no query-relevance score we can assign to a fact pulled by parent_id
alone, and emitting ``score=0.0`` facts would muddy the contract.

The manager never writes to storage; it only reads LanceDB + markdown.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import UTC, tzinfo
from typing import TYPE_CHECKING, Any, NamedTuple

from everalgo.rank import DEFAULT_RANK_CONFIG, RankConfig, arank

from everos.config import load_settings
from everos.config.settings import DeciderSettings
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
from everos.core.utils.datetime import to_display_tz
from everos.memory.models import CanonicalMessage
from everos.memory.ports import ClusterStore, Filter, SessionBufferStore, SessionKey

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
from .hierarchy import build_ep_to_fact_parents, fuse_lanes, score_episodes
from .llm_multiround import RoundDecider, search_episodes_llm_multiround
from .recall import (
    KindRecaller,
    RecallResult,
    maxsim_episode_recall,
    merge_by_id_max,
    merge_stored,
    rank_input,
)
from .shaper import (
    reshape_hybrid_output,
    shape_agent_cases,
    shape_agent_cases_ranked,
    shape_agent_skills,
    shape_agent_skills_ranked,
    shape_episodes,
)
from .skill_hybrid import search_agent_skills_hybrid
from .trace import record_branch, record_stage

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

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

# Agent HYBRID lane size in unlimited mode: each lane recalls 50 so the
# rerank has a pool to pick the 10 from. ``top_k > 0`` recalls ``top_k * 2``.
_AGENT_HYBRID_LANE_CAP = 50

# Vector ``radius`` (cosine similarity threshold) default for **unlimited
# mode only**. In ``top_k > 0`` mode we trust the truncation cap to ditch
# low-quality tail; in ``top_k = -1`` mode we would otherwise return up to
# 100 candidates with no quality floor, so we layer a default 0.5
# similarity threshold the way enterprise does (enterprise uses 0.6 — we
# pick 0.5 slightly looser because LanceDB cosine vs Milvus cosine score
# distributions can drift a bit on the same model).
_DEFAULT_UNLIMITED_RADIUS = 0.5

# ``maxsim_atomic`` (VECTOR) sizing: 200 atomic facts max-pooled onto at most
# ``top_k`` episodes, 50 in unlimited mode.
_MAXSIM_FACT_POOL = 200
_MAXSIM_TOP_K_CAP = 50

# Episode HYBRID lane sizes. Each episode lane recalls ``max(top_k * 2, 50)``;
# the atomic-fact lane max-pools 200 facts onto at most 50 episodes; scoring
# reads up to 20 facts per kept episode, of which it uses the best.
_HYBRID_LANE_FLOOR = 50
_HYBRID_FACT_POOL = 200
_HYBRID_FACT_LANE_EPISODES = 50
_HYBRID_FACTS_PER_EPISODE = 20

# The unprocessed buffer (``memorize`` track) is a single
# shared track because boundary detection is single-pass — switching mode
# requires a fresh process. Hard-coded here (instead of importing) to keep
# the memory layer free of service-layer imports per the DDD direction rule.


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
        session_buffer: SessionBufferStore,
        cluster_store: ClusterStore,
        embedding: EmbeddingProvider | None,
        reranker: RerankProvider | None,
        llm_client: LLMClient,
        decider_client: LLMClient,
        search_tokenizer: Tokenizer | None = None,
        tuning_source: Callable[[], DeciderSettings],
        trace_path_source: Callable[[], str | None],
        tz: tzinfo = UTC,
    ) -> None:
        self._tuning_source = tuning_source
        self._trace_path_source = trace_path_source
        self._ep = episode_recaller
        self._fact = atomic_fact_recaller
        self._case = agent_case_recaller
        self._skill = agent_skill_recaller
        self._profile = profile_recaller
        self._session_buffer = session_buffer
        self._clusters = cluster_store
        self._embedding = embedding
        self._reranker = reranker
        self._llm = llm_client
        # Multi-round retrieval runs its own client ([llm.decider]): it inherits
        # [llm] field by field but never shares another task's SDK retries.
        self._decider_llm = decider_client
        self._search_tokenizer = search_tokenizer
        # The zone the agentic sufficiency prompt shows candidate dates in.
        self._tz = tz

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
            record_branch(
                "_load_unprocessed",
                "unprocessed",
                [],
                status="skipped",
                skip_reason="no_session_filter",
            )
            return []
        session = SessionKey(
            app_id=req.app_id, project_id=req.project_id, session_id=session_id
        )
        try:
            messages = await self._session_buffer.list(session)
            results = [_unprocessed_message_to_dto(m, session) for m in messages]
        except Exception as exc:
            record_branch(
                "_load_unprocessed",
                "unprocessed",
                [],
                status="error",
                session_id=session_id,
                error_type=type(exc).__name__,
            )
            raise
        record_branch(
            "_load_unprocessed", "unprocessed", results, session_id=session_id
        )
        return results

    # ── Agent partition ─────────────────────────────────────────────

    async def _search_cases_and_skills(
        self, req: SearchRequest, where: Filter
    ) -> tuple[list[SearchAgentCaseItem], list[SearchAgentSkillItem]]:
        """Cases + skills, serial when bridging.

        HYBRID + LLM rerank runs serially: reranked cases feed the
        skill bridge. Every other method runs the two kinds in parallel
        with no bridge — the bridge only pays off after rerank has
        produced high-quality case scores to inherit.
        """
        if _effective_llm_rerank(req):
            cases = await self._search_agent_cases(req, where)
            skills = await self._search_agent_skills(
                req, where, case_scores={c.entry_id: c.score for c in cases}
            )
            return cases, skills

        cases, skills = await asyncio.gather(
            self._search_agent_cases(req, where),
            self._search_agent_skills(req, where),
        )
        return cases, skills

    # ── Episodes ────────────────────────────────────────────────────

    async def search_episodes_with_decider(
        self, req: SearchRequest, decider: RoundDecider
    ) -> list[SearchEpisodeItem]:
        """Run the ``llm_multiround`` loop with an injected decider.

        This is the seam a Phase-2 RL policy occupies: the loop, its retrieval,
        its block rendering and its stop conditions are EverOS's own, and only
        the per-round decision comes from the caller. An RL environment that
        re-implements the loop instead drifts: measured against this one, a
        hand-built environment returned a different candidate set on 25/25
        sampled sub-queries (Jaccard median 0.538), because the public search
        route dispatches to the hybrid hierarchy pipeline rather than to
        ``llm_multiround``'s per-sub-query ``rrf(sparse, dense)[:topk]``.
        """
        req = req.model_copy(update={"method": SearchMethod.LLM_MULTIROUND})
        self._validate_components(req)
        where = compile_filters(
            req.filters,
            owner_type=req.owner_type,
            owner_id=req.owner_id,
            app_id=req.app_id,
            project_id=req.project_id,
        )
        return await self._search_episodes(req, where, decider=decider)

    async def _search_episodes(
        self,
        req: SearchRequest,
        where: Filter,
        *,
        decider: RoundDecider | None = None,
    ) -> list[SearchEpisodeItem]:
        if req.method == SearchMethod.AGENTIC:
            results = await search_episodes_agentic(
                req.query,
                owner_id=req.owner_id,
                where=where,
                app_id=req.app_id,
                project_id=req.project_id,
                episode_recaller=self._ep,
                atomic_fact_recaller=self._fact,
                cluster_store=self._clusters,
                embed_query_fn=self._embedding.embed_query,  # type: ignore[union-attr]
                reranker=self._reranker,  # type: ignore[arg-type]
                llm=self._llm,  # type: ignore[arg-type]
                top_k=self._top_k(req.top_k),
                radius=_effective_radius(req),
                tz=self._tz,
            )
            record_stage("agentic", "episode", results)
            return results

        if req.method == SearchMethod.LLM_MULTIROUND:
            results = await search_episodes_llm_multiround(
                req.query,
                owner_id=req.owner_id,
                where=where,
                app_id=req.app_id,
                project_id=req.project_id,
                episode_recaller=self._ep,
                atomic_fact_recaller=self._fact,
                embed_query_fn=self._embedding.embed_query,  # type: ignore[union-attr]
                llm=self._decider_llm,  # type: ignore[arg-type]
                top_k=self._top_k(req.top_k),
                reranker=self._reranker,  # type: ignore[arg-type]
                decider=decider,
                tuning_source=self._tuning_source,
                trace_path_source=self._trace_path_source,
            )
            record_stage("agentic", "episode", results)
            return results

        fusion_mode, _ = resolve_pipeline(req.method, "episode")
        top_k = self._top_k(
            req.top_k,
            cap=_MAXSIM_TOP_K_CAP
            if req.method == SearchMethod.VECTOR
            else _DEFAULT_TOP_K_CAP,
        )

        # ── KEYWORD / VECTOR: single-route recall ──
        if fusion_mode is None:
            with memory_span(
                "everos.search.recall",
                observation_type="retriever",
                metadata={"phase": "single_route", "method": req.method.value},
            ):
                if req.method == SearchMethod.KEYWORD:
                    recalled = await self._ep.sparse_recall(
                        req.query, where, limit=self._recall_limit(req.top_k)
                    )
                else:
                    recalled = await self._maxsim_atomic_recall(req, where, top_k)
            if req.method == SearchMethod.KEYWORD:
                record_stage("single_route", "episode", recalled.candidates)
            # ``atomic_facts`` stays empty: facts come back only when the HYBRID
            # pipeline surfaces them with a score (see ``reshape_hybrid_output``).
            # Single-route recall has no per-fact score against the query, so
            # we do not back-fill — that would emit ``score=0.0`` facts whose
            # semantics are ambiguous.
            return shape_episodes(recalled.candidates[:top_k], recalled.stored_by_id)

        # ── HYBRID: three recall lanes ranked by ``hierarchy`` ──
        shaped = await self._search_episodes_hybrid(req, where, top_k)
        if req.min_score is not None:
            shaped = [s for s in shaped if s.score >= req.min_score]
            record_stage("filtered", "episode", shaped)
        return shaped

    async def _search_episodes_hybrid(
        self, req: SearchRequest, where: Filter, top_k: int
    ) -> list[SearchEpisodeItem]:
        """Recall the three episode lanes and rank them (see ``hierarchy``).

        ``radius`` also floors the facts each kept episode is scored on, so a
        fact below it neither lifts the episode's score nor is attached.
        """
        lanes = await self._hybrid_lanes(req, where, top_k)
        with memory_span(
            "everos.search.rank",
            observation_type="span",
            metadata={"phase": "hierarchy"},
        ):
            kept = fuse_lanes(
                lanes.sparse.candidates,
                lanes.dense.candidates,
                lanes.facts.candidates,
                top_k=top_k,
            )
            record_stage("rrf", "episode", kept)
            facts = await self._fact.facts_for_episodes(
                build_ep_to_fact_parents(kept),
                where,
                per_episode=_HYBRID_FACTS_PER_EPISODE,
                query_vector=lanes.vector,
            )
            facts = facts.within_radius(_effective_radius(req))
            scored = score_episodes(
                kept,
                facts.buckets,
                sparse=lanes.sparse.candidates,
                dense=lanes.dense.candidates,
                fact_lane=lanes.facts.candidates,
            )
            record_stage("hierarchy", "episode", scored)
        return reshape_hybrid_output(
            scored,
            episodes_by_id=merge_stored(lanes.sparse, lanes.dense, lanes.facts),
            facts_by_id=facts.stored_by_id,
        )

    async def _hybrid_lanes(
        self, req: SearchRequest, where: Filter, top_k: int
    ) -> _HybridLanes:
        """BM25, episode-vector and atomic-fact lanes, recalled in parallel.

        ``radius`` floors both vector lanes; the fact lane on the max-pooled
        fact cosine, as the VECTOR route does. With no fact in scope the fact
        lane is empty rather than a copy of the episode-vector lane.
        """
        with memory_span(
            "everos.search.recall",
            observation_type="retriever",
            metadata={"phase": "hybrid_lanes", "method": req.method.value},
        ):
            vector = await self._embed_query(req.query)
            pool = max(top_k * 2, _HYBRID_LANE_FLOOR)
            sparse, dense, facts = await asyncio.gather(
                self._ep.sparse_recall(req.query, where, limit=pool),
                self._ep.dense_recall(vector, where, limit=pool)
                if vector
                else _no_recall(),
                maxsim_episode_recall(
                    vector,
                    where,
                    facts=self._fact,
                    episodes=self._ep,
                    top_k=_HYBRID_FACT_LANE_EPISODES,
                    fact_limit=_HYBRID_FACT_POOL,
                    fallback_limit=None,
                    owner_id=req.owner_id,
                ),
            )
        radius = _effective_radius(req)
        dense = dense.within_radius(radius)
        facts = facts.within_radius(radius)
        record_stage("sparse", "episode", sparse.candidates)
        record_stage("dense", "episode", dense.candidates)
        return _HybridLanes(
            sparse=sparse,
            dense=dense,
            facts=facts,
            vector=vector,
        )

    # ── Agent cases ─────────────────────────────────────────────────

    async def _search_agent_cases(
        self, req: SearchRequest, where: Filter
    ) -> list[SearchAgentCaseItem]:
        if req.method == SearchMethod.AGENTIC:
            results = await search_agent_cases_agentic(
                req.query,
                where=where,
                case_recaller=self._case,
                embed_query_fn=self._embedding.embed_query,  # type: ignore[union-attr]
                reranker=self._reranker,  # type: ignore[arg-type]
                llm=self._llm,  # type: ignore[arg-type]
                top_k=self._top_k(req.top_k, cap=_AGENT_TOP_K_CAP),
                radius=_effective_radius(req),
                tz=self._tz,
            )
            record_stage("agentic", "agent_case", results)
            return results
        fusion_mode, _ = resolve_pipeline(req.method, "agent_case")
        enable_rerank = _effective_llm_rerank(req)
        top_k = self._top_k(req.top_k, cap=_AGENT_TOP_K_CAP)

        if fusion_mode is None:
            recalled = await self._single_route_recall(
                self._case, req, where, top_k, cap=_AGENT_TOP_K_CAP
            )
            return shape_agent_cases(recalled.candidates[:top_k], recalled.stored_by_id)

        sparse, dense = await self._recall_sparse_dense(
            self._case, req, where, top_k, cap=_AGENT_HYBRID_LANE_CAP
        )
        with memory_span(
            "everos.search.rank",
            observation_type="span",
            metadata={"phase": fusion_mode, "kind": "agent_case"},
        ):
            output = await arank(
                rank_input(
                    req.query,
                    memory_type=self._case.everalgo_memory_type,
                    sparse=sparse,
                    dense=dense,
                    top_k=top_k,
                ),
                config=RankConfig(fusion_mode=fusion_mode)
                if fusion_mode != "rrf"
                else DEFAULT_RANK_CONFIG,
                llm=self._llm,
                enable_rerank=enable_rerank,
                rerank_top_k=top_k,
            )
        record_stage(
            "fusion",
            "agent_case",
            output.items,
            metadata={"fusion_mode": fusion_mode, "enable_rerank": enable_rerank},
        )
        return shape_agent_cases_ranked(output, merge_stored(sparse, dense))

    # ── Agent skills ────────────────────────────────────────────────

    async def _search_agent_skills(
        self,
        req: SearchRequest,
        where: Filter,
        *,
        case_scores: Mapping[str, float] | None = None,
    ) -> list[SearchAgentSkillItem]:
        """Rank agent skills. ``case_scores`` (case entry id -> reranked
        score) is supplied only on HYBRID + LLM-rerank to feed the case→skill
        bridge; ``None`` everywhere else. Entry ids, not index row ids: a
        skill's ``source_case_ids`` holds the md-side entry ids.
        """
        if req.method == SearchMethod.AGENTIC:
            results = await search_agent_skills_agentic(
                req.query,
                where=where,
                skill_recaller=self._skill,
                embed_query_fn=self._embedding.embed_query,  # type: ignore[union-attr]
                reranker=self._reranker,  # type: ignore[arg-type]
                llm=self._llm,  # type: ignore[arg-type]
                top_k=self._top_k(req.top_k, cap=_AGENT_TOP_K_CAP),
                radius=_effective_radius(req),
                tz=self._tz,
            )
            record_stage("agentic", "agent_skill", results)
            return results
        fusion_mode, _ = resolve_pipeline(req.method, "agent_skill")
        top_k = self._top_k(req.top_k, cap=_AGENT_TOP_K_CAP)

        if fusion_mode is None:
            recalled = await self._single_route_recall(
                self._skill, req, where, top_k, cap=_AGENT_TOP_K_CAP
            )
            return shape_agent_skills(
                recalled.candidates[:top_k], recalled.stored_by_id
            )

        sparse, dense = await self._recall_sparse_dense(
            self._skill, req, where, top_k, cap=_AGENT_HYBRID_LANE_CAP
        )

        # Case→skill bridge: union skills surfaced via lineage cases into
        # the dense pool with their max-pooled source-case score.
        if case_scores:
            bridged = await self._skill.bridged_from_cases(
                case_scores, where, top_k=top_k
            )
            dense = merge_by_id_max(dense, bridged)
        bridge_inputs = [("agent_skill", "dense")]
        if case_scores is not None:
            bridge_inputs.append(("agent_case", "fusion"))
        record_stage(
            "bridged_dense", "agent_skill", dense.candidates, inputs=bridge_inputs
        )

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
                    rank_input(
                        req.query,
                        memory_type=self._skill.everalgo_memory_type,
                        sparse=sparse,
                        dense=dense,
                        top_k=top_k,
                    ),
                    config=DEFAULT_RANK_CONFIG,
                    llm=self._llm,
                    enable_rerank=True,
                    rerank_top_k=top_k,
                )
            record_stage(
                "fusion",
                "agent_skill",
                output.items,
                metadata={"fusion_mode": "rrf", "enable_rerank": True},
                inputs=[
                    ("agent_skill", "sparse"),
                    ("agent_skill", "bridged_dense"),
                ],
            )
            return shape_agent_skills_ranked(output, merge_stored(sparse, dense))

        # Cross-encoder lane (default): rrf + skill-shaped cross-encoder rerank.
        # The rank span is emitted inside build_skill_rerank_fn (callbacks),
        # so the cross-encoder rerank is covered uniformly with the agentic
        # path rather than double-wrapped here.
        results = await search_agent_skills_hybrid(
            req.query,
            sparse=sparse,
            dense=dense,
            reranker=self._reranker,  # type: ignore[arg-type]
            top_k=top_k,
            rerank_pool=self._recall_limit(req.top_k, cap=_AGENT_HYBRID_LANE_CAP),
        )
        record_stage(
            "fusion",
            "agent_skill",
            results,
            metadata={"fusion_mode": "skill_hybrid", "enable_rerank": False},
            inputs=[("agent_skill", "sparse"), ("agent_skill", "bridged_dense")],
        )
        return results

    # ── Profile ─────────────────────────────────────────────────────

    async def _fetch_profile(self, req: SearchRequest) -> list[SearchProfileItem]:
        if not req.include_profile or req.owner_type != "user":
            return []
        return await self._profile.fetch(
            req.owner_id, app_id=req.app_id, project_id=req.project_id
        )

    # ── Recall helpers ──────────────────────────────────────────────

    async def _single_route_recall(
        self,
        recaller: KindRecaller,
        req: SearchRequest,
        where: Filter,
        top_k: int,
        *,
        cap: int = _DEFAULT_TOP_K_CAP,
    ) -> RecallResult[Any]:
        with memory_span(
            "everos.search.recall",
            observation_type="retriever",
            metadata={"phase": "single_route", "method": req.method.value},
        ):
            if req.method == SearchMethod.KEYWORD:
                recalled = await recaller.sparse_recall(
                    req.query, where, limit=self._recall_limit(req.top_k, cap=cap)
                )
            else:
                vector = await self._embed_query(req.query)
                recalled = await recaller.dense_recall(
                    vector, where, limit=self._recall_limit(req.top_k, cap=cap)
                )
            recalled = recalled.within_radius(_effective_radius(req))
            record_stage("single_route", recaller.kind, recalled.candidates)
            return recalled

    async def _recall_sparse_dense(
        self,
        recaller: KindRecaller,
        req: SearchRequest,
        where: Filter,
        top_k: int,
        *,
        cap: int = _DEFAULT_TOP_K_CAP,
    ) -> tuple[RecallResult[Any], RecallResult[Any]]:
        """Fan out keyword + vector recall in parallel.

        Dense recall is empty when no embedding provider is configured.
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
                else _no_recall(),
            )
            dense = dense.within_radius(_effective_radius(req))
            record_stage("sparse", recaller.kind, sparse.candidates)
            record_stage("dense", recaller.kind, dense.candidates)
            return sparse, dense

    async def _maxsim_atomic_recall(
        self, req: SearchRequest, where: Filter, top_k: int
    ) -> RecallResult[Any]:
        """MaxSim recall: atomic-fact ANN max-pooled onto parent episodes."""
        recalled = await maxsim_episode_recall(
            await self._embed_query(req.query),
            where,
            facts=self._fact,
            episodes=self._ep,
            top_k=top_k,
            fact_limit=_MAXSIM_FACT_POOL,
            fallback_limit=self._recall_limit(req.top_k),
            owner_id=req.owner_id,
        )
        recalled = recalled.within_radius(_effective_radius(req))
        record_stage("maxsim_atomic", "episode", recalled.candidates)
        return recalled

    async def _embed_query(self, query: str) -> list[float]:
        if self._embedding is None:
            return []
        vector = await self._embedding.embed_query(query)
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

    # ── Component guards ────────────────────────────────────────────

    def _validate_components(self, req: SearchRequest) -> None:
        """Fail fast when the chosen method needs a provider that is missing.

        Every provider branch (embedding, rerank) raises the same
        :class:`ProviderNotConfiguredError` -> HTTP 422 with a toml-pointing
        message; vocabulary is uniform across both. The LLM is never
        checked: the application is not built without one.

        Each check reads the provider this manager was built with: a
        ``None`` provider is an unconfigured tier, and refusing it here gives
        a clean 422 instead of an opaque ``AttributeError('NoneType')`` deep
        in recall.

        Agent HYBRID has two rerank lanes selected by ``enable_llm_rerank``:
        the LLM lane (``True``) reranks via the LLM and needs no rerank
        provider; the cross-encoder lane (``False``, default) does.
        """
        method = req.method
        is_agent_hybrid = method == SearchMethod.HYBRID and req.owner_type == "agent"

        needs_embedding = method in (
            SearchMethod.VECTOR,
            SearchMethod.HYBRID,
            SearchMethod.AGENTIC,
            SearchMethod.LLM_MULTIROUND,
        )
        if needs_embedding and self._embedding is None:
            raise ProviderNotConfiguredError(
                provider="embedding",
                feature=_feature_name(method, req.owner_type),
            )

        if method == SearchMethod.LLM_MULTIROUND and req.owner_type == "agent":
            raise RuntimeError(
                "method='llm_multiround' is only supported for user memory; "
                "provide user_id instead of agent_id"
            )

        # agent HYBRID cross-encoder lane (enable_llm_rerank=False, the
        # default) reaches ``search_agent_skills_hybrid``, which needs a
        # real rerank provider; the LLM lane reranks via the LLM instead
        # and is exempt.
        if is_agent_hybrid and not req.enable_llm_rerank and self._reranker is None:
            raise ProviderNotConfiguredError(
                provider="rerank",
                feature="agent_hybrid",
                alternative_hint=(
                    "Set enable_llm_rerank=true to route through the "
                    "LLM rerank lane instead of the cross-encoder "
                    "provider."
                ),
            )

        if method == SearchMethod.AGENTIC and self._reranker is None:
            raise ProviderNotConfiguredError(
                provider="rerank", feature="agentic_search"
            )


def _feature_name(method: SearchMethod, owner_type: str) -> str:
    """Map a search method (+ owner partition) to its 422 ``feature`` tag.

    HYBRID splits by ``owner_type`` because the episode path (embed only)
    and the agent path (embed, plus a rerank lane) have different
    prerequisites and deserve distinct log/response tags.

    Note: this vocabulary (``"vector"``, ``"user_hybrid"``, ``"agent_hybrid"``,
    ``"agentic_search"``) is separate from the capability-level vocabulary
    of :func:`everos.service.compute_disabled_features`. See its module docstring
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


def _unprocessed_message_to_dto(
    message: CanonicalMessage, session: SessionKey
) -> UnprocessedMessageDTO:
    """Render one buffered message as its public DTO.

    Mirrors :class:`MessageItemDTO`'s ``content`` shorthand: a single-item
    ``[{"type":"text","text":...}]`` payload collapses to the inner string;
    every other shape stays as the opaque ``list[dict]`` so multimodal
    payloads round-trip without lossy flattening.
    """
    content_items = message.content_items
    if (
        len(content_items) == 1
        and isinstance(content_items[0], dict)
        and content_items[0].get("type") == "text"
        and isinstance(content_items[0].get("text"), str)
    ):
        content: str | list[dict[str, object]] = content_items[0]["text"]
    else:
        content = content_items
    tool_calls = (
        [tc.model_dump() for tc in message.tool_calls]
        if message.tool_calls is not None
        else None
    )
    return UnprocessedMessageDTO(
        id=message.message_id,
        app_id=session.app_id,
        project_id=session.project_id,
        session_id=message.session_id,
        sender_id=message.sender_id,
        sender_name=message.sender_name,
        role=message.role,
        content=content,
        timestamp=to_display_tz(message.timestamp),
        tool_calls=tool_calls,
        tool_call_id=message.tool_call_id,
    )


class _HybridLanes(NamedTuple):
    """The three episode HYBRID recall lanes and the query vector."""

    sparse: RecallResult[Any]
    dense: RecallResult[Any]
    facts: RecallResult[Any]
    vector: list[float]


async def _no_recall() -> RecallResult[Any]:
    return RecallResult.empty()
