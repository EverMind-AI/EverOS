"""SearchManager — top-level orchestrator for ``POST /api/v2/memory/search``.

Hard partition by ``owner_type``:

* ``user``  → ``episodes`` (+ ``profiles`` when ``include_profile=true``)
* ``agent`` → ``agent_cases`` + ``agent_skills``
* ``group`` → ``episodes`` and pictures of one or several groups, as one pool
  (+ the asker's and named members' member-in-group ``profiles`` when
  ``include_profile=true``)

Every request first goes through the access policy
(:mod:`everos.memory.access`): its read scope's predicate replaces the owner
part of every recall's ``where``, so visibility is decided in the index query
and no ranking -- the multi-round decider included -- sees a candidate the
asker may not. A group scope recalls no atomic facts (there are none, and the
fact table has no window columns) and attaches no personal profile. Its
multi-round question names the asker (``"<display name>: <question>"``,
section 8.2), and ``include_profile`` attaches member-in-group profiles only.

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
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from everalgo.rank import DEFAULT_RANK_CONFIG, RankConfig, arank
from everalgo.rank.fusion import rrf

from everos.config import load_settings
from everos.config.settings import DeciderSettings
from everos.core.context import resolve_request_id
from everos.core.errors import (
    ConfigurationError,
    InvalidInputError,
    ProviderNotConfiguredError,
)
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import (
    capture_input,
    capture_output,
    current_trace_ids,
    emit_recall_scores,
    memory_span,
)
from everos.core.utils.datetime import to_display_tz
from everos.memory.access import (
    AccessPolicy,
    ReadScope,
    ScopeRequest,
    resolve_identity,
)
from everos.memory.groups import named_members, roster_names
from everos.memory.models import CanonicalMessage, ImageMemory
from everos.memory.ports import (
    ClusterStore,
    Filter,
    Scope,
    SessionBufferStore,
    SessionKey,
    Stored,
    all_of,
    eq,
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
    SearchImageItem,
    SearchMethod,
    SearchProfileItem,
    SearchRequest,
    SearchResponse,
    SearchScope,
    SearchWindow,
    UnprocessedMessageDTO,
)
from .filters import compile_scoped
from .hierarchy import build_ep_to_fact_parents, heap_expand
from .llm_multiround import (
    RoundDecider,
    search_episodes_and_images_llm_multiround,
)
from .recall import (
    KindRecaller,
    NoAtomicFacts,
    RecallResult,
    maxsim_episode_recall,
    merge_by_id_max,
    merge_stored,
    rank_input,
)
from .shaper import (
    attach_referenced_pictures,
    reshape_hybrid_output,
    shape_agent_cases,
    shape_agent_cases_ranked,
    shape_agent_skills,
    shape_agent_skills_ranked,
    shape_episodes,
    shape_images,
)
from .skill_hybrid import search_agent_skills_hybrid

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

    from everos.component.embedding import EmbeddingProvider, VisionEmbeddingProvider
    from everos.component.rerank import RerankProvider
    from everos.component.tokenizer import Tokenizer
    from everos.memory.ports.images import ImageStore

    from .recall import (
        AgentCaseRecaller,
        AgentSkillRecaller,
        AtomicFactRecaller,
        EpisodeRecaller,
        ImageMemoryRecaller,
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

# Members a group search attaches the profiles of, besides the asker's, when
# the query names them (docs/group-memory.md, section 5.3).
_NAMED_PROFILES_CAP = 3

# ``maxsim_atomic`` recall pool sizing — atomic facts are ~28× denser than
# episodes (one memcell → 1 episode + ~28 atomic facts), so the fact pool
# is sized as ``top_k_episode * 20`` to consistently cover enough distinct
# parent memcells before the max-pool reduction. Capped to keep the ANN
# scan bounded on very large top_k requests.
_MAXSIM_FACT_MULTIPLIER = 20
_MAXSIM_FACT_POOL_CAP = 2000

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
        vision_embedding: VisionEmbeddingProvider | None = None,
        reranker: RerankProvider | None,
        llm_client: LLMClient,
        decider_client: LLMClient,
        search_tokenizer: Tokenizer | None = None,
        tuning_source: Callable[[], DeciderSettings],
        trace_path_source: Callable[[], str | None],
        image_recaller: ImageMemoryRecaller | None = None,
        image_store: ImageStore | None = None,
        access: AccessPolicy,
    ) -> None:
        self._access = access
        self._no_facts = NoAtomicFacts()
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
        # Multi-round retrieval runs its own client: [decider] falls back to [llm]
        # field by field, but never shares the extraction client's SDK retries.
        self._decider_llm = decider_client
        self._search_tokenizer = search_tokenizer
        self._images = image_recaller
        self._image_store = image_store
        # The picture-side embedder, when configured. Its query encoder puts the
        # user's question into the space the pictures' bytes live in; without
        # it the visual route is skipped and search is exactly as before.
        self._vision = vision_embedding

    # ── Public entry ────────────────────────────────────────────────

    async def search(
        self,
        req: SearchRequest,
        *,
        question_image_ids: Sequence[str] = (),
        asserted_asker: str | None = None,
    ) -> SearchResponse:
        """Answer one search request.

        Args:
            req: The request.
            question_image_ids: Content hashes of the pictures the question
                carries, already stored by the caller. Only the multi-round
                route uses them (it shows them to the decider every round);
                the others rank nothing they would inform and ignore them.
            asserted_asker: The asker a gateway asserted (``[access]
                identity_header``); a body ``asker_id`` must agree with it.

        Raises:
            ForbiddenError: The asker may not read the scope (403).
            InvalidInputError: A group scope without an asker, or a method
                that has no group route (422).
        """
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
            # The read scope first: who may see what decides every ``where``.
            # Then the filters: a malformed `filters` payload is a user input
            # error (422) and should surface before the server-side component
            # guard (500). The two steps are independent.
            scope = await self._read_scope(req, asserted_asker)
            where = compile_scoped(req.filters, scope.predicate)
            self._validate_components(req)

            if scope.is_group:
                (episodes, images), profiles, unprocessed = await asyncio.gather(
                    self._search_episodes(
                        req, where, question_image_ids=question_image_ids, scope=scope
                    ),
                    self._group_profiles(req, scope),
                    self._load_unprocessed(req, scope),
                )
                images = await self._resolve_episode_pictures(
                    req, episodes, images, scope=scope
                )
                data = SearchData(
                    episodes=episodes,
                    images=images,
                    profiles=profiles,
                    unprocessed_messages=unprocessed,
                    scope=_scope_view(scope),
                )
            elif req.owner_type == "user":
                (episodes, images), profiles, unprocessed = await asyncio.gather(
                    self._search_episodes(
                        req, where, question_image_ids=question_image_ids
                    ),
                    self._fetch_profile(req),
                    self._load_unprocessed(req, scope),
                )
                images = await self._resolve_episode_pictures(req, episodes, images)
                data = SearchData(
                    episodes=episodes,
                    images=images,
                    profiles=profiles,
                    unprocessed_messages=unprocessed,
                )
            else:  # "agent"
                (cases, skills), unprocessed = await asyncio.gather(
                    self._search_cases_and_skills(
                        req, where, case_where=scope.cases_where(where)
                    ),
                    self._load_unprocessed(req, scope),
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

    async def _read_scope(
        self, req: SearchRequest, asserted_asker: str | None
    ) -> ReadScope:
        return await self._access.read_scope(
            resolve_identity(header=asserted_asker, body_asker=req.asker_id),
            ScopeRequest(
                app_id=req.app_id,
                project_id=req.project_id,
                user_id=req.user_id,
                agent_id=req.agent_id,
                group_ids=req.scope_group_ids,
                audience=req.audience,
                context_group_id=req.context_group_id,
            ),
        )

    async def _load_unprocessed(
        self, req: SearchRequest, scope: ReadScope
    ) -> list[UnprocessedMessageDTO]:
        """Load in-flight buffer rows for ``filters.session_id`` (if present).

        Returns ``[]`` unless ``filters`` carries a top-level ``session_id``
        eq scalar — buffer rows have no ``user_id`` / ``agent_id`` attribution
        (boundary detection runs before owner inference), so session is the
        only meaningful query dimension.

        Which of them a read sees is the access policy's (section 9.3, row 3):
        a group scope sees a bound session's messages inside the asker's
        windows; a personal or agent scope sees a session's buffer only when
        its owner is a sender of that session, and never a group session's.
        Before this, any owner id could read any session's buffer.
        """
        session_id = _extract_top_level_session_id(req.filters)
        if session_id is None:
            return []
        session = SessionKey(
            app_id=req.app_id, project_id=req.project_id, session_id=session_id
        )
        messages = await self._session_buffer.list(session)

        async def owner_has_memory() -> bool:
            """Whether the owner took part in the session earlier: memory of
            theirs from it, when nothing buffered is theirs."""
            where = all_of(scope.predicate, eq("session_id", session_id))
            recaller = self._case if scope.owner_type == "agent" else self._ep
            return await recaller.has_any(where)

        visible = await self._access.buffered_messages(
            scope, session, messages, owner_has_memory=owner_has_memory
        )
        return [_unprocessed_message_to_dto(m, session) for m in visible]

    # ── Agent partition ─────────────────────────────────────────────

    async def _search_cases_and_skills(
        self, req: SearchRequest, where: Filter, *, case_where: Filter
    ) -> tuple[list[SearchAgentCaseItem], list[SearchAgentSkillItem]]:
        """Cases + skills, serial when bridging.

        HYBRID + LLM rerank runs serially: reranked cases feed the
        skill bridge. Every other method runs the two kinds in parallel
        with no bridge — the bridge only pays off after rerank has
        produced high-quality case scores to inherit. Cases are recalled
        under ``case_where``: the read scope's group context decides which
        of the cases the agent learned in groups join.
        """
        if _effective_llm_rerank(req):
            cases = await self._search_agent_cases(req, case_where)
            skills = await self._search_agent_skills(
                req, where, case_scores={c.entry_id: c.score for c in cases}
            )
            return cases, skills

        cases, skills = await asyncio.gather(
            self._search_agent_cases(req, case_where),
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
        scope = await self._read_scope(req, None)
        where = compile_scoped(req.filters, scope.predicate)
        episodes, _images = await self._search_episodes(
            req, where, decider=decider, scope=scope
        )
        return episodes

    async def _search_episodes(
        self,
        req: SearchRequest,
        where: Filter,
        *,
        decider: RoundDecider | None = None,
        question_image_ids: Sequence[str] = (),
        scope: ReadScope | None = None,
    ) -> tuple[list[SearchEpisodeItem], list[SearchImageItem]]:
        """User-track retrieval: the episodes and the pictures ranked with them.

        A group scope runs the same routes over a recall with no atomic facts.
        """
        facts = self._no_facts if scope is not None and scope.is_group else self._fact
        if req.method == SearchMethod.AGENTIC:
            episodes = await search_episodes_agentic(
                req.query,
                owner_id=req.owner_id,
                where=where,
                app_id=req.app_id,
                project_id=req.project_id,
                episode_recaller=self._ep,
                atomic_fact_recaller=facts,
                cluster_store=self._clusters,
                embed_query_fn=self._embedding.embed_query,  # type: ignore[union-attr]
                reranker=self._reranker,  # type: ignore[arg-type]
                llm=self._llm,  # type: ignore[arg-type]
                top_k=self._top_k(req.top_k),
                radius=_effective_radius(req),
            )
            # The agentic arm is the pre-picture pipeline and knows nothing about
            # pictures. Rather than splice images into a route that never ranked
            # them, it returns episodes alone and leaves ``images`` empty --
            # an empty array is honest, an image pinned to the wrong rank is not.
            return episodes, []

        if req.method == SearchMethod.LLM_MULTIROUND:
            return await search_episodes_and_images_llm_multiround(
                _question(req, scope),
                owner_id=req.owner_id,
                where=where,
                app_id=req.app_id,
                project_id=req.project_id,
                episode_recaller=self._ep,
                atomic_fact_recaller=facts,
                embed_query_fn=self._embedding.embed_query,  # type: ignore[union-attr]
                llm=self._decider_llm,  # type: ignore[arg-type]
                top_k=self._top_k(req.top_k),
                reranker=self._reranker,  # type: ignore[arg-type]
                decider=decider,
                tuning_source=self._tuning_source,
                trace_path_source=self._trace_path_source,
                image_recaller=self._images,
                image_store=self._image_store,
                question_image_ids=question_image_ids,
                visual_embed_query_fn=(
                    self._vision.embed_query if self._vision is not None else None
                ),
            )

        fusion_mode, _ = resolve_pipeline(req.method, "episode")
        top_k = self._top_k(req.top_k)

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
                    recalled = await self._maxsim_atomic_recall(
                        req, where, top_k, facts
                    )
            # ``atomic_facts`` stays empty: facts come back only when the HYBRID
            # pipeline surfaces them with a score (see ``reshape_hybrid_output``).
            # Single-route recall has no per-fact score against the query, so
            # we do not back-fill — that would emit ``score=0.0`` facts whose
            # semantics are ambiguous.
            episodes = shape_episodes(
                recalled.candidates[:top_k], recalled.stored_by_id
            )
            images = await self._image_hits(req, where, top_k)
            return episodes, images

        # ── HYBRID: heap-expand hierarchy over parallel sparse + dense recall ──
        sparse, dense, query_vector = await self._recall_sparse_dense(
            self._ep, req, where, top_k
        )
        with memory_span(
            "everos.search.rank",
            observation_type="span",
            metadata={"phase": fusion_mode},
        ):
            rrf_candidates = rrf(sparse.candidates, dense.candidates)
            fact_buckets = await facts.facts_for_episodes(
                build_ep_to_fact_parents(rrf_candidates),
                where,
                per_episode=max(top_k * 2, 20),
                query_vector=query_vector,
            )
            scored = heap_expand(
                sparse=sparse.candidates,
                dense=dense.candidates,
                episode_to_facts=fact_buckets.buckets,
                top_k=top_k,
            )
            shaped = reshape_hybrid_output(
                scored,
                episodes_by_id=merge_stored(sparse, dense),
                facts_by_id=fact_buckets.stored_by_id,
            )
            if req.min_score is not None:
                shaped = [s for s in shaped if s.score >= req.min_score]
            images = await self._image_hits(req, where, top_k)
            return shaped, images

    async def _resolve_episode_pictures(
        self,
        req: SearchRequest,
        episodes: list[SearchEpisodeItem],
        images: list[SearchImageItem],
        *,
        scope: ReadScope | None = None,
    ) -> list[SearchImageItem]:
        """Attach the pictures the selected episodes refer to; returns the extra ones.

        An episode body names its pictures by id and never restates what they
        show, so a selected episode whose picture memory was not selected too
        would otherwise hand the answering model a dangling reference and no
        content — worse than 1.0, where the picture's text was inlined.

        The lookup goes through the same scope filter the search used, and only
        current pictures come back: a forgotten or superseded picture is not
        revived by an episode that still refers to it.

        Args:
            req: The request, for the scope the reference is resolved in — a
                reference must not cross into another owner's or space's
                pictures.
            episodes: The selected episodes (mutated: their ``images`` filled).
            images: The pictures *ranked* alongside them.

        Returns:
            ``images`` unchanged. A picture reached only through a reference is
            attached to the episode that names it and is not added to the
            ranked list: the list is what the ranking chose, and padding it
            with unranked rows would report as core memory something no
            decider selected. A picture that is both ranked and referred to
            appears once, in the list, and is not repeated inside the episode.
        """
        wanted = {image_id for episode in episodes for image_id in episode.image_ids}
        if self._images is None or not wanted:
            return images
        selected = {image.image_id for image in images}
        if scope is not None and scope.is_group:
            # Through the read scope: a picture memory outside the asker's
            # windows stays out even when a visible episode refers to it.
            stored = await self._images.find_by_image_ids_in(
                scope.predicate, list(wanted)
            )
        else:
            stored = await self._images.find_by_image_ids(
                Scope(
                    app_id=req.app_id,
                    project_id=req.project_id,
                    owner_id=req.owner_id,
                    owner_type="user",
                ),
                list(wanted),
            )
        by_image_id = _picture_by_image_id(stored)
        attached = attach_referenced_pictures(
            episodes,
            image_ids=wanted,
            stored_images=by_image_id,
            selected_ids=selected,
            thumbnail_edge=self._tuning_source().image_thumbnail_edge,
        )
        logger.debug(
            "search_episode_pictures_resolved",
            episodes=len(episodes),
            referenced=len(wanted),
            selected=len(selected),
            attached=len(attached),
            unresolved=len(wanted) - len(selected) - len(attached),
        )
        return images

    async def _image_hits(
        self, req: SearchRequest, where: Filter, top_k: int
    ) -> list[SearchImageItem]:
        """Picture memories for a single-route or hybrid search.

        Those two routes rank the episode hierarchy through algorithms that
        know nothing about pictures (heap-expand over atomic facts), so
        pictures cannot join their fusion without changing what the route
        measures. They get their own recall over the same query and the same
        filters, and are ranked on their own score. That is a real limitation
        of these routes and the reason multi-round is the default: it is the
        only route where the two kinds compete in one pool.
        """
        if self._images is None:
            return []
        recalled = await self._images.sparse_recall(req.query, where, limit=top_k)
        if req.method == SearchMethod.VECTOR and self._embedding is not None:
            vector = await self._embedding.embed_query(req.query)
            dense = await self._images.dense_recall(vector, where, limit=top_k)
            recalled = merge_by_id_max(recalled, dense)
        if req.min_score is not None:
            recalled = recalled.within_radius(req.min_score)
        return shape_images(
            recalled.candidates[:top_k],
            recalled.stored_by_id,
            thumbnail_edge=self._tuning_source().image_thumbnail_edge,
        )

    # ── Agent cases ─────────────────────────────────────────────────

    async def _search_agent_cases(
        self, req: SearchRequest, where: Filter
    ) -> list[SearchAgentCaseItem]:
        if req.method == SearchMethod.AGENTIC:
            return await search_agent_cases_agentic(
                req.query,
                where=where,
                case_recaller=self._case,
                embed_query_fn=self._embedding.embed_query,  # type: ignore[union-attr]
                reranker=self._reranker,  # type: ignore[arg-type]
                llm=self._llm,  # type: ignore[arg-type]
                top_k=self._top_k(req.top_k),
                radius=_effective_radius(req),
            )
        fusion_mode, _ = resolve_pipeline(req.method, "agent_case")
        enable_rerank = _effective_llm_rerank(req)
        top_k = self._top_k(req.top_k, cap=_AGENT_TOP_K_CAP)

        if fusion_mode is None:
            recalled = await self._single_route_recall(
                self._case, req, where, top_k, cap=_AGENT_TOP_K_CAP
            )
            return shape_agent_cases(recalled.candidates[:top_k], recalled.stored_by_id)

        sparse, dense, _ = await self._recall_sparse_dense(
            self._case, req, where, top_k, cap=_AGENT_TOP_K_CAP
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
            return await search_agent_skills_agentic(
                req.query,
                where=where,
                skill_recaller=self._skill,
                embed_query_fn=self._embedding.embed_query,  # type: ignore[union-attr]
                reranker=self._reranker,  # type: ignore[arg-type]
                llm=self._llm,  # type: ignore[arg-type]
                top_k=self._top_k(req.top_k, cap=_AGENT_TOP_K_CAP),
                radius=_effective_radius(req),
            )
        fusion_mode, _ = resolve_pipeline(req.method, "agent_skill")
        top_k = self._top_k(req.top_k, cap=_AGENT_TOP_K_CAP)

        if fusion_mode is None:
            recalled = await self._single_route_recall(
                self._skill, req, where, top_k, cap=_AGENT_TOP_K_CAP
            )
            return shape_agent_skills(
                recalled.candidates[:top_k], recalled.stored_by_id
            )

        sparse, dense, _ = await self._recall_sparse_dense(
            self._skill, req, where, top_k, cap=_AGENT_TOP_K_CAP
        )

        # Case→skill bridge: union skills surfaced via lineage cases into
        # the dense pool with their max-pooled source-case score.
        if case_scores:
            bridged = await self._skill.bridged_from_cases(
                case_scores, where, top_k=top_k
            )
            dense = merge_by_id_max(dense, bridged)

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
            return shape_agent_skills_ranked(output, merge_stored(sparse, dense))

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
        return await self._profile.fetch(
            req.owner_id, app_id=req.app_id, project_id=req.project_id
        )

    async def _group_profiles(
        self, req: SearchRequest, scope: ReadScope
    ) -> list[SearchProfileItem]:
        """The asker's member-in-group profiles and those of the members named.

        Section 5.3: with ``include_profile``, every group of the scope
        contributes the asker's profile in it and those of the members the
        query names by roster name or alias -- at most
        :data:`_NAMED_PROFILES_CAP` of them, in the order the query names
        them. Each holds only the items the asker's windows allow; a personal
        profile is never read.
        """
        if not req.include_profile or scope.asker_id is None:
            return []
        candidates = [
            member
            for view in scope.groups
            for member in roster_names(view.roster)
            if member.member_id != scope.asker_id
        ]
        # A member of several groups of the pool is one name, named once.
        named = list(dict.fromkeys(named_members(req.query, candidates)))[
            :_NAMED_PROFILES_CAP
        ]
        wanted = [scope.asker_id, *named]
        fetched = await asyncio.gather(
            *(
                self._profile.fetch_member(scope, view.group_id, member)
                for view in scope.groups
                for member in wanted
            )
        )
        return [item for item in fetched if item is not None]

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
                return await recaller.sparse_recall(
                    req.query, where, limit=self._recall_limit(req.top_k, cap=cap)
                )
            vector = await self._embed_query(req.query)
            recalled = await recaller.dense_recall(
                vector, where, limit=self._recall_limit(req.top_k, cap=cap)
            )
            return recalled.within_radius(_effective_radius(req))

    async def _recall_sparse_dense(
        self,
        recaller: KindRecaller,
        req: SearchRequest,
        where: Filter,
        top_k: int,
        *,
        cap: int = _DEFAULT_TOP_K_CAP,
    ) -> tuple[RecallResult[Any], RecallResult[Any], list[float]]:
        """Fan out keyword + vector recall in parallel.

        The third return is the query embedding itself — the HYBRID
        pipeline passes it into ``facts_for_episodes`` so per-fact
        cosine scoring reuses the same vector instead of re-embedding
        the query. Dense recall is empty when no embedding provider is
        configured.
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
            return sparse, dense.within_radius(_effective_radius(req)), vector

    async def _maxsim_atomic_recall(
        self,
        req: SearchRequest,
        where: Filter,
        top_k: int,
        facts: AtomicFactRecaller,
    ) -> RecallResult[Any]:
        """MaxSim recall: atomic-fact ANN max-pooled onto parent episodes.

        The fact pool is sized ``top_k * 20`` (facts are ~28x denser than
        episodes), capped to keep the ANN bounded on large ``top_k``.
        """
        recalled = await maxsim_episode_recall(
            await self._embed_query(req.query),
            where,
            facts=facts,
            episodes=self._ep,
            top_k=top_k,
            fact_limit=min(top_k * _MAXSIM_FACT_MULTIPLIER, _MAXSIM_FACT_POOL_CAP),
            fallback_limit=self._recall_limit(req.top_k),
            owner_id=req.owner_id,
        )
        return recalled.within_radius(_effective_radius(req))

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
        if req.owner_type == "group" and req.method == SearchMethod.AGENTIC:
            raise InvalidInputError(
                "method 'agentic' has no group route; use hybrid or llm_multiround"
            )
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


def _question(req: SearchRequest, scope: ReadScope | None) -> str:
    """The question the multi-round loop starts from (section 8.2).

    On a group scope it names the asker -- ``"<display name>: <question>"``,
    the line shape the decider reads candidates in -- so "what do I still
    have to do?" is read as that member's question. ``[groups]
    asker_in_question = false`` keeps the question as asked.
    """
    if scope is None or not scope.is_group or scope.asker_id is None:
        return req.query
    if not load_settings().groups.asker_in_question:
        return req.query
    for view in scope.groups:
        current = view.roster.current(scope.asker_id)
        if current is not None:
            return f"{current.display_name}: {req.query}"
    return req.query


def _scope_view(scope: ReadScope) -> SearchScope:
    """The scope block a group search returns: groups, asker, windows."""
    return SearchScope(
        group_ids=list(scope.owner_ids),
        asker_id=scope.asker_id or "",
        audience=scope.audience,
        windows=[
            SearchWindow(group_id=view.group_id, from_=w.start, to=w.end)
            for view in scope.groups
            for w in view.windows
        ],
    )


def _picture_by_image_id(
    stored: Sequence[Stored[ImageMemory]],
) -> dict[str, Stored[ImageMemory]]:
    """One picture per content hash, preferring the current occurrence.

    A picture can have several occurrences (one per arrival), all pointing at
    the same file. Resolving an episode's reference needs one description, and
    the current one is the memory about it — an older occurrence is history
    kept for "what did it say before".
    """
    chosen: dict[str, Stored[ImageMemory]] = {}
    for row in stored:
        image_id = row.item.image_id
        if not image_id:
            continue
        prior = chosen.get(image_id)
        if prior is None or (prior.item.superseded_by and not row.item.superseded_by):
            chosen[image_id] = row
    return chosen


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


async def _no_recall() -> RecallResult[Any]:
    return RecallResult.empty()
