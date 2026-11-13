"""SearchManager — top-level orchestrator for ``POST /api/v2/memory/search``.

Multi-round retrieval (``llm_multiround``) is the only search method, and
every search runs one flow: the read scope, the pool it covers, the loop over
that pool, the pictures the chosen episodes name, the profiles, the buffer.
What differs is only what the scope switches on:

* ``user`` → the user's ``episodes`` and pictures, ranked in one pool (+ the
  user's profile when ``include_profile=true``)
* ``group`` → ``episodes``, scenes and pictures of one or several groups (or
  of every group of a project the asker belongs to), as one pool (+ the
  asker's and named members' member-in-group ``profiles`` when
  ``include_profile=true``); the question names the asker, the loop follows
  stored links, and the response adds the visible current group profile,
  the constraint block, the ``coverage`` of the loop and an ``audience`` on
  every record (docs/group-memory.md, sections 8.3-8.7)
* ``user`` with ``include_groups`` (off by default) → the personal pool and
  the user's current groups, read as a group search by the user reads them,
  ranked as one pool; the question names the user, every record names its
  owner, and the user's own member-in-group profiles and the groups' current
  decisions are the only group attachments: no link expansion, no
  constraint block, no ``coverage`` (section 8.8)

An agent's memory is not searched: the request DTO refuses ``agent_id``.

Every request first goes through the access policy
(:mod:`everos.memory.access`): its read scope's predicate replaces the owner
part of every recall's ``where``, so visibility is decided in the index query
and no ranking -- the multi-round decider included -- sees a candidate the
asker may not. A group scope attaches no personal profile. Its multi-round
question names the asker (``"<display name>: <question>"``, section 8.2), its
loop follows the core's stored links (``[search] expansion``), and
``include_profile`` attaches member-in-group profiles only.

Recall needs no embedding provider. Multi-round recall runs BM25 and vector
recall for every sub-query; a deployment without an embedder runs BM25 alone
(keyword recall), and the loop, the decider and the ranking are otherwise the
same.

The manager never writes memory; it reads the derived index and markdown,
and asks the picture store for scaled copies (a disposable cache under
``figures/.cache/``): the decider's thumbnails, and the answering model's copy
every returned picture names as ``model_path``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple

from everos.config import GroupSearchSettings, load_settings
from everos.config.settings import DeciderSettings
from everos.core.context import resolve_request_id
from everos.core.errors import ConfigurationError
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
from everos.memory.extract.pipeline.group_render import label_cell
from everos.memory.groups import named_members, roster_names
from everos.memory.models import CanonicalMessage, ImageMemory, MemCell
from everos.memory.ports import (
    EpisodeIndex,
    Filter,
    GroupProfileIndex,
    MemcellStore,
    MessageForgetStore,
    SceneStore,
    SessionBufferStore,
    SessionKey,
    Stored,
    all_of,
    any_of,
    eq,
)

from .audience import with_audience
from .constraints import bind_constraints, build_constraints
from .decider_health import DeciderStatus
from .dto import (
    ConstraintItem,
    FilterNode,
    SearchData,
    SearchDegradation,
    SearchEpisodeItem,
    SearchGroupProfile,
    SearchImageItem,
    SearchProfileItem,
    SearchRequest,
    SearchResponse,
    SearchScope,
    SearchWindow,
    UnprocessedMessageDTO,
)
from .expansion import GroupLinks
from .filters import compile_scoped
from .group_profile import visible_profile, visible_profile_items
from .llm_multiround import (
    Coverage,
    DeciderFailure,
    RoundDecider,
    run_llm_multiround,
)
from .shaper import attach_referenced_pictures, name_copies

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

    from everos.component.embedding import EmbeddingProvider, VisionEmbeddingProvider
    from everos.component.images import ImageLimits
    from everos.memory.ports.images import ImageStore

    from .recall import EpisodeRecaller, ImageMemoryRecaller, ProfileRecaller

logger = get_logger(__name__)

# ``top_k = -1`` (unlimited) is capped at this many results.
_DEFAULT_TOP_K_CAP = 100

# Members a group search attaches the profiles of, besides the asker's, when
# the query names them (docs/group-memory.md, section 5.3).
_NAMED_PROFILES_CAP = 3


def _top_score(data: SearchData) -> float:
    """Max relevance score across the returned episodes (0.0 when empty).

    Profiles are excluded — they are a KV fetch with no query-relevance score.
    The multi-round score is the max RRF score across sub-queries, not a
    calibrated ``[0, 1]`` relevance, so it is reported without a hit verdict.
    """
    return max((item.score for item in data.episodes), default=0.0)


class _Ranked(NamedTuple):
    """What the multi-round loop returned: its ranking, its stop and its core.

    ``core`` is the decider's core as result ids; the constraint block of a
    group search is built from it. ``failures`` are the rounds the decider
    did not decide, which the response reports as ``degraded``.
    """

    episodes: list[SearchEpisodeItem]
    images: list[SearchImageItem]
    coverage: Coverage
    core: list[str]
    failures: tuple[DeciderFailure, ...] = ()


def _degraded(failures: Sequence[DeciderFailure]) -> list[SearchDegradation]:
    """The response's ``degraded`` entries for the rounds the decider missed."""
    return [
        SearchDegradation(
            round=f.round_idx,
            outcome=f.outcome,
            reason=f.reason,
            attempts=f.attempts,
        )
        for f in failures
    ]


class SearchManager:
    """Orchestrates multi-round retrieval, profiles and the session buffer."""

    def __init__(
        self,
        *,
        episode_recaller: EpisodeRecaller,
        profile_recaller: ProfileRecaller,
        session_buffer: SessionBufferStore,
        embedding: EmbeddingProvider | None,
        vision_embedding: VisionEmbeddingProvider | None = None,
        llm_client: LLMClient,
        decider_client: LLMClient,
        tuning_source: Callable[[], DeciderSettings],
        trace_path_source: Callable[[], str | None],
        image_recaller: ImageMemoryRecaller | None = None,
        image_store: ImageStore | None = None,
        image_limits: ImageLimits | None = None,
        access: AccessPolicy,
        group_profile_index: GroupProfileIndex | None = None,
        episode_index: EpisodeIndex | None = None,
        memcells: MemcellStore | None = None,
        message_forget: MessageForgetStore | None = None,
        scenes: SceneStore | None = None,
        group_search: GroupSearchSettings | None = None,
        decider_status: DeciderStatus | None = None,
    ) -> None:
        self._access = access
        # Counts the built-in decider's rounds for GET /health; an injected
        # policy's rounds are not the configured decider's and are not counted.
        self._decider_status = decider_status
        # Group scopes (docs/group-memory.md, sections 8.3-8.6): the group
        # profile's items, the segments constraints quote, the scenes
        # expansion follows, and the switches that decide all three.
        self._group_profile = group_profile_index
        self._episode_index = episode_index
        self._memcells = memcells
        self._message_forget = message_forget
        self._scenes = scenes
        self._group_search = group_search or GroupSearchSettings()
        self._tuning_source = tuning_source
        self._trace_path_source = trace_path_source
        self._ep = episode_recaller
        self._profile = profile_recaller
        self._session_buffer = session_buffer
        self._embedding = embedding
        # The [llm] client: the constraint binding call of a group search
        # ([search] constraints = "model").
        self._llm = llm_client
        # Multi-round retrieval runs its own client: [decider] falls back to [llm]
        # field by field, but never shares the extraction client's SDK retries.
        self._decider_llm = decider_client
        self._images = image_recaller
        self._image_store = image_store
        # ``[images]``: the size of the answering model's copy each returned
        # picture names (``model_path``). Without it the field stays empty.
        self._image_limits = image_limits
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
                carries, already stored by the caller. The decider is shown
                them every round.
            asserted_asker: The asker a gateway asserted (``[access]
                identity_header``); a body ``asker_id`` must agree with it.

        Raises:
            ForbiddenError: The asker may not read the scope (403).
            InvalidInputError: A group scope without an asker (422).
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
            # error (422).
            scope = await self._read_scope(req, asserted_asker)
            groups = await self._member_groups(req, scope)
            where = compile_scoped(req.filters, _pooled(scope, groups))
            ranked, profiles, unprocessed, group_profile = await asyncio.gather(
                self._search_episodes(
                    req,
                    where,
                    question_image_ids=question_image_ids,
                    scope=scope,
                    groups=groups,
                ),
                self._profiles(req, scope, groups),
                self._load_unprocessed(req, scope, groups=groups),
                self._attach_group_profile(req, scope, groups),
            )
            await self._resolve_episode_pictures(ranked, scope, groups)
            images = await self._name_copies(ranked.episodes, ranked.images)
            data = await self._data(
                req,
                scope,
                groups,
                ranked=ranked,
                images=images,
                profiles=profiles,
                unprocessed=unprocessed,
                group_profile=group_profile,
            )

            # Returned hits (ids only) — content, so only when capture_content
            # is on. This is the point of a search trace: what came back.
            capture_output(span, {"episodes": [e.id for e in data.episodes]})

            # Recall-quality signal on the span (always on; the Langfuse
            # scores push is separate and gated on creds — see the score sink).
            # No hit verdict: the multi-round score is not calibrated, and a
            # fixed threshold over it would be a misleading near-constant.
            top_score = _top_score(data)
            span.set_attribute("everos.search.top_score", top_score)
            ids = current_trace_ids()
            if ids is not None:
                emit_recall_scores(
                    trace_id=ids[0],
                    observation_id=ids[1],
                    top_score=top_score,
                    hit=None,
                    method=req.method.value,
                )

            return SearchResponse(request_id=request_id, data=data)

    # ── The response ────────────────────────────────────────────────

    async def _data(
        self,
        req: SearchRequest,
        scope: ReadScope,
        groups: ReadScope | None,
        *,
        ranked: _Ranked,
        images: list[SearchImageItem],
        profiles: list[SearchProfileItem],
        unprocessed: list[UnprocessedMessageDTO],
        group_profile: SearchGroupProfile | None,
    ) -> SearchData:
        """What every search answers, with the features its scope switches on.

        Every group record gets its audience (section 8.6). A group scope adds
        the scope block, the constraint block (section 8.4, built from the
        records before their audience is set) and the loop's ``coverage``. A
        personal search with ``include_groups`` adds the scope block of the
        groups it covered and names the owner of every record; it has no
        constraint block and no ``coverage``.
        """
        constraints: list[ConstraintItem] = []
        if scope.is_group:
            constraints = await self._constraints(req, scope, ranked, images)
        episodes = ranked.episodes
        pool = _groups_of(scope, groups)
        if pool is not None:
            episodes = with_audience(pool, episodes)
            images = with_audience(pool, images)
        if req.include_groups:
            logger.debug(
                "search_include_groups",
                groups=0 if groups is None else len(groups.owner_ids),
                episodes=len(episodes),
                images=len(images),
                group_records=sum(1 for r in (*episodes, *images) if r.group_id),
            )
            episodes = [_owned_episode(e) for e in episodes]
            images = [_owned_image(i) for i in images]
        return SearchData(
            episodes=episodes,
            images=images,
            profiles=profiles,
            unprocessed_messages=unprocessed,
            scope=_scope_block(req, scope, groups),
            group_profile=group_profile,
            constraints=constraints,
            coverage=ranked.coverage if scope.is_group else None,
            degraded=_degraded(ranked.failures),
        )

    async def _member_groups(
        self, req: SearchRequest, scope: ReadScope
    ) -> ReadScope | None:
        """The groups a personal request also searches (``include_groups``).

        The access policy's (:meth:`AccessPolicy.member_groups`): each group
        read exactly as a group search by the user reads it. Their predicate
        is ORed onto the personal one, so every recall of the loop ranks
        personal and group records together, the decider sees them in one
        rendering, and nothing outside the user's windows is ever a
        candidate. ``None`` for a group scope, without ``include_groups``, or
        when the user belongs to none of the groups.
        """
        if not req.include_groups or scope.is_group:
            return None
        return await self._access.member_groups(scope, only=req.include_group_ids)

    # ── Unprocessed buffer ──────────────────────────────────────────

    async def _read_scope(
        self, req: SearchRequest, asserted_asker: str | None
    ) -> ReadScope:
        """The access policy's scope; an ``audience`` the request does not
        name is the asker's default (``group`` for a group's deliverer)."""
        return await self._access.read_scope(
            resolve_identity(header=asserted_asker, body_asker=req.asker_id),
            ScopeRequest(
                app_id=req.app_id,
                project_id=req.project_id,
                user_id=req.user_id,
                group_ids=req.scope_group_ids,
                audience=req.audience if "audience" in req.model_fields_set else None,
                project_scope=req.project_scope,
            ),
        )

    async def _load_unprocessed(
        self,
        req: SearchRequest,
        scope: ReadScope,
        *,
        groups: ReadScope | None = None,
    ) -> list[UnprocessedMessageDTO]:
        """Load in-flight buffer rows for ``filters.session_id`` (if present).

        Returns ``[]`` unless ``filters`` carries a top-level ``session_id``
        eq scalar — buffer rows have no ``user_id`` attribution (boundary
        detection runs before owner inference), so session is the only
        meaningful query dimension.

        Which of them a read sees is the access policy's (section 9.3, row 3):
        a group scope sees a bound session's messages inside the asker's
        windows; a personal scope sees a session's buffer only when its owner
        is a sender of that session, and never a group session's. A personal
        scope with ``groups`` (``include_groups``) sees a session of one of
        those groups as a group search by its owner would: inside the owner's
        windows, without the forgotten messages.
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
            return await self._ep.has_any(where)

        visible = await self._access.buffered_messages(
            scope, session, messages, owner_has_memory=owner_has_memory
        )
        if groups is not None and not visible:
            # A personal read never sees a bound session; the session of a
            # group searched alongside is read by that group's rules.
            visible = await self._access.buffered_messages(
                groups, session, messages, owner_has_memory=owner_has_memory
            )
        return [_unprocessed_message_to_dto(m, session) for m in visible]

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
        sampled sub-queries (Jaccard median 0.538), because it reached the
        public search route's former hybrid pipeline rather than
        ``llm_multiround``'s per-sub-query ``rrf(sparse, dense)[:topk]``.
        """
        scope = await self._read_scope(req, None)
        groups = await self._member_groups(req, scope)
        where = compile_scoped(req.filters, _pooled(scope, groups))
        ranked = await self._search_episodes(
            req, where, decider=decider, scope=scope, groups=groups
        )
        if not req.include_groups:
            return ranked.episodes
        return [_owned_episode(e) for e in ranked.episodes]

    async def _search_episodes(
        self,
        req: SearchRequest,
        where: Filter,
        *,
        scope: ReadScope,
        groups: ReadScope | None,
        decider: RoundDecider | None = None,
        question_image_ids: Sequence[str] = (),
    ) -> _Ranked:
        """The episodes and the pictures ranked with them, by multi-round.

        A group scope's loop also follows the core's stored links (section
        8.3) when ``[search] expansion`` is on. ``groups`` are the groups a
        personal search also covers (``include_groups``): their roster names
        the asker in the question; ``where`` already holds them. A personal
        search's question is the one asked.
        """
        expander = self._expander(scope, where)
        result = await run_llm_multiround(
            _question(req, _groups_of(scope, groups)),
            # A project scope names no owner of its own: the scope's first group.
            owner_id=scope.owner_id,
            where=where,
            app_id=req.app_id,
            project_id=req.project_id,
            episode_recaller=self._ep,
            embed_query_fn=self._embed_query,
            llm=self._decider_llm,
            top_k=self._top_k(req.top_k),
            decider=decider,
            tuning_source=self._tuning_source,
            trace_path_source=self._trace_path_source,
            image_recaller=self._images,
            image_store=self._image_store,
            question_image_ids=question_image_ids,
            visual_embed_query_fn=(
                self._vision.embed_query if self._vision is not None else None
            ),
            expander=expander,
            link_slots=self._group_search.link_slots if expander else 0,
        )
        if decider is None and self._decider_status is not None:
            self._decider_status.record(result.rounds, result.failures)
        return _Ranked(
            result.episodes,
            result.images,
            result.coverage,
            list(result.core),
            result.failures,
        )

    async def _resolve_episode_pictures(
        self, ranked: _Ranked, scope: ReadScope, groups: ReadScope | None
    ) -> None:
        """Attach to each selected episode the pictures it refers to.

        An episode body names its pictures by id and never restates what they
        show, so a selected episode whose picture memory was not selected too
        would otherwise hand the answering model a dangling reference and no
        content — worse than 1.0, where the picture's text was inlined.

        A reference resolves in its episode's own scope, through that scope's
        read predicate: a user's episode among the user's pictures, a group's
        among the pictures of the groups read that the asker may see -- a
        picture memory outside the asker's windows stays out even when a
        visible episode refers to it, and a personal episode never resolves
        to a group's picture, nor a group's to the user's own. Only current
        pictures come back: a forgotten or superseded picture is not revived
        by an episode that still refers to it.

        ``ranked.episodes`` are mutated (their ``images`` filled); the ranked
        pictures stay what the ranking chose (:meth:`_attach_pictures`).
        """
        pool = _groups_of(scope, groups)
        if not scope.is_group:
            await self._attach_pictures(
                [e for e in ranked.episodes if e.group_id is None],
                ranked.images,
                scope.predicate,
            )
        if pool is not None:
            await self._attach_pictures(
                [e for e in ranked.episodes if e.group_id is not None],
                ranked.images,
                pool.predicate,
            )

    async def _attach_pictures(
        self,
        episodes: list[SearchEpisodeItem],
        images: list[SearchImageItem],
        where: Filter,
    ) -> None:
        """Attach the pictures ``episodes`` refer to, looked up inside ``where``.

        A picture reached only through a reference is attached to the episode
        that names it and is not added to the ranked list: the list is what
        the ranking chose, and padding it with unranked rows would report as
        core memory something no decider selected. A picture that is both
        ranked and referred to appears once, in the list, and is not repeated
        inside the episode.
        """
        wanted = {image_id for episode in episodes for image_id in episode.image_ids}
        if self._images is None or not wanted:
            return
        selected = {image.image_id for image in images}
        stored = await self._images.find_by_image_ids(where, list(wanted))
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

    async def _name_copies(
        self, episodes: Sequence[SearchEpisodeItem], images: list[SearchImageItem]
    ) -> list[SearchImageItem]:
        """Every returned picture's copies, named by the store (:func:`name_copies`)."""
        return await name_copies(
            episodes,
            images,
            store=self._image_store,
            limits=self._image_limits,
            thumbnail_edge=self._tuning_source().image_thumbnail_edge,
        )

    # ── Profile ─────────────────────────────────────────────────────

    async def _profiles(
        self, req: SearchRequest, scope: ReadScope, groups: ReadScope | None
    ) -> list[SearchProfileItem]:
        """The profiles ``include_profile`` attaches, one per owner it reaches.

        A personal search attaches the user's profile (naming its owner with
        ``include_groups``). Every group of the pool contributes the asker's
        member-in-group profile in it, cut to the asker's windows; a group
        search also those of the members the query names by roster name or
        alias -- at most :data:`_NAMED_PROFILES_CAP` of them, in the order the
        query names them (section 5.3) -- and a personal search over its
        groups never another member's. A personal profile is never read in a
        group scope.
        """
        if not req.include_profile:
            return []
        found: list[SearchProfileItem] = []
        if not scope.is_group:
            found = await self._profile.fetch(
                req.owner_id, app_id=req.app_id, project_id=req.project_id
            )
            if req.include_groups:
                found = [i.model_copy(update={"owner_type": "user"}) for i in found]
        pool = _groups_of(scope, groups)
        if pool is None or pool.asker_id is None:
            return found
        members = [pool.asker_id]
        if scope.is_group:
            members += _named(req, scope)
        fetched = await asyncio.gather(
            *(
                self._profile.fetch_member(pool, view.group_id, member)
                for view in pool.groups
                for member in members
            )
        )
        return [*found, *(item for item in fetched if item is not None)]

    # ── Group-scope additions (sections 8.3-8.6) ────────────────────

    def _expander(self, scope: ReadScope | None, where: Filter) -> GroupLinks | None:
        """The link expander of a group multi-round search, when it is on."""
        settings = self._group_search
        if (
            scope is None
            or not scope.is_group
            or not settings.expansion
            or settings.link_slots <= 0
            or self._group_profile is None
            or self._episode_index is None
        ):
            return None
        return GroupLinks(
            scope=scope,
            where=where,
            episodes=self._episode_index,
            profile=self._group_profile,
            scenes=self._scenes,
        )

    async def _attach_group_profile(
        self, req: SearchRequest, scope: ReadScope, groups: ReadScope | None
    ) -> SearchGroupProfile | None:
        """The visible current group-profile items (section 5.3), each kind capped.

        On by default for a group scope (``include_group_profile``): it is
        what grounds a request about the group itself -- its purpose, its
        current plan, who does what -- when the ranking found little. A
        personal search over its groups (``include_groups``) attaches their
        current decisions only, the cap across all the groups. One index
        read, no model call; ``include_group_profile = false`` leaves it out.
        """
        pool = _groups_of(scope, groups)
        if (
            req.include_group_profile is False
            or pool is None
            or self._group_profile is None
        ):
            return None
        stored = await visible_profile_items(
            self._group_profile,
            pool,
            where=None if scope.is_group else eq("kind", "decision"),
        )
        attached, _view = visible_profile(
            stored, cap=self._group_search.group_profile_cap
        )
        return SearchGroupProfile(
            topics=with_audience(pool, attached.topics),
            decisions=with_audience(pool, attached.decisions),
            dissent=with_audience(pool, attached.dissent),
            roles=with_audience(pool, attached.roles),
        )

    async def _constraints(
        self,
        req: SearchRequest,
        scope: ReadScope,
        ranked: _Ranked,
        images: Sequence[SearchImageItem],
    ) -> list[ConstraintItem]:
        """The constraint block (section 8.4), on by default for a group scope.

        Built from the decider's core among the returned records. ``[search]
        constraints = "model"`` adds the binding call over the rules layer's
        constraints.
        """
        if req.constraints is False:
            return []
        if (
            self._group_profile is None
            or self._memcells is None
            or self._message_forget is None
        ):
            return []
        core_ids = set(ranked.core)
        core = [r for r in (*ranked.episodes, *images) if r.id in core_ids]
        constraints, cells = await build_constraints(
            scope=scope,
            core=core,
            profile=self._group_profile,
            memcells=self._memcells,
            forgotten=self._message_forget,
        )
        if self._group_search.constraints != "model" or not constraints:
            return constraints
        bound, _dropped = await bind_constraints(
            self._llm,
            question=req.query,
            scope=scope,
            constraints=constraints,
            cells=_labelled(cells, constraints, scope),
        )
        return bound

    # ── Embedding / limits ──────────────────────────────────────────

    async def _embed_query(self, query: str) -> list[float]:
        """The query vector; ``[]`` without an embedder, which skips vector recall."""
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

    @staticmethod
    def _top_k(top_k: int) -> int:
        """Resolve ``-1`` to :data:`_DEFAULT_TOP_K_CAP`; pass others through."""
        return _DEFAULT_TOP_K_CAP if top_k == -1 else top_k


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


def _named(req: SearchRequest, scope: ReadScope) -> list[str]:
    """The members other than the asker the query names, at most
    :data:`_NAMED_PROFILES_CAP`, in the order the query names them.

    A member of several groups of the pool is one name, named once.
    """
    candidates = [
        member
        for view in scope.groups
        for member in roster_names(view.roster)
        if member.member_id != scope.asker_id
    ]
    return list(dict.fromkeys(named_members(req.query, candidates)))[
        :_NAMED_PROFILES_CAP
    ]


def _groups_of(scope: ReadScope, groups: ReadScope | None) -> ReadScope | None:
    """The groups a read covers: a group scope's own, or those a personal
    read also covers (``include_groups``); ``None`` for a personal read alone."""
    return scope if scope.is_group else groups


def _scope_block(
    req: SearchRequest, scope: ReadScope, groups: ReadScope | None
) -> SearchScope | None:
    """The scope block of the response: a group scope's, or the groups a
    personal search with ``include_groups`` covered; none otherwise."""
    if scope.is_group:
        return _scope_view(scope)
    if req.include_groups:
        return _scope_with_groups(scope, groups)
    return None


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


def _pooled(scope: ReadScope, groups: ReadScope | None) -> Filter:
    """The owner part of every recall's ``where``: the personal scope, ORed
    with the groups it also covers (``include_groups``) when there are any."""
    if groups is None:
        return scope.predicate
    return any_of(scope.predicate, groups.predicate)


def _scope_with_groups(scope: ReadScope, groups: ReadScope | None) -> SearchScope:
    """The scope block of a personal search with ``include_groups``.

    The user (``owner_type = user``, the asker) and the groups searched with
    the user's windows in each; no group when the user belongs to none.
    """
    if groups is None:
        return SearchScope(
            owner_type="user", group_ids=[], asker_id=scope.asker_id or ""
        )
    return _scope_view(groups).model_copy(update={"owner_type": "user"})


def _owner_of(record: SearchEpisodeItem | SearchImageItem) -> Literal["user", "group"]:
    return "group" if record.group_id is not None else "user"


def _owned_image(image: SearchImageItem) -> SearchImageItem:
    """A picture memory naming its owner (``include_groups``)."""
    return image.model_copy(update={"owner_type": _owner_of(image)})


def _owned_episode(episode: SearchEpisodeItem) -> SearchEpisodeItem:
    """An episode naming its owner, and so do the pictures attached to it."""
    return episode.model_copy(
        update={
            "owner_type": _owner_of(episode),
            "images": [_owned_image(image) for image in episode.images],
        }
    )


def _labelled(
    cells: Mapping[str, MemCell],
    constraints: Sequence[ConstraintItem],
    scope: ReadScope,
) -> dict[str, MemCell]:
    """The segments with their speakers named by their group's roster.

    A segment belongs to the group of the constraints that cite it; one no
    constraint cites keeps its senders' own names.
    """
    group_of = {c.source.memcell_id: c.scope.group_id for c in constraints}
    out: dict[str, MemCell] = {}
    for memcell_id, cell in cells.items():
        view = scope.group(group_of.get(memcell_id, ""))
        out[memcell_id] = label_cell(cell, view.roster) if view is not None else cell
    return out


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
