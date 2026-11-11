"""SearchManager — top-level orchestrator for ``POST /api/v2/memory/search``.

Multi-round retrieval (``llm_multiround``) is the only search method. Hard
partition by ``owner_type``:

* ``user``  → ``episodes`` and pictures, ranked in one pool (+ ``profiles``
  when ``include_profile=true``)
* ``group`` → ``episodes``, scenes and pictures of one or several groups (or
  of every group of a project the asker belongs to), as one pool (+ the
  asker's and named members' member-in-group ``profiles`` when
  ``include_profile=true``), with the visible current group profile, the
  constraint block, the ``coverage`` of the loop and an ``audience`` on every
  record (docs/group-memory.md, sections 8.3-8.7)

An agent's memory is not searched: the request DTO refuses ``agent_id``.

Every request first goes through the access policy
(:mod:`everos.memory.access`): its read scope's predicate replaces the owner
part of every recall's ``where``, so visibility is decided in the index query
and no ranking -- the multi-round decider included -- sees a candidate the
asker may not. A group scope attaches no personal profile. Its multi-round
question names the asker (``"<display name>: <question>"``, section 8.2), its
loop follows the core's stored links (``[search] expansion``), and
``include_profile`` attaches member-in-group profiles only.

The embedding provider is the one component guard: multi-round recall runs
BM25 and vector recall for every sub-query, so a request without an embedder
is refused (422) before any recall.

The manager never writes to storage; it only reads the derived index and
markdown.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

from everos.config import GroupSearchSettings, load_settings
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
    Scope,
    SessionBufferStore,
    SessionKey,
    Stored,
    all_of,
    eq,
)

from .audience import with_audience
from .constraints import bind_constraints, build_constraints
from .dto import (
    ConstraintItem,
    FilterNode,
    SearchData,
    SearchEpisodeItem,
    SearchGroupProfile,
    SearchImageItem,
    SearchMethod,
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
    RoundDecider,
    run_llm_multiround,
)
from .shaper import attach_referenced_pictures

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

    from everos.component.embedding import EmbeddingProvider, VisionEmbeddingProvider
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
    group search is built from it.
    """

    episodes: list[SearchEpisodeItem]
    images: list[SearchImageItem]
    coverage: Coverage
    core: list[str]


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
        access: AccessPolicy,
        group_profile_index: GroupProfileIndex | None = None,
        episode_index: EpisodeIndex | None = None,
        memcells: MemcellStore | None = None,
        message_forget: MessageForgetStore | None = None,
        scenes: SceneStore | None = None,
        group_search: GroupSearchSettings | None = None,
    ) -> None:
        self._access = access
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
            ProviderNotConfiguredError: No embedding provider (422).
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
            # error (422) and should surface before the server-side component
            # guard. The two steps are independent.
            scope = await self._read_scope(req, asserted_asker)
            where = compile_scoped(req.filters, scope.predicate)
            self._validate_components()

            if scope.is_group:
                ranked, profiles, unprocessed, group_profile = await asyncio.gather(
                    self._search_episodes(
                        req, where, question_image_ids=question_image_ids, scope=scope
                    ),
                    self._group_profiles(req, scope),
                    self._load_unprocessed(req, scope),
                    self._attach_group_profile(req, scope),
                )
                images = await self._resolve_episode_pictures(
                    req, ranked.episodes, ranked.images, scope=scope
                )
                constraints = await self._constraints(req, scope, ranked, images)
                data = SearchData(
                    episodes=with_audience(scope, ranked.episodes),
                    images=with_audience(scope, images),
                    profiles=profiles,
                    unprocessed_messages=unprocessed,
                    scope=_scope_view(scope),
                    group_profile=group_profile,
                    constraints=constraints,
                    coverage=ranked.coverage,
                )
            else:
                ranked, profiles, unprocessed = await asyncio.gather(
                    self._search_episodes(
                        req, where, question_image_ids=question_image_ids
                    ),
                    self._fetch_profile(req),
                    self._load_unprocessed(req, scope),
                )
                images = await self._resolve_episode_pictures(
                    req, ranked.episodes, ranked.images
                )
                data = SearchData(
                    episodes=ranked.episodes,
                    images=images,
                    profiles=profiles,
                    unprocessed_messages=unprocessed,
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
                group_ids=req.scope_group_ids,
                audience=req.audience,
                project_scope=req.project_scope,
            ),
        )

    async def _load_unprocessed(
        self, req: SearchRequest, scope: ReadScope
    ) -> list[UnprocessedMessageDTO]:
        """Load in-flight buffer rows for ``filters.session_id`` (if present).

        Returns ``[]`` unless ``filters`` carries a top-level ``session_id``
        eq scalar — buffer rows have no ``user_id`` attribution (boundary
        detection runs before owner inference), so session is the only
        meaningful query dimension.

        Which of them a read sees is the access policy's (section 9.3, row 3):
        a group scope sees a bound session's messages inside the asker's
        windows; a personal scope sees a session's buffer only when its owner
        is a sender of that session, and never a group session's.
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
        self._validate_components()
        scope = await self._read_scope(req, None)
        where = compile_scoped(req.filters, scope.predicate)
        ranked = await self._search_episodes(req, where, decider=decider, scope=scope)
        return ranked.episodes

    async def _search_episodes(
        self,
        req: SearchRequest,
        where: Filter,
        *,
        decider: RoundDecider | None = None,
        question_image_ids: Sequence[str] = (),
        scope: ReadScope | None = None,
    ) -> _Ranked:
        """The episodes and the pictures ranked with them, by multi-round.

        A group scope's loop also follows the core's stored links (section
        8.3) when ``[search] expansion`` is on.
        """
        expander = self._expander(scope, where)
        result = await run_llm_multiround(
            _question(req, scope),
            # A project scope names no owner of its own: the scope's first group.
            owner_id=scope.owner_id if scope is not None else req.owner_id,
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
        return _Ranked(
            result.episodes, result.images, result.coverage, list(result.core)
        )

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
        self, req: SearchRequest, scope: ReadScope
    ) -> SearchGroupProfile | None:
        """The visible current group-profile items (section 5.3), each kind capped.

        On by default for a group scope (``include_group_profile``): it is
        what grounds a request about the group itself -- its purpose, its
        current plan, who does what -- when the ranking found little.
        """
        wanted = req.include_group_profile is not False
        if not wanted or self._group_profile is None:
            return None
        stored = await visible_profile_items(self._group_profile, scope)
        attached, _view = visible_profile(
            stored, cap=self._group_search.group_profile_cap
        )
        return SearchGroupProfile(
            topics=with_audience(scope, attached.topics),
            decisions=with_audience(scope, attached.decisions),
            dissent=with_audience(scope, attached.dissent),
            roles=with_audience(scope, attached.roles),
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

    # ── Component guards ────────────────────────────────────────────

    def _validate_components(self) -> None:
        """Fail fast when the embedding provider is missing.

        Multi-round recall embeds every sub-query, so without an embedder the
        request is refused with :class:`ProviderNotConfiguredError` -> HTTP
        422 and a toml-pointing message, instead of an opaque
        ``AttributeError('NoneType')`` deep in recall. The LLM is never
        checked: the application is not built without one.
        """
        if self._embedding is None:
            raise ProviderNotConfiguredError(
                provider="embedding", feature=SearchMethod.LLM_MULTIROUND.value
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
