"""GetManager — top-level orchestrator for ``POST /api/v2/memory/get``.

Hard partition by ``(owner_type, memory_type)`` (validated by
:class:`GetRequest`):

* ``user`` + ``episode``       → ``data.episodes`` (never a scene version)
* ``user`` + ``scene``         → ``data.scenes``: the user's current scene
  versions, rows of the episode table (``parent_type = scene``)
* ``user`` + ``profile``       → ``data.profiles`` (one-row KV fetch
  from the ``user_profile`` table; at most one item)
* ``group`` + ``episode``      → ``data.episodes`` inside the asker's windows
* ``group`` + ``scene``        → ``data.scenes``: the scene versions, which share
  the episode table (``parent_type = scene``) and its windows
* ``group`` + ``image_memory`` → ``data.images`` inside the asker's windows
* ``group`` + ``member_profile`` (``member_id``) → ``data.profiles``: the
  member's profile in each group of the scope, its items cut to the windows
* ``group`` + ``group_profile`` → ``data.group_profile``: the items whose
  sources lie in the asker's windows, current ones only, or every version but
  the forgotten with ``include_superseded``
* ``user`` + ``episode`` / ``scene`` with ``include_groups`` → the user's
  records and those of the user's current groups inside the user's windows,
  one list sorted and paged together, each item naming its owner

Reads only — never writes. The owner part of every ``where`` is the access
policy's read scope (:mod:`everos.memory.access`); filters are compiled onto
it through :func:`compile_scoped_for_get` so the column allow-list stays
shared with :mod:`memory.search`. Pagination + sort run in the index
behind the injected ports.
"""

from __future__ import annotations

from everos.core.context import resolve_request_id
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import from_timestamp_ms, to_display_tz
from everos.memory.access import (
    AccessPolicy,
    ReadScope,
    ScopeRequest,
    resolve_identity,
)
from everos.memory.models import (
    Episode,
    ImageMemory,
)
from everos.memory.ports import (
    EpisodeIndex,
    GroupProfileIndex,
    ImageMemoryIndex,
    Page,
    ProfileIndex,
    Scope,
    Stored,
    all_of,
    any_of,
    eq,
    ne,
)
from everos.memory.search import (
    fetch_member_profile,
    fetch_user_profile,
    group_source,
    item_versions,
    shape_profile_item,
    visible_profile_items,
)

from .dto import (
    GetData,
    GetEpisodeItem,
    GetImageItem,
    GetMemoryType,
    GetProfileItem,
    GetRequest,
    GetResponse,
)
from .filters_adapter import compile_scoped_for_get

logger = get_logger(__name__)


class GetManager:
    """Dispatch ``GetRequest`` to the matching index and shape the stored
    domain objects into the public DTO.

    Args:
        episode_index: Episode reads.
        profile_index: User and member-in-group profile reads.
        image_index: Picture-memory reads (group listings); ``None`` refuses
            them.
        group_profile_index: Group-profile reads; ``None`` refuses them.
    """

    def __init__(
        self,
        *,
        episode_index: EpisodeIndex,
        profile_index: ProfileIndex,
        access: AccessPolicy,
        image_index: ImageMemoryIndex | None = None,
        group_profile_index: GroupProfileIndex | None = None,
    ) -> None:
        self._access = access
        self._images = image_index
        self._group_profile = group_profile_index
        self._ep = episode_index
        self._profile = profile_index

    # ── Public entry ─────────────────────────────────────────────────

    async def get(
        self, req: GetRequest, *, asserted_asker: str | None = None
    ) -> GetResponse:
        """One page of the requested kind, inside the asker's read scope.

        Raises:
            ForbiddenError: The asker may not read the scope (403).
            InvalidInputError: A group scope without an asker (422).
        """
        request_id = resolve_request_id()
        descending = req.sort_order == "desc"
        scope = await self._access.read_scope(
            resolve_identity(header=asserted_asker, body_asker=req.asker_id),
            ScopeRequest(
                app_id=req.app_id,
                project_id=req.project_id,
                user_id=req.user_id,
                group_ids=req.scope_group_ids,
                audience=req.audience,
            ),
        )
        where = compile_scoped_for_get(req.filters, scope.predicate)
        page = Page(
            sort_by=req.sort_by,
            number=req.page,
            size=req.page_size,
            descending=descending,
        )
        if req.include_groups:
            data = await self._list_with_groups(req, scope, page)
            return GetResponse(request_id=request_id, data=data)

        match req.memory_type:
            case GetMemoryType.EPISODE:
                # Scene versions share the episode table; they list as scenes.
                where = all_of(where, ne("parent_type", "scene"))
                episodes = await self._ep.paginate(where, page)
                items = [self._shape_episode(e) for e in episodes.items]
                data = GetData(
                    episodes=items,
                    total_count=episodes.total,
                    count=len(items),
                )
            case GetMemoryType.SCENE:
                scenes = await self._ep.paginate(
                    all_of(where, eq("parent_type", "scene")), page
                )
                items = [self._shape_episode(e) for e in scenes.items]
                data = GetData(scenes=items, total_count=scenes.total, count=len(items))
            case GetMemoryType.IMAGE_MEMORY:
                if self._images is None:
                    raise RuntimeError("picture listings need the picture index")
                images = await self._images.paginate(where, page)
                shaped = [_shape_image(i) for i in images.items]
                data = GetData(
                    images=shaped, total_count=images.total, count=len(shaped)
                )
            case GetMemoryType.MEMBER_PROFILE:
                assert req.member_id is not None  # the request validates it
                members = await self._member_profiles(scope, req.member_id)
                data = GetData(
                    profiles=members, total_count=len(members), count=len(members)
                )
            case GetMemoryType.GROUP_PROFILE:
                if self._group_profile is None:
                    raise RuntimeError("group-profile listings need their index")
                found = await (
                    item_versions(self._group_profile, scope)
                    if req.include_superseded
                    else visible_profile_items(self._group_profile, scope)
                )
                if not descending:
                    found.reverse()
                start = (req.page - 1) * req.page_size
                shaped = [
                    shape_profile_item(i) for i in found[start : start + req.page_size]
                ]
                data = GetData(
                    group_profile=shaped, total_count=len(found), count=len(shaped)
                )
            case GetMemoryType.PROFILE:
                profiles = await self._fetch_profile(req)
                data = GetData(
                    profiles=profiles,
                    total_count=len(profiles),
                    count=len(profiles),
                )

        return GetResponse(request_id=request_id, data=data)

    async def _list_with_groups(
        self, req: GetRequest, scope: ReadScope, page: Page
    ) -> GetData:
        """The user's episodes or scenes and their groups', as one list.

        ``include_groups``: the groups come from the access policy
        (:meth:`AccessPolicy.member_groups`), each read as a group listing by
        the user reads it. Their predicate is ORed onto the personal one, so
        one index query sorts, counts and pages both; every item names its
        owner.
        """
        groups = await self._access.member_groups(scope, only=req.include_group_ids)
        pooled = (
            scope.predicate
            if groups is None
            else any_of(scope.predicate, groups.predicate)
        )
        where = compile_scoped_for_get(req.filters, pooled)
        scenes = req.memory_type == GetMemoryType.SCENE
        kind = eq("parent_type", "scene") if scenes else ne("parent_type", "scene")
        found = await self._ep.paginate(all_of(where, kind), page)
        items = [_owned(self._shape_episode(e)) for e in found.items]
        if scenes:
            return GetData(scenes=items, total_count=found.total, count=len(items))
        return GetData(episodes=items, total_count=found.total, count=len(items))

    # ── Shapers ──────────────────────────────────────────────────────

    @staticmethod
    def _shape_episode(stored: Stored[Episode]) -> GetEpisodeItem:
        episode = stored.item
        group = stored.scope.owner_type == "group"
        return GetEpisodeItem(
            id=stored.ref.id,
            user_id=None if group else episode.owner_id,
            owner_type="group" if group else None,
            group_id=episode.owner_id if group else None,
            source=group_source(episode) if group else None,
            app_id=stored.scope.app_id,
            project_id=stored.scope.project_id,
            session_id=episode.session_id,
            timestamp=to_display_tz(from_timestamp_ms(episode.timestamp)),
            sender_ids=episode.sender_ids,
            subject_ids=list(episode.subject_ids),
            origin=episode.origin,
            summary=episode.summary,
            subject=episode.subject,
            episode=episode.episode,
            type="Scene" if episode.parent_type == "scene" else "Conversation",
        )

    # ── Profile ──────────────────────────────────────────────────────

    async def _member_profiles(
        self, scope: ReadScope, member_id: str
    ) -> list[GetProfileItem]:
        """The member's profile in each group of ``scope``, cut to its windows."""
        items: list[GetProfileItem] = []
        for view in scope.groups:
            found = await fetch_member_profile(
                self._profile, scope, view.group_id, member_id
            )
            if found is not None:
                items.append(GetProfileItem(**found.fields()))
        return items

    async def _fetch_profile(self, req: GetRequest) -> list[GetProfileItem]:
        """The owner's single profile, as ``/search`` fetches it.

        Profile is one-row-per-owner KV — there is no pagination / sort /
        filter surface, so at most one item is returned. Empty list (not 404)
        keeps the response valid during the cold-start window before a
        profile has been synthesised.
        """
        if not req.owner_id:
            return []
        found = await fetch_user_profile(
            self._profile,
            Scope(
                app_id=req.app_id,
                project_id=req.project_id,
                owner_id=req.owner_id,
                owner_type="user",
            ),
        )
        if found is None:
            logger.debug("get_profile_miss", owner_id=req.owner_id)
            return []
        return [GetProfileItem(**found.fields())]


def _owned(item: GetEpisodeItem) -> GetEpisodeItem:
    """A listed record naming its owner (``include_groups``)."""
    owner = "group" if item.group_id is not None else "user"
    return item.model_copy(update={"owner_type": owner})


def _shape_image(stored: Stored[ImageMemory]) -> GetImageItem:
    image = stored.item
    return GetImageItem(
        id=stored.ref.id,
        entry_id=stored.ref.entry_id,
        image_id=image.image_id,
        group_id=stored.scope.owner_id,
        app_id=stored.scope.app_id,
        project_id=stored.scope.project_id,
        session_id=image.session_id,
        timestamp=to_display_tz(from_timestamp_ms(image.timestamp)),
        sender_ids=list(image.sender_ids),
        subject_ids=list(image.subject_ids),
        origin=image.origin,
        description=image.description,
        ocr=image.ocr,
        name=image.name,
        original_path=image.original_relpath,
        media_type=image.media_type,
        width=image.width,
        height=image.height,
    )
