"""Every version of one memory: the use case behind ``GET /memory/history``.

Given any id a read returned, the answer is the whole chain that id belongs
to, current version first, then the older ones newest first. Each version
says its state -- ``current``, ``superseded`` (a newer version replaced it),
``merged`` (a merge of several replaced it), ``forgotten`` -- its time, what
it rests on, who wrote it when a person did, and the id of the version that
replaced it (``next``).

Read from markdown and the forget log only -- no model call, no index:

* an episode or a picture memory: the owner's files of the kind and their
  ``deprecated_entries`` marks, followed both ways
  (:func:`~._records.linear_chain`);
* a scene: every version of the scene (``scene_id``), by version number;
* a profile item (``user.md``, a member's profile in a group): the current
  item and the values it replaced, through ``supersedes`` / ``superseded_by``
  in the profile's history; on ``user.md`` also a value a live forget took
  out (``forgotten``), from the forget log.

Inside a group the answer holds only the versions the asker may see (their
membership windows, docs/group-memory.md, section 9.1); a record none of
whose versions they may see is answered exactly as one that does not exist.
What a forget inside a group took out is not in the files any more, and not
in a history either.
"""

from __future__ import annotations

__all__ = ["HistoryReader", "MemoryHistory", "MemoryVersion"]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from everos.core.errors import NotFoundError
from everos.memory.access import ScopeRequest, visible
from everos.memory.extract.pipeline import profile_items
from everos.memory.models import Episode, ImageMemory, Profile, Scene

from ._records import (
    RecordKind,
    Version,
    image_chain,
    kind_of,
    linear_chain,
    memory_id,
    scene_chain,
)

if TYPE_CHECKING:
    from everos.memory.access import AccessPolicy, Identity, ReadScope
    from everos.memory.deps import Stores
    from everos.memory.ports import Scope, Stored

type VersionState = Literal["current", "superseded", "forgotten", "merged"]


@dataclass(frozen=True)
class MemoryVersion:
    """One version of a memory.

    Attributes:
        id: The version's id (``<owner>_<entry>``, or a profile item's id).
        state: ``current``, ``superseded``, ``merged`` or ``forgotten``.
        timestamp: The memory's own time (epoch ms): when what it says
            happened, the same for every version of an episode.
        next: The version that replaced this one; ``None`` for the current
            one, a forgotten one, or a profile value retired without one.
        supersedes: The version this one replaced, where it names one.
        edited_by: Who wrote this version through ``POST /memory/update``;
            ``None`` for a version a model wrote.
        edited_at: When (epoch ms).
        held: Whether the model may not change it.
        replaced_at: When a profile value was replaced (epoch ms).
        reason: Why a model retired a profile value, when it said.
        sources: What the version rests on: conversation, segments,
            speakers, pictures, a scene's episodes.
        content: What it says.
    """

    id: str
    state: VersionState
    timestamp: int | None
    next: str | None = None
    supersedes: str | None = None
    edited_by: str | None = None
    edited_at: int | None = None
    held: bool = False
    replaced_at: int | None = None
    reason: str | None = None
    sources: Mapping[str, Any] = field(default_factory=dict)
    content: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryHistory:
    """The versions of one memory, current first.

    Attributes:
        kind: ``episode``, ``image_memory``, ``scene``, ``profile_item`` or
            ``group_profile_item``.
        id: The id asked for.
        current: The current version's id; ``None`` when the memory is
            forgotten or was retired.
        versions: Every version, current (or newest) first.
        member_id: The member whose profile holds the item, in a group.
    """

    kind: RecordKind
    id: str
    current: str | None
    versions: tuple[MemoryVersion, ...]
    member_id: str | None = None


class HistoryReader:
    """Reads the versions of one memory.

    Args:
        stores: Markdown and system state.
        access: Read scopes; inside a group, which versions the asker sees.
    """

    def __init__(self, *, stores: Stores, access: AccessPolicy) -> None:
        self._stores = stores
        self._access = access

    async def history(
        self, *, scope: Scope, identity: Identity, record_id: str
    ) -> MemoryHistory:
        """The versions of ``record_id`` the asker may see, current first.

        Raises:
            InvalidInputError: The id names no record kind; no asker for a
                group.
            ForbiddenError: A personal memory read by someone else; a group the
                asker is not a current member of.
            NotFoundError: No such record, or none of its versions visible.
        """
        kind, entry_id = kind_of(scope, record_id)
        reads = await self._access.read_scope(
            identity,
            ScopeRequest(
                app_id=scope.app_id,
                project_id=scope.project_id,
                group_ids=(scope.owner_id,),
            )
            if scope.owner_type == "group"
            else ScopeRequest(
                app_id=scope.app_id,
                project_id=scope.project_id,
                user_id=scope.owner_id,
            ),
        )
        if kind == "episode":
            chain = linear_chain(entry_id, await self._stores.episode.read_all(scope))
            versions = [_episode_version(scope, v) for v in chain.versions]
        elif kind == "image_memory":
            chain = image_chain(entry_id, await self._picture_rows(scope, entry_id))
            versions = [_picture_version(scope, v) for v in chain.versions]
        elif kind == "scene":
            chain = scene_chain(entry_id, await self._stores.scene.read_all(scope))
            versions = [_scene_version(scope, v) for v in chain.versions]
        elif kind == "group_profile_item":
            versions = await self._group_profile_item(scope, entry_id)
        elif scope.owner_type == "group":
            return await self._member_item(scope, reads, entry_id)
        else:
            versions = await self._profile_item(scope, entry_id)
        if scope.owner_type == "group":
            versions = [v for v in versions if _visible(reads, scope, v)]
        if not versions:
            raise NotFoundError(f"no memory {memory_id(scope, entry_id)!r}")
        current = next((v.id for v in versions if v.state == "current"), None)
        return MemoryHistory(
            kind=kind,
            id=memory_id(scope, entry_id),
            current=current,
            versions=tuple(versions),
        )

    async def _picture_rows(
        self, scope: Scope, entry_id: str
    ) -> list[Stored[ImageMemory]]:
        """The occurrences of the picture ``entry_id`` is a memory of."""
        found = await self._stores.image_memory.read_many(scope, [entry_id])
        if not found:
            return []
        rows = await self._stores.image_memory.read_all(scope)
        image_id = found[0].item.image_id
        return [row for row in rows if row.item.image_id == image_id]

    # ── profile items ────────────────────────────────────────────────────

    async def _profile_item(self, scope: Scope, item_id: str) -> list[MemoryVersion]:
        profile = await self._stores.profile.read(scope)
        history = profile.history if profile is not None else []
        forgotten = await self._forgotten_items(scope)
        return _item_versions(profile, history, item_id, forgotten)

    async def _member_item(
        self, scope: Scope, reads: ReadScope, item_id: str
    ) -> MemoryHistory:
        for member_id in await self._stores.member_profile.members(scope):
            held = await self._stores.member_profile.read(scope, member_id)
            if held is None or not profile_items.find_item(
                held.profile, held.history, item_id
            ):
                continue
            versions = [
                v
                for v in _item_versions(held.profile, held.history, item_id, {})
                if _visible(reads, scope, v)
            ]
            if not versions:
                break
            current = next((v.id for v in versions if v.state == "current"), None)
            return MemoryHistory(
                kind="profile_item",
                id=item_id,
                current=current,
                versions=tuple(versions),
                member_id=member_id,
            )
        raise NotFoundError(f"no profile item {item_id!r} in the group")

    async def _forgotten_items(self, scope: Scope) -> dict[str, Mapping[str, Any]]:
        """Profile values a live forget of the user took out, by item id."""
        found: dict[str, Mapping[str, Any]] = {}
        for entry in await self._stores.forget_log.live_entries(
            app_id=scope.app_id, project_id=scope.project_id, owner_id=scope.owner_id
        ):
            for _bucket, item in entry.removed_profile_items:
                item_id = item.get("item_id")
                if isinstance(item_id, str):
                    found.setdefault(item_id, item)
        return found

    async def _group_profile_item(
        self, scope: Scope, entry_id: str
    ) -> list[MemoryVersion]:
        """A group-profile item and the items it replaced or that replaced it."""
        items = {
            s.ref.entry_id: s for s in await self._stores.group_profile.items(scope)
        }
        if entry_id not in items:
            return []
        successor = {
            s.item.supersedes: key for key, s in items.items() if s.item.supersedes
        }
        newest = entry_id
        seen = {newest}
        while newest in successor and successor[newest] not in seen:
            newest = successor[newest]
            seen.add(newest)
        versions: list[MemoryVersion] = []
        current: str | None = newest
        while current is not None and current in items and len(versions) < 256:
            stored = items[current]
            item = stored.item
            state: VersionState = (
                "forgotten"
                if item.status == "forgotten"
                else "superseded"
                if item.superseded_by or current != newest
                else "current"
            )
            versions.append(
                MemoryVersion(
                    id=memory_id(scope, current),
                    state=state,
                    timestamp=item.observed_at,
                    next=memory_id(scope, successor[current])
                    if current in successor
                    else None,
                    supersedes=memory_id(scope, item.supersedes)
                    if item.supersedes
                    else None,
                    sources={
                        "memcell_ids": sorted({s.memcell_id for s in item.sources}),
                        "speaker_ids": sorted({s.speaker_id for s in item.sources}),
                        "source_start": item.source_start,
                        "source_end": item.source_end,
                    },
                    content={
                        "kind": item.kind,
                        "statement": item.statement,
                        "name": item.name,
                        "summary": item.summary,
                    },
                )
            )
            current = item.supersedes
        return versions


# ── one version as the answer shows it ───────────────────────────────────


def _edit_of(item: Episode | ImageMemory | Scene) -> dict[str, Any]:
    return {
        "edited_by": item.edited_by,
        "edited_at": item.edited_at,
        "held": item.held,
    }


def _episode_version(scope: Scope, version: Version[Episode]) -> MemoryVersion:
    item = version.stored.item
    return MemoryVersion(
        id=memory_id(scope, version.entry_id),
        state=version.state,
        timestamp=item.timestamp,
        next=memory_id(scope, version.successor) if version.successor else None,
        supersedes=memory_id(scope, item.supersedes) if item.supersedes else None,
        sources={
            "session_id": item.session_id,
            "memcell_id": item.parent_id if item.parent_type == "memcell" else None,
            "parent_type": item.parent_type,
            "sender_ids": list(item.sender_ids),
            "image_ids": list(item.image_ids),
            "source_start": item.source_start,
        },
        content={
            "subject": item.subject,
            "summary": item.summary,
            "content": item.episode,
        },
        **_edit_of(item),
    )


def _picture_version(scope: Scope, version: Version[ImageMemory]) -> MemoryVersion:
    item = version.stored.item
    return MemoryVersion(
        id=memory_id(scope, version.entry_id),
        state=version.state,
        timestamp=item.timestamp,
        next=memory_id(scope, version.successor) if version.successor else None,
        supersedes=memory_id(scope, item.supersedes) if item.supersedes else None,
        sources={
            "session_id": item.session_id,
            "memcell_id": item.parent_id or None,
            "image_id": item.image_id,
            "sender_ids": list(item.sender_ids),
        },
        content={"description": item.description, "ocr": item.ocr},
        **_edit_of(item),
    )


def _scene_version(scope: Scope, version: Version[Scene]) -> MemoryVersion:
    item = version.stored.item
    return MemoryVersion(
        id=memory_id(scope, version.entry_id),
        state=version.state,
        timestamp=item.timestamp,
        next=memory_id(scope, version.successor) if version.successor else None,
        supersedes=memory_id(scope, item.supersedes) if item.supersedes else None,
        sources={
            "scene_id": item.scene_id,
            "version": item.version,
            "member_ids": [memory_id(scope, m) for m in item.member_ids],
            "sender_ids": list(item.sender_ids),
            "source_start": item.source_start,
        },
        content={
            "subject": item.subject,
            "summary": item.summary,
            "content": item.narrative,
        },
        **_edit_of(item),
    )


def _item_versions(
    profile: Profile | None,
    history: Sequence[Mapping[str, Any]],
    item_id: str,
    forgotten: Mapping[str, Mapping[str, Any]],
) -> list[MemoryVersion]:
    """The chain of profile values ``item_id`` belongs to, current first.

    Forward through ``superseded_by`` to the newest value, then back through
    ``supersedes``. A value a live forget took out is listed ``forgotten``
    (on ``user.md``); its links end there.
    """
    current = {
        str((item.model_extra or {}).get("item_id")): item.model_dump()
        for bucket in profile_items.BUCKETS
        for item in (getattr(profile, bucket) if profile is not None else [])
    }
    replaced: dict[str, Mapping[str, Any]] = {}
    for entry in history:  # newest first: the first entry of an id is its last
        if isinstance(entry, Mapping) and isinstance(entry.get("item_id"), str):
            replaced.setdefault(entry["item_id"], entry)

    def look(key: str) -> tuple[VersionState, Mapping[str, Any]] | None:
        if key in current:
            return "current", current[key]
        if key in replaced:
            entry = replaced[key]
            return ("merged" if entry.get("merged_into") else "superseded"), entry
        if key in forgotten:
            return "forgotten", forgotten[key]
        return None

    if look(item_id) is None:
        return []
    newest, seen = item_id, {item_id}
    while True:
        found = look(newest)
        assert found is not None
        nxt = found[1].get("superseded_by")
        if not isinstance(nxt, str) or nxt in seen or look(nxt) is None:
            break
        newest = nxt
        seen.add(nxt)
    versions: list[MemoryVersion] = []
    key: str | None = newest
    walked: set[str] = set()
    while key is not None and key not in walked and len(versions) < 256:
        walked.add(key)
        found = look(key)
        if found is None:
            break
        state, data = found
        merged = data.get("merged_into")
        nxt = data.get("superseded_by")
        versions.append(
            MemoryVersion(
                id=key,
                state=state,
                timestamp=_int(data.get("observed_at")) or _int(data.get("source_end")),
                next=nxt
                if isinstance(nxt, str)
                else (merged[0] if isinstance(merged, list) and merged else None),
                supersedes=data.get("supersedes")
                if isinstance(data.get("supersedes"), str)
                else None,
                edited_by=data.get("edited_by"),
                edited_at=_int(data.get("edited_at")),
                held=data.get("held") is True,
                replaced_at=_int(data.get("replaced_at")),
                reason=data.get("reason")
                if isinstance(data.get("reason"), str)
                else None,
                sources={
                    "memcell_ids": list(data.get("source_memcell_ids") or []),
                    "image_ids": profile_items.item_source_images(data),
                    "source_start": _int(data.get("source_start")),
                    "source_end": _int(data.get("source_end")),
                },
                content={
                    k: data[k]
                    for k in ("description", "category", "trait")
                    if data.get(k) is not None
                },
            )
        )
        link = data.get("supersedes")
        key = link if isinstance(link, str) else None
    return versions


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _visible(reads: ReadScope, scope: Scope, version: MemoryVersion) -> bool:
    """Whether a group asker's windows hold the version's whole span."""
    start = version.sources.get("source_start")
    end = version.sources.get("source_end") or version.timestamp
    if not isinstance(end, int):
        return False
    return visible(
        reads,
        owner_type="group",
        owner_id=scope.owner_id,
        source_start=start if isinstance(start, int) else end,
        source_end=end,
    )
