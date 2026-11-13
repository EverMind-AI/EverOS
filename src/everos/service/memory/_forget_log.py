"""The forget log's record of one forget, and what the other forgets read from it.

One format for every owner type (``undo_json`` with ``"format": 2``): the
marks the forget wrote with what each entry carried before, the versions it
wrote in place of others, what it took out of profile documents with each
item's place, the messages, files, clusters and queues it changed. An undo
reads it back (:mod:`._forget_undo`); so do the other forgets of the owner and
its reads:

* an entry stays forgotten while any live forget marked it
  (:meth:`Logs.named_by_others`);
* a user's profile item goes once every segment it rests on is forgotten,
  this forget's or one before it (:meth:`Logs.forgotten_segments`), and comes
  back with the undo that remembers one of them again
  (:meth:`Logs.removed_by_others`);
* a group's history leaves out the versions a live forget retired
  (:meth:`Logs.retired`).

Rows written before this format are read too, so stores built before it still
undo: a user's (a picture's or a conversation's columns, or an ``undo`` of
records, everything or a condition) and a group's (``undo`` with what it took
out of markdown, ``withdrawn``).
"""

from __future__ import annotations

__all__ = [
    "FORMAT",
    "Logs",
    "Named",
    "Removed",
    "is_current_format",
    "is_legacy_group",
    "named_in",
    "segments_of",
]

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from everos.memory.extract.pipeline import profile_items
from everos.memory.ports import UNDO_FORMAT, SessionKey, retired_in

if TYPE_CHECKING:
    from everos.memory.deps import Stores
    from everos.memory.ports import ForgetEntry, Scope

FORMAT = UNDO_FORMAT
"""``undo_json["format"]`` of the one log format."""


@dataclass(frozen=True)
class Named:
    """The entries forgets marked forgotten, by kind (entry ids)."""

    episodes: set[str] = field(default_factory=set)
    image_memories: set[str] = field(default_factory=set)
    scenes: set[str] = field(default_factory=set)
    group_profile: set[str] = field(default_factory=set)

    def of(self, kind: str) -> set[str]:
        return {
            "episode": self.episodes,
            "image_memory": self.image_memories,
            "scene": self.scenes,
            "group_profile": self.group_profile,
        }[kind]


@dataclass(frozen=True)
class Removed:
    """One profile value a forget took out of a document.

    Attributes:
        person_id: Whose document.
        bucket: ``explicit_info``, ``implicit_traits``, ``history`` or
            :data:`~everos.memory.extract.pipeline.profile_items.SUMMARY`.
        item: The value as it was stored.
        position: Its place in its bucket; ``None`` in a row written before
            the log kept places (put back by its description then).
    """

    person_id: str
    bucket: str
    item: dict[str, Any]
    position: int | None = None


def is_current_format(entry: ForgetEntry) -> bool:
    return (entry.undo or {}).get("format") == FORMAT


def is_legacy_group(entry: ForgetEntry) -> bool:
    """A group forget logged when forgets took records out of markdown."""
    return (
        entry.owner_type == "group"
        and entry.undo is not None
        and not is_current_format(entry)
    )


def named_in(entry: ForgetEntry) -> Named:
    """The entries one forget marked forgotten, by kind.

    A row of this format lists its marks. An older user's row: a forget by
    records, owner or condition kept its sets in ``undo``; a conversation's
    named its episodes in ``episode_ids`` and its picture memories in
    ``entry_ids``; a picture's its picture memories (its episodes were
    superseded, not forgotten). An older group's row marked nothing.
    """
    undo = entry.undo or {}
    if is_current_format(entry):
        marked = undo.get("marked") or {}
        return Named(
            episodes={str(r[1]) for r in marked.get("episode") or []},
            image_memories={str(r[1]) for r in marked.get("image_memory") or []},
            scenes={str(r[1]) for r in marked.get("scene") or []},
            group_profile={str(r[1]) for r in marked.get("group_profile") or []},
        )
    if is_legacy_group(entry):
        return Named()
    if undo:
        return Named(
            episodes={str(row[1]) for row in undo.get("episodes") or []},
            image_memories={str(row[1]) for row in undo.get("image_memories") or []},
            scenes={str(row[1]) for row in undo.get("scenes") or []},
        )
    if entry.target == "session":
        return Named(
            episodes={old for old, _ in entry.episode_ids},
            image_memories=set(entry.entry_ids),
        )
    if entry.target == "image":
        return Named(image_memories=set(entry.entry_ids))
    return Named()


def removed_in(entry: ForgetEntry) -> list[Removed]:
    """What one forget took out of profile documents (both formats)."""
    if is_current_format(entry):
        found: list[Removed] = []
        for change in (entry.undo or {}).get("profiles") or []:
            person = str(change["person"])
            found += [
                Removed(person, str(bucket), dict(item), int(n))
                for bucket, n, item in change.get("items") or []
            ]
            found += [
                Removed(person, "history", dict(item), int(n))
                for n, item in change.get("history") or []
            ]
            if change.get("summary"):
                found.append(
                    Removed(
                        person,
                        profile_items.SUMMARY,
                        {"summary": str(change["summary"])},
                    )
                )
        return found
    return [
        Removed(entry.owner_id, bucket, dict(item))
        for bucket, item in entry.removed_profile_items
    ]


async def segments_of(
    stores: Stores, scope: Scope, sessions: Collection[str]
) -> set[str]:
    """The memcell ids of the segments of ``sessions`` in ``scope``'s space."""
    found: set[str] = set()
    for session_id in sessions:
        cells = await stores.memcell.cells_in_session(
            SessionKey(
                app_id=scope.app_id,
                project_id=scope.project_id,
                session_id=session_id,
            )
        )
        found.update(cells)
    return found


class Logs:
    """What one owner's live forgets say, read for another forget or a read."""

    def __init__(self, stores: Stores) -> None:
        self._stores = stores

    async def live(self, scope: Scope) -> list[ForgetEntry]:
        """The owner's forgets no undo reversed, newest first."""
        return await self._stores.forget_log.live_entries(
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            owner_type=scope.owner_type,
        )

    async def named_by_others(self, scope: Scope, *, except_id: str) -> Named:
        """What the owner's live forgets other than ``except_id`` keep forgotten."""
        named = Named()
        for entry in await self.live(scope):
            if entry.forget_id == except_id:
                continue
            found = named_in(entry)
            named.episodes.update(found.episodes)
            named.image_memories.update(found.image_memories)
            named.scenes.update(found.scenes)
            named.group_profile.update(found.group_profile)
        return named

    async def forgotten_segments(
        self,
        scope: Scope,
        *,
        except_id: str | None = None,
        except_session: str | None = None,
    ) -> set[str]:
        """The segments a user's live forgets have forgotten whole, but one.

        A conversation's forget covers every segment of the conversation; a
        forget of this format or by records, owner or condition the segments
        it logged.
        """
        sessions: list[str] = []
        found: set[str] = set()
        for entry in await self.live(scope):
            if entry.forget_id == except_id:
                continue
            if entry.target == "session" and not is_current_format(entry):
                if entry.target_id != except_session:
                    sessions.append(entry.target_id)
            elif entry.undo:
                if entry.target == "session" and entry.target_id == except_session:
                    continue
                found.update(str(s) for s in entry.undo.get("segments") or [])
        return found | await segments_of(self._stores, scope, dict.fromkeys(sessions))

    async def forgotten_items(self, scope: Scope) -> set[str]:
        """The ids of profile values the owner's live forgets took out."""
        return {
            str(r.item["item_id"])
            for entry in await self.live(scope)
            for r in removed_in(entry)
            if isinstance(r.item.get("item_id"), str)
        }

    async def removed_by_others(
        self, scope: Scope, segments: Collection[str], *, except_id: str
    ) -> list[Removed]:
        """What a user's other live forgets took out by the segment rule,
        resting on one of ``segments``.

        Undoing a forget remembers its segments again; an item another forget
        took out because every segment it rests on was forgotten -- these
        among them -- is no longer wholly forgotten, so it comes back with
        this undo. An item a forget named by id, or took with a picture,
        stays out.
        """
        wanted = set(segments)
        found: list[Removed] = []
        for other in await self.live(scope):
            if other.forget_id == except_id:
                continue
            removed = removed_in(other)
            if is_current_format(other):
                places = {
                    (str(change["person"]), str(bucket), int(n))
                    for change in (other.undo or {}).get("profiles") or []
                    for bucket, n in change.get("by_segments") or []
                }
                candidates = [
                    r for r in removed if (r.person_id, r.bucket, r.position) in places
                ]
            elif other.target == "session":
                candidates = removed
            elif other.undo:
                by_segments = {str(i) for i in other.undo.get("segment_items") or []}
                candidates = [
                    r for r in removed if r.item.get("item_id") in by_segments
                ]
            else:
                continue
            found += [r for r in candidates if profile_items.rests_on(r.item, wanted)]
        return found

    async def retired(self, scope: Scope) -> set[str]:
        """The entry ids a group's live forgets keep out of its history."""
        found: set[str] = set()
        for entry in await self.live(scope):
            found |= retired_in(entry)
        return found
