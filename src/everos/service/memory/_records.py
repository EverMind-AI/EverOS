"""Records by id, and the chain of versions one record id belongs to.

``POST /memory/update``, ``GET /memory/history`` and a forget by id all take
the id a read returned: ``<owner>_<entry_id>`` (or the bare entry id) for a
daily-log record -- an episode, a picture memory, a scene version -- and the
``item_id`` (``pi_<hex>``) for a profile item. The entry id's prefix names the
kind; the owner's type picks the prefixes (a group's records carry a ``g``).

A daily-log record is one version of a memory. The versions are linked by the
marks in their files' ``deprecated_entries``: a version names its successor's
entry id, or the forget marker, or nothing while it is current. A version a
person wrote also names the version it replaced (``supersedes``). The chain is
read from markdown, never from the index, so it never waits for the cascade.
"""

from __future__ import annotations

__all__ = [
    "Chain",
    "RecordKind",
    "Version",
    "entry_id_of",
    "episode_end",
    "image_chain",
    "kind_of",
    "linear_chain",
    "memory_id",
    "picture_end",
    "scene_chain",
]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from everos.core.errors import InvalidInputError
from everos.memory.models import FORGOTTEN_MARK, Episode, ImageMemory, Scene

if TYPE_CHECKING:
    from everos.memory.deps import Stores
    from everos.memory.ports import Scope, Stored

type RecordKind = Literal[
    "episode", "image_memory", "scene", "profile_item", "group_profile_item"
]

_USER_PREFIXES: Mapping[str, RecordKind] = {
    "ep_": "episode",
    "im_": "image_memory",
    "sc_": "scene",
    "pi_": "profile_item",
}
_GROUP_PREFIXES: Mapping[str, RecordKind] = {
    "gep_": "episode",
    "gim_": "image_memory",
    "gsc_": "scene",
    "gp_": "group_profile_item",
    "pi_": "profile_item",
}

_CHAIN_CAP = 256
"""Versions one chain is followed through; a longer one is cut there."""


def entry_id_of(owner_id: str, record_id: str) -> str:
    """The entry id ``record_id`` names: ``<owner>_<entry>`` loses its owner."""
    prefix = f"{owner_id}_"
    return record_id[len(prefix) :] if record_id.startswith(prefix) else record_id


def kind_of(scope: Scope, record_id: str) -> tuple[RecordKind, str]:
    """``(kind, entry id)`` of one record id in ``scope``'s owner.

    Raises:
        InvalidInputError: The id names no record kind of this owner type.
    """
    entry_id = entry_id_of(scope.owner_id, record_id)
    prefixes = _GROUP_PREFIXES if scope.owner_type == "group" else _USER_PREFIXES
    for prefix, kind in prefixes.items():
        if entry_id.startswith(prefix):
            return kind, entry_id
    raise InvalidInputError(
        f"{record_id!r} is no record id of a {scope.owner_type}: expected an "
        f"id a read returned ({', '.join(prefixes)})"
    )


def memory_id(scope: Scope, entry_id: str) -> str:
    """The id a read shows for an entry: ``<owner>_<entry>``; a profile item's
    ``item_id`` as it is."""
    if entry_id.startswith("pi_"):
        return entry_id
    return f"{scope.owner_id}_{entry_id}"


@dataclass(frozen=True)
class Version[T: (Episode, ImageMemory, Scene)]:
    """One version of a daily-log record, with its mark.

    Attributes:
        stored: The version as its file holds it.
        mark: ``None`` while current, the forget marker, or the successor's
            entry id.
        merged: The successor is a merge of several versions (one an earlier
            2.0 build's Reflection wrote); only meaningful when ``mark`` names
            one.
    """

    stored: Stored[T]
    mark: str | None
    merged: bool = False

    @property
    def entry_id(self) -> str:
        return self.stored.ref.entry_id

    @property
    def state(self) -> Literal["current", "superseded", "forgotten", "merged"]:
        if self.mark is None:
            return "current"
        if self.mark == FORGOTTEN_MARK:
            return "forgotten"
        return "merged" if self.merged else "superseded"

    @property
    def successor(self) -> str | None:
        """The entry id of the next version, ``None`` at the end of the chain."""
        if self.mark is None or self.mark == FORGOTTEN_MARK:
            return None
        return self.mark


@dataclass
class Chain[T: (Episode, ImageMemory, Scene)]:
    """The versions linked to one entry, newest first.

    Attributes:
        versions: Every version of the chain: the end of the chain first (the
            current version, or the last one when the chain was forgotten),
            then the others, newest write first.
        asked: The entry id the chain was read for.
    """

    asked: str
    versions: list[Version[T]] = field(default_factory=list)

    @property
    def newest(self) -> Version[T] | None:
        return self.versions[0] if self.versions else None

    @property
    def current(self) -> Version[T] | None:
        """The live version: the newest one, while it carries no mark."""
        newest = self.newest
        return newest if newest is not None and newest.mark is None else None

    def get(self, entry_id: str) -> Version[T] | None:
        return next((v for v in self.versions if v.entry_id == entry_id), None)


def _write_order(entry_id: str) -> tuple[str, int]:
    """An entry id's place in write order: ``<prefix>_<YYYYMMDD>_<seq>``."""
    parts = entry_id.rsplit("_", 2)
    if len(parts) == 3 and parts[2].isdecimal():
        return parts[1], int(parts[2])
    return entry_id, 0


def linear_chain[T: (Episode, ImageMemory)](
    asked: str, rows: Sequence[tuple[Stored[T], str | None]]
) -> Chain[T]:
    """The chain ``asked`` belongs to, out of the owner's rows of one kind.

    Followed forward through the marks to the newest version and back through
    every version whose mark names one in the chain (a merge has several), so
    the chain is the same whichever of its ids is asked for.
    """
    by_id = {stored.ref.entry_id: (stored, mark) for stored, mark in rows}
    if asked not in by_id:
        return Chain(asked=asked)
    predecessors: dict[str, list[str]] = {}
    for entry_id, (_, mark) in by_id.items():
        if mark and mark != FORGOTTEN_MARK:
            predecessors.setdefault(mark, []).append(entry_id)
    newest = asked
    seen = {asked}
    while len(seen) < _CHAIN_CAP:
        mark = by_id[newest][1]
        if not mark or mark == FORGOTTEN_MARK or mark not in by_id or mark in seen:
            break
        newest = mark
        seen.add(newest)
    members: list[str] = []
    frontier = [newest]
    found = {newest}
    while frontier and len(found) < _CHAIN_CAP:
        entry_id = frontier.pop()
        members.append(entry_id)
        for older in predecessors.get(entry_id, []):
            if older not in found:
                found.add(older)
                frontier.append(older)
    versions: list[Version[T]] = []
    older = sorted((m for m in members if m != newest), key=_write_order, reverse=True)
    for entry_id in [newest, *older]:
        stored, mark = by_id[entry_id]
        successor = by_id.get(mark or "")
        merged = (
            successor is not None
            and getattr(successor[0].item, "parent_type", None) == "cluster"
        )
        versions.append(Version(stored=stored, mark=mark, merged=merged))
    return Chain(asked=asked, versions=versions)


def image_chain(asked: str, rows: Sequence[Stored[ImageMemory]]) -> Chain[ImageMemory]:
    """:func:`linear_chain` over picture memories, whose marks ride on the item."""
    return linear_chain(asked, [(row, row.item.superseded_by) for row in rows])


def scene_chain(
    asked: str, rows: Sequence[tuple[Stored[Scene], str | None]]
) -> Chain[Scene]:
    """Every version of the scene ``asked`` is a version of, newest first."""
    version = next((s for s, _ in rows if s.ref.entry_id == asked), None)
    if version is None:
        return Chain(asked=asked)
    scene_id = version.item.scene_id
    mine = sorted(
        ((s, m) for s, m in rows if s.item.scene_id == scene_id),
        key=lambda pair: (pair[0].item.version, _write_order(pair[0].ref.entry_id)),
        reverse=True,
    )
    return Chain(
        asked=asked,
        versions=[Version(stored=s, mark=m or None) for s, m in mine],
    )


async def episode_end(
    stores: Stores, scope: Scope, entry_id: str
) -> tuple[Stored[Episode], str | None] | None:
    """The last version reached from ``entry_id`` through the marks, and its mark."""
    last: tuple[Stored[Episode], str | None] | None = None
    seen: set[str] = set()
    current = entry_id
    while current not in seen and len(seen) < _CHAIN_CAP:
        seen.add(current)
        found = await stores.episode.read_many(scope, [current])
        if not found:
            break
        stored = found[0]
        mark = (await stores.episode.deprecations(scope, stored.ref.md_path)).get(
            current
        ) or None
        last = (stored, mark)
        if mark is None or mark == FORGOTTEN_MARK:
            break
        current = mark
    return last


async def picture_end(
    stores: Stores, scope: Scope, entry_id: str
) -> tuple[Stored[ImageMemory], str | None] | None:
    """:func:`episode_end` for a picture memory, whose mark rides on the item."""
    last: tuple[Stored[ImageMemory], str | None] | None = None
    seen: set[str] = set()
    current = entry_id
    while current not in seen and len(seen) < _CHAIN_CAP:
        seen.add(current)
        found = await stores.image_memory.read_many(scope, [current])
        if not found:
            break
        stored = found[0]
        mark = stored.item.superseded_by or None
        last = (stored, mark)
        if mark is None or mark == FORGOTTEN_MARK:
            break
        current = mark
    return last
