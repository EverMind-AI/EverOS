"""Reversing a forget that did not complete, from its write-ahead record.

A forget's log entry is written before its first change, and ahead of each
step the run saves into it the change the step makes
(:class:`._forget_apply.Applied`): whatever a failed run -- or a process that
stopped part-way -- may have changed is named there. The reversal
(docs/api.md, "Forget durability"):

1. each version the run wrote in place of another -- recorded, or found from
   the one it was writing (:func:`._forget_apply.writing_key`) -- leaves
   markdown, so the version it replaced is current again (a withdrawal drops
   the mark naming it); its index row goes and both files go back to the
   cascade;
2. a profile change the document never took (it still holds what the change
   took out) is left out;
3. the undo's own steps (:class:`._forget_undo.Undoer`, ``roll_back``) put
   back the rest -- marks that stand, clusters, files, queues, member
   documents, profile values, message marks -- and stamp the entry
   ``rolled_back``.

Each step finds what is still to do, so a reversal cut short is finished by
the next one.
"""

from __future__ import annotations

__all__ = ["Rollback"]

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from everos.memory.extract.pipeline import profile_items

from ._forget_apply import INDEX_TABLES, index_ref, writing_key

if TYPE_CHECKING:
    from everos.memory.deps import Stores
    from everos.memory.models import Profile
    from everos.memory.ports import ForgetEntry, IndexEraser, Scope

    from ._forget_undo import Undoer

_WRITTEN = ("episode", "scene", "group_profile")
"""The kinds a forget writes new versions of."""

_LISTS = ("clusters", "files", "group_queued", "members_removed")
"""The record's lists a reversal reads as they are, the change in flight added."""


class Rollback:
    """Reverses a run of a forget that did not complete.

    Args:
        stores: Markdown and system state.
        eraser: Erases the index rows of the versions taken out; hands the
            files changed back to the cascade.
        undoer: Puts back the rest, as an undo does.
    """

    def __init__(self, *, stores: Stores, eraser: IndexEraser, undoer: Undoer) -> None:
        self._stores = stores
        self._eraser = eraser
        self._undoer = undoer

    async def run(self, scope: Scope, entry: ForgetEntry) -> None:
        """Reverse what ``entry`` records the run may have changed and stamp
        it ``rolled_back``; the caller holds the owner's forget lock."""
        undo = dict(entry.undo or {})
        pending: Mapping[str, Any] = undo.pop("pending", None) or {}
        written = _merged(undo.get("written"), pending.get("written"))
        found = await self._found(scope, pending.get("writing"))
        if found is not None:
            written.setdefault(found[0], []).append(found[1])
        wrote = await self._withdraw(scope, written)
        marked = _merged(undo.get("marked"), pending.get("marked"))
        record: dict[str, Any] = {
            **undo,
            # A mark that named a version taken out names nothing now.
            "marked": {
                kind: [
                    [r[0], r[1], None if r[2] in wrote else r[2], *r[3:]] for r in rows
                ]
                for kind, rows in marked.items()
            },
            "written": {},
            "profiles": await self._taken(
                scope, [*(undo.get("profiles") or []), *(pending.get("profiles") or [])]
            ),
            # Only what this run changed comes back: not what it covered, nor
            # what other forgets took from its segments.
            "segments": [],
        }
        record.pop("covered", None)
        for key in _LISTS:
            record[key] = [*(undo.get(key) or []), *(pending.get(key) or [])]
        await self._undoer.undo(
            scope, dataclasses.replace(entry, undo=record), mode="roll_back"
        )

    async def _found(
        self, scope: Scope, writing: Sequence[Any] | None
    ) -> tuple[str, list[str]] | None:
        """The version the run was writing when it stopped, when the files
        hold it: ``(kind, [old md_path, old id, new md_path, new id])``."""
        if not writing:
            return None
        kind, old_md, old_id, key = (str(value) for value in writing)
        if kind == "episode":
            old = await self._stores.episode.read_many(scope, [old_id])
            if not old:
                return None
            candidates: Sequence[Any] = [
                e
                for e in await self._stores.episode.read_for_segments(
                    scope, [old[0].item.parent_id]
                )
                if writing_key(kind, e.item) == key
            ]
        elif kind == "scene":
            candidates = [
                v
                for v in await self._stores.scene.versions(scope)
                if writing_key(kind, v.item) == key
            ]
        else:  # a group-profile item names the version it supersedes
            candidates = [
                i
                for i in await self._stores.group_profile.items(scope)
                if i.item.supersedes == key
            ]
        new = [c for c in candidates if c.ref.entry_id != old_id]
        if not new:
            return None
        return kind, [old_md, old_id, new[-1].ref.md_path, new[-1].ref.entry_id]

    async def _withdraw(
        self, scope: Scope, written: Mapping[str, Sequence[Sequence[Any]]]
    ) -> set[str]:
        """Take the versions the run wrote out of markdown; their ids."""
        wrote: set[str] = set()
        for kind in _WRITTEN:
            rows = written.get(kind) or []
            ids = [str(row[3]) for row in rows if row[3]]
            if not ids:
                continue
            wrote.update(ids)
            store = {
                "episode": self._stores.episode,
                "scene": self._stores.scene,
                "group_profile": self._stores.group_profile,
            }[kind]
            withdrawal = await store.withdraw(scope, ids)
            if withdrawal.entries:
                await self._eraser.erase(
                    scope,
                    {
                        INDEX_TABLES[kind]: [
                            index_ref(scope, e.md_path, e.entry_id)
                            for e in withdrawal.entries
                        ]
                    },
                )
            paths = {str(row[0]) for row in rows}
            paths |= {e.md_path for e in withdrawal.entries}
            paths |= {md_path for md_path, _, _ in withdrawal.released}
            await self._eraser.requeue(kind, paths)
        return wrote

    async def _taken(
        self, scope: Scope, changes: Sequence[Mapping[str, Any]]
    ) -> list[Mapping[str, Any]]:
        """The profile changes their documents took; one a document never
        took (it still holds what the change took out) has nothing to undo."""
        kept: list[Mapping[str, Any]] = []
        for change in changes:
            person = str(change["person"])
            held = (
                await self._stores.profile.read(scope, person)
                if scope.owner_type == "group"
                else await self._stores.profile.read(scope)
            )
            if held is not None and _took(change, held):
                kept.append(change)
        return kept


def _merged(*parts: Mapping[str, Sequence[Any]] | None) -> dict[str, list[Any]]:
    """Per kind, the rows of every part in order."""
    merged: dict[str, list[Any]] = {}
    for part in parts:
        for kind, rows in (part or {}).items():
            merged.setdefault(kind, []).extend(list(row) for row in rows)
    return merged


def _took(change: Mapping[str, Any], held: Profile) -> bool:
    """Whether ``held`` shows ``change`` written (a document is written whole):
    it holds none of the values the change took out, or -- for a change that
    only queued segments -- it holds them queued."""
    items = (*held.explicit_info, *held.implicit_traits)
    ids = {(i.model_extra or {}).get("item_id") for i in items} - {None}
    described = {
        profile_items.normalised(i.description) for i in items if i.description
    }
    took = False
    for _bucket, _n, item in change.get("items") or []:
        description = profile_items.normalised(str(item.get("description") or ""))
        if item.get("item_id") in ids or description in described:
            return False
        took = True
    for _n, entry in change.get("history") or []:
        if entry in held.history:
            return False
        took = True
    summary = change.get("summary")
    if summary:
        if held.summary == summary:
            return False
        took = True
    if took:
        return True
    queued = [str(m) for m in change.get("queued") or []]
    return bool(queued) and set(queued) <= set(held.pending_memcell_ids)
