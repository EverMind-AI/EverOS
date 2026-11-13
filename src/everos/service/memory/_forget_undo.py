"""Undoing one forget by its id: a status change, and copies where it wrote.

What an undo puts back, read from the forget log (:mod:`._forget_log`):

1. **Its marks.** Every entry the forget marked ``forgotten`` gets back the
   mark it carried before (none, for a current one) -- unless another live
   forget marked it too: an entry stays forgotten while any forget naming it
   stands.
2. **What it wrote.** A version the forget wrote in place of another (a text
   edit without a picture, a re-extracted episode, a group's scene narrated
   again) is superseded by a copy of the version it replaced, written as the
   newest: the logs only ever grow, and a scene's version number keeps
   rising.
3. **Profiles.** Values a forget took out of a profile document come back in
   their places, by id; a row written before the log kept places puts them
   back by description. A user's undo also brings back what another live
   forget took by the segment rule resting on a segment this one remembers
   again. Stale segments leave the queues; a member's removed document comes
   back whole.
4. **Messages, files, clusters.** The messages' forget marks are stamped
   undone, picture files come back into circulation, scene clusters are put
   back as they were.

**Order.** An undo is refused (409) while a version the forget wrote is no
longer current -- a later forget, undo or edit changed it: undo that one first.
Forgets of different records undo in any order.

The log entry is stamped last, so an undo cut short can be sent again; a
second undo of the same forget restores nothing. Rows logged before this
format undo too: a user's are read into it, a group's (whose forget took
records out of markdown) are reversed by putting them back
(:func:`reverse_legacy_group`).
"""

from __future__ import annotations

__all__ = ["Undoer"]

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from everos.core.errors import ConflictError
from everos.core.observability.logging import get_logger
from everos.memory import group_profile_lock, profile_lock, scenes_lock
from everos.memory.extract.pipeline import profile_items
from everos.memory.extract.scene import restore_revived_scenes, scene_cluster_kind
from everos.memory.models import FORGOTTEN_MARK, Cluster, Profile
from everos.memory.ports import (
    NO_SOURCE_CHECK,
    MemoryRef,
    Scope,
    Stored,
    Withdrawal,
    WithdrawnEntry,
)

from ._forget_apply import member_from_record
from ._forget_log import (
    Logs,
    Named,
    Removed,
    is_current_format,
    is_legacy_group,
    removed_in,
    segments_of,
)
from ._forget_models import ForgetTarget, ForgottenPicture, UndoResult
from ._records import memory_id

if TYPE_CHECKING:
    from everos.memory.deps import Stores
    from everos.memory.ports import ForgetEntry, IndexEraser

logger = get_logger("everos.service.memory")

_KINDS = ("episode", "image_memory", "scene", "group_profile")


@dataclass
class _Record:
    """One forget as its undo reads it, whatever format logged it.

    Attributes:
        marked: Per kind, ``(md_path, entry_id, previous mark)`` it marked.
        written: Per kind, ``(old md_path, old id, new md_path, new id)``.
        revive: A user's scenes come back with their episodes (any scene whose
            newest version is forgotten while an episode of it is alive).
        clusters: Clusters as they were; ``whole`` ones were removed.
        files: Picture files to bring back.
        files_of_pictures: The files of the picture memories restored come
            back (a conversation's forget logged before files were named).
        segments: A user's segments remembered again.
        removed: Profile values to put back.
        others: Bring back what other live forgets took by the segment rule.
        members_removed: A group's member documents to write back.
        group_queued: Segments to take off the group profile's queue.
        queued: Per member, segments to take off the member's queue.
    """

    marked: dict[str, list[tuple[str, str, str | None]]] = field(default_factory=dict)
    written: dict[str, list[tuple[str, str, str, str]]] = field(default_factory=dict)
    revive: bool = False
    clusters: list[dict[str, Any]] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    files_of_pictures: bool = False
    segments: list[str] = field(default_factory=list)
    removed: list[Removed] = field(default_factory=list)
    others: bool = False
    members_removed: list[dict[str, Any]] = field(default_factory=list)
    group_queued: list[str] = field(default_factory=list)
    queued: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class _Done:
    """What an undo put back so far, for its answer and the cascade."""

    episodes: list[str] = field(default_factory=list)
    image_memories: list[Stored[Any]] = field(default_factory=list)
    scenes: list[str] = field(default_factory=list)
    group_profile: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    profiles: list[str] = field(default_factory=list)
    items: list[str] = field(default_factory=list)
    item_count: int = 0
    retired: list[str] = field(default_factory=list)
    paths: dict[str, set[str]] = field(default_factory=dict)


class Undoer:
    """Reverses one logged forget.

    Args:
        stores: Markdown and system state.
        eraser: Hands what an undo changed back to the cascade.
    """

    def __init__(self, *, stores: Stores, eraser: IndexEraser) -> None:
        self._stores = stores
        self._eraser = eraser
        self._logs = Logs(stores)

    async def undo(self, scope: Scope, entry: ForgetEntry) -> UndoResult:
        """Put back what ``entry`` forgot; the caller holds the owner's forget lock.

        Raises:
            ConflictError: A version the forget wrote is no longer current.
        """
        target = target_of(entry)
        if entry.undone:
            return UndoResult(forget_id=entry.forget_id, applied=False, target=target)
        if is_legacy_group(entry):
            return await reverse_legacy_group(
                self._stores, self._eraser, scope, entry, target=target
            )
        record = await self._record(scope, entry)
        await self._check_order(scope, record)
        keep = await self._logs.named_by_others(scope, except_id=entry.forget_id)
        done = _Done()
        await self._rewrite_back(scope, record, done)
        await self._unmark(scope, record, keep, done, forget_id=entry.forget_id)
        if record.revive and done.episodes:
            revived = await restore_revived_scenes(
                self._stores.episode, self._stores.scene, scope, keep=keep.scenes
            )
            done.scenes += [s.ref.entry_id for s in revived]
            done.paths.setdefault("scene", set()).update(s.ref.md_path for s in revived)
        await self._restore_clusters(scope, record.clusters)
        files = list(record.files)
        if record.files_of_pictures:
            files += [
                row.item.image_id for row in done.image_memories if row.item.image_id
            ]
        for image_id in dict.fromkeys(files):
            await self._stores.images.restore(image_id)
            done.files.append(image_id)
        for kind, paths in done.paths.items():
            if paths:
                await self._eraser.requeue(kind, paths)
        await self._group_queue(scope, record.group_queued)
        if scope.owner_type == "group":
            await self._restore_members(scope, record, done)
            await self._stores.message_forget.undo(entry.forget_id)
            await self._stores.forget_log.mark_undone(entry.forget_id)
        else:
            async with profile_lock(scope):
                await self._restore_profile(scope, entry, record, done)
                await self._stores.message_forget.undo(entry.forget_id)
                await self._stores.forget_log.mark_undone(entry.forget_id)
        logger.info(
            "memory.undo_completed",
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_type=scope.owner_type,
            owner_id=scope.owner_id,
            forget_id=entry.forget_id,
            episodes=len(done.episodes),
            image_memories=len(done.image_memories),
            scenes=len(done.scenes),
            profile_items=done.item_count,
        )
        return UndoResult(
            forget_id=entry.forget_id,
            applied=True,
            target=target,
            messages=tuple(
                (str(session_id), str(message_id))
                for session_id, ids in (entry.undo or {}).get("messages") or []
                for message_id in ids
            ),
            episodes=tuple(memory_id(scope, e) for e in dict.fromkeys(done.episodes)),
            image_memories=tuple(
                memory_id(scope, row.ref.entry_id) for row in done.image_memories
            ),
            pictures=tuple(
                ForgottenPicture(
                    entry_id=row.ref.entry_id,
                    image_id=row.item.image_id,
                    name=row.item.name,
                )
                for row in done.image_memories
            ),
            images=tuple(done.files),
            scenes=tuple(memory_id(scope, s) for s in dict.fromkeys(done.scenes)),
            profiles=tuple(done.profiles),
            profile_items=tuple(done.items),
            profile_item_count=done.item_count,
            group_profile_items=tuple(memory_id(scope, e) for e in done.group_profile),
            retired=tuple(memory_id(scope, e) for e in done.retired),
        )

    # ── the record, whatever format logged it ────────────────────────────

    async def _record(self, scope: Scope, entry: ForgetEntry) -> _Record:
        undo = entry.undo or {}
        if is_current_format(entry):
            marked = undo.get("marked") or {}
            return _Record(
                marked={
                    kind: [(str(r[0]), str(r[1]), r[2]) for r in marked.get(kind) or []]
                    for kind in _KINDS
                },
                written={
                    kind: [
                        (str(r[0]), str(r[1]), str(r[2]), str(r[3]))
                        for r in (undo.get("written") or {}).get(kind) or []
                    ]
                    for kind in ("episode", "scene")
                },
                revive=scope.owner_type == "user",
                clusters=list(undo.get("clusters") or []),
                files=[str(i) for i in undo.get("files") or []],
                segments=[str(s) for s in undo.get("segments") or []],
                removed=removed_in(entry),
                others=scope.owner_type == "user",
                members_removed=list(undo.get("members_removed") or []),
                group_queued=[str(m) for m in undo.get("group_queued") or []],
                queued={
                    str(change["person"]): [str(m) for m in change.get("queued") or []]
                    for change in undo.get("profiles") or []
                    if change.get("queued")
                },
            )
        if undo:  # a user's forget by records, everything or a condition
            return _Record(
                marked={
                    "episode": [
                        (str(md), str(e), None) for md, e in undo.get("episodes") or []
                    ],
                    "image_memory": [
                        (str(r[0]), str(r[1]), None)
                        for r in undo.get("image_memories") or []
                    ],
                    "scene": [
                        (str(md), str(e), None) for md, e in undo.get("scenes") or []
                    ],
                },
                revive=True,
                clusters=[
                    {"cluster": c, "whole": True} for c in undo.get("clusters") or []
                ],
                files=[str(i) for i in undo.get("files") or []],
                segments=[str(s) for s in undo.get("segments") or []],
                removed=removed_in(entry),
                others=True,
            )
        if entry.target == "image":
            return await self._legacy_picture(scope, entry)
        if entry.target == "session":
            return await self._legacy_session(scope, entry)
        return _Record(removed=removed_in(entry))

    async def _legacy_picture(self, scope: Scope, entry: ForgetEntry) -> _Record:
        """A picture's forget logged in the columns: its memories, its rewrites."""
        stores = self._stores
        pictures = await stores.image_memory.read_many(scope, list(entry.entry_ids))
        originals = await stores.episode.read_many(
            scope, [old for old, _ in entry.episode_ids]
        )
        replacements = await stores.episode.read_many(
            scope, [new for _, new in entry.episode_ids if new]
        )
        paths = {row.ref.entry_id: row.ref.md_path for row in originals}
        paths |= {row.ref.entry_id: row.ref.md_path for row in replacements}
        return _Record(
            marked={
                "image_memory": [
                    (row.ref.md_path, row.ref.entry_id, None) for row in pictures
                ]
            },
            written={
                "episode": [
                    (paths[old], old, paths.get(new, ""), new)
                    for old, new in entry.episode_ids
                    if old in paths and new
                ]
            },
            files=[entry.target_id],
            removed=removed_in(entry),
        )

    async def _legacy_session(self, scope: Scope, entry: ForgetEntry) -> _Record:
        """A conversation's forget logged in the columns: what it marked.

        A row that named none of its entries covered every one of them.
        """
        stores, session_id = self._stores, entry.target_id
        named = {old for old, _ in entry.episode_ids}
        episodes = [
            (md, e, None)
            for md, e in await stores.episode.entries_for_session(scope, session_id)
            if not named or e in named
        ]
        pictures = set(entry.entry_ids)
        images = [
            (md, e, None)
            for md, e, _ in await stores.image_memory.entries_for_session(
                scope, session_id
            )
            if not pictures or e in pictures
        ]
        return _Record(
            marked={"episode": episodes, "image_memory": images},
            revive=scope.owner_type == "user",
            files_of_pictures=True,
            segments=sorted(await segments_of(stores, scope, [session_id])),
            removed=removed_in(entry),
            others=True,
        )

    # ── the steps ────────────────────────────────────────────────────────

    async def _check_order(self, scope: Scope, record: _Record) -> None:
        """Refuse while a later change replaced what this forget wrote."""
        for kind, rows in record.written.items():
            for _old_md, _old, new_md, new in rows:
                if not new_md:
                    continue
                if (await self._store(kind).deprecations(scope, new_md)).get(new):
                    raise ConflictError(
                        "a later forget or edit changed what this forget wrote; "
                        "undo that one first"
                    )

    async def _rewrite_back(self, scope: Scope, record: _Record, done: _Done) -> None:
        """Each version the forget replaced is written again, superseding its
        replacement."""
        episodes = record.written.get("episode") or []
        if episodes:
            found = {
                row.ref.entry_id: row
                for row in await self._stores.episode.read_many(
                    scope, [old for _, old, _, _ in episodes]
                )
            }
            pairs = [
                (found[old], new_md, new)
                for _, old, new_md, new in episodes
                if old in found
            ]
            refs = await self._stores.episode.append_many(
                scope, [row.item for row, _, _ in pairs], only_if=NO_SOURCE_CHECK
            )
            for (_row, new_md, new), ref in zip(pairs, refs, strict=True):
                paths = done.paths.setdefault("episode", set())
                if new_md:
                    await self._stores.episode.deprecate(
                        scope,
                        entries=[(new_md, new)],
                        replaced_by=ref.entry_id,
                        only_if=NO_SOURCE_CHECK,
                    )
                    done.retired.append(new)
                    paths.add(new_md)
                done.episodes.append(ref.entry_id)
                paths.add(ref.md_path)
        scenes = record.written.get("scene") or []
        if scenes:
            async with scenes_lock(scope):
                versions = {
                    v.ref.entry_id: v for v in await self._stores.scene.versions(scope)
                }
                for _old_md, old, _new_md, new in scenes:
                    if old not in versions or new not in versions:
                        continue
                    replacement = versions[new]
                    copy = versions[old].item.model_copy(
                        update={"version": replacement.item.version + 1}
                    )
                    ref = await self._stores.scene.append(
                        scope, copy, supersedes=replacement, only_if=NO_SOURCE_CHECK
                    )
                    if ref is not None:
                        done.scenes.append(ref.entry_id)
                        done.retired.append(new)
                        done.paths.setdefault("scene", set()).update(
                            {ref.md_path, replacement.ref.md_path}
                        )

    async def _unmark(
        self,
        scope: Scope,
        record: _Record,
        keep: Named,
        done: _Done,
        *,
        forget_id: str,
    ) -> None:
        """Each mark this forget wrote that still stands goes back to what it was."""
        for kind in _KINDS:
            rows = [
                row
                for row in record.marked.get(kind) or []
                if row[1] not in keep.of(kind)
            ]
            if not rows:
                continue
            store = self._store(kind)
            lock = (
                scenes_lock(scope)
                if kind == "scene"
                else group_profile_lock(scope.app_id, scope.project_id, scope.owner_id)
                if kind == "group_profile"
                else None
            )
            if lock is None:
                standing = await self._standing(store, scope, rows, forget_id)
                await self._put_marks(store, scope, standing)
            else:
                async with lock:
                    standing = await self._standing(store, scope, rows, forget_id)
                    await self._put_marks(store, scope, standing)
            ids = [e for _, e, _ in standing]
            if kind == "episode":
                done.episodes += ids
            elif kind == "scene":
                done.scenes += ids
            elif kind == "group_profile":
                done.group_profile += ids
            else:
                done.image_memories += await self._stores.image_memory.read_many(
                    scope, ids
                )
            done.paths.setdefault(kind, set()).update(md for md, _, _ in standing)

    async def _standing(
        self,
        store: Any,
        scope: Scope,
        rows: Sequence[tuple[str, str, str | None]],
        forget_id: str,
    ) -> list[tuple[str, str, str | None]]:
        """The rows whose forget mark still stands (an early picture forget
        marked with its own id)."""
        marks: dict[str, dict[str, str]] = {}
        for md_path in dict.fromkeys(md for md, _, _ in rows):
            marks[md_path] = await store.deprecations(scope, md_path)
        return [
            row
            for row in rows
            if marks[row[0]].get(row[1]) in (FORGOTTEN_MARK, forget_id)
        ]

    async def _put_marks(
        self, store: Any, scope: Scope, rows: Sequence[tuple[str, str, str | None]]
    ) -> None:
        cleared = [e for _, e, prev in rows if not prev]
        if cleared:
            await store.clear_deprecations(scope, cleared)
        for md_path, entry_id, prev in rows:
            if prev:
                await store.deprecate(
                    scope,
                    entries=[(md_path, entry_id)],
                    replaced_by=prev,
                    only_if=NO_SOURCE_CHECK,
                )

    async def _restore_clusters(
        self, scope: Scope, changes: Sequence[Mapping[str, Any]]
    ) -> None:
        if not changes:
            return
        async with scenes_lock(scope):
            for change in reversed(changes):
                cluster = Cluster.model_validate(change["cluster"])
                if cluster.id is None:
                    continue
                await self._stores.cluster.delete(cluster.id)
                await self._stores.cluster.upsert(
                    scope, scene_cluster_kind(scope), cluster, member_type="episode"
                )

    async def _group_queue(self, scope: Scope, queued: Collection[str]) -> None:
        if not queued:
            return
        async with group_profile_lock(scope.app_id, scope.project_id, scope.owner_id):
            pending = await self._stores.group_profile.pending(scope)
            left = [m for m in pending if m not in set(queued)]
            if left != pending:
                await self._stores.group_profile.set_pending(scope, left)

    async def _restore_members(
        self, scope: Scope, record: _Record, done: _Done
    ) -> None:
        """A group's member documents and items, each under the member's lock."""
        for removed in record.members_removed:
            member_id = str(removed["member_id"])
            async with profile_lock(scope, member_id):
                if await self._stores.profile.read(scope, member_id) is None:
                    await self._stores.profile.write(
                        scope, member_from_record(removed["profile"])
                    )
                    done.profiles.append(member_id)
        by_person: dict[str, list[Removed]] = {}
        for value in record.removed:
            by_person.setdefault(value.person_id, []).append(value)
        for person in record.queued:
            by_person.setdefault(person, [])
        for person, values in by_person.items():
            async with profile_lock(scope, person):
                held = await self._stores.profile.read(scope, person)
                if held is None:
                    continue
                restored, ids, count = _put_back(held, values)
                restored = _unqueue(restored or held, record.queued.get(person, ()))
                if restored is held:
                    continue
                await self._stores.profile.write(scope, restored)
            done.items += ids
            done.item_count += count

    async def _restore_profile(
        self, scope: Scope, entry: ForgetEntry, record: _Record, done: _Done
    ) -> None:
        """A user's values back, and what other forgets took that rests on a
        segment remembered again; the caller holds the profile lock."""
        values = list(record.removed)
        if record.others and record.segments:
            values += await self._logs.removed_by_others(
                scope, record.segments, except_id=entry.forget_id
            )
        queued = record.queued.get(scope.owner_id, [])
        if not values and not queued:
            return
        held = await self._stores.profile.read(scope)
        restored, ids, count = _put_back(held, values, owner_id=scope.owner_id)
        if held is not None and queued:
            # The segments a message forget queued for the next update leave
            # the queue: the items it took out are back.
            base = restored or held
            unqueued = _unqueue(base, queued)
            if unqueued.pending_memcell_ids != base.pending_memcell_ids:
                restored = unqueued
        if restored is None:
            return
        await self._stores.profile.write(scope, restored)
        done.items += ids
        done.item_count += count

    def _store(self, kind: str) -> Any:
        return {
            "episode": self._stores.episode,
            "image_memory": self._stores.image_memory,
            "scene": self._stores.scene,
            "group_profile": self._stores.group_profile,
        }[kind]


def target_of(entry: ForgetEntry) -> ForgetTarget | None:
    """The target a forget addressed, as its row records it."""
    data = (entry.undo or {}).get("target")
    if isinstance(data, Mapping):
        return ForgetTarget.from_json(data)
    if entry.target == "image":
        return ForgetTarget(kind="image", image_id=entry.target_id)
    if entry.target == "session":
        return ForgetTarget(kind="session", session_id=entry.target_id)
    return None


def _put_back(
    held: Profile | None, values: Sequence[Removed], *, owner_id: str | None = None
) -> tuple[Profile | None, list[str], int]:
    """``held`` with ``values`` back: by place where the log kept it, else by
    description (a row written before places were kept). Returns the profile
    (``None`` when there is nothing to write), the ids back, and how many
    items (not replaced values) came back."""
    placed = [
        v for v in values if v.position is not None or v.bucket == profile_items.SUMMARY
    ]
    loose = [v for v in values if v not in placed]
    ids: list[str] = []
    count = 0
    profile = held
    if placed and profile is not None:
        summary = next(
            (
                str(v.item.get("summary") or "")
                for v in placed
                if v.bucket == profile_items.SUMMARY
            ),
            None,
        )
        profile, back = profile_items.put_back(
            profile,
            profile_items.Taken(
                items=tuple(
                    (v.bucket, int(v.position or 0), dict(v.item))
                    for v in placed
                    if v.bucket in profile_items.BUCKETS
                ),
                history=tuple(
                    (int(v.position or 0), dict(v.item))
                    for v in placed
                    if v.bucket == "history"
                ),
                summary=summary,
            ),
        )
        ids += [i for i in back if i]
        count += len(back)
    if loose:
        base = profile if profile is not None else held
        restored, items, replaced = profile_items.restore_items(
            base,
            [(v.bucket, v.item) for v in loose],
            owner_id=owner_id or (held.owner_id if held is not None else ""),
            timestamp=_newest_observed(loose),
        )
        if items or replaced:
            profile = restored
            count += items
            ids += [
                str(v.item["item_id"])
                for v in loose
                if v.bucket in profile_items.BUCKETS
                and isinstance(v.item.get("item_id"), str)
            ][:items]
    if profile is held:
        return None, [], 0
    return profile, ids, count


def _unqueue(profile: Profile, queued: Collection[str]) -> Profile:
    """``profile`` without the segments a forget queued on it."""
    if not queued:
        return profile
    wanted = set(queued)
    return profile.model_copy(
        update={
            "pending_memcell_ids": [
                m for m in profile.pending_memcell_ids if m not in wanted
            ]
        }
    )


def _newest_observed(values: Sequence[Removed]) -> int:
    newest = 0
    for value in values:
        at = value.item.get("observed_at")
        if isinstance(at, int):
            newest = max(newest, at)
    return newest


async def reverse_legacy_group(
    stores: Stores,
    eraser: IndexEraser,
    scope: Scope,
    entry: ForgetEntry,
    *,
    target: ForgetTarget | None,
) -> UndoResult:
    """Undo a group forget logged when forgets took records out of markdown.

    As such a forget was always undone: what it wrote is taken out, what it
    took out is put back under its own ids, member documents and items come
    back, queued segments leave the queues, the messages' marks are stamped
    undone. Refused (409) while a later forget took out what it wrote.
    """
    undo = entry.undo or {}
    written = undo.get("written") or {}
    episodes_written = [str(e) for e in written.get("episode") or []]
    scenes_written = [str(e) for e in written.get("scene") or []]
    present = {
        s.ref.entry_id for s in await stores.episode.read_many(scope, episodes_written)
    }
    versions = {v.ref.entry_id for v in await stores.scene.versions(scope)}
    if set(episodes_written) - present or set(scenes_written) - versions:
        raise ConflictError(
            "a later forget changed what this one wrote; undo that one first"
        )
    retired: list[str] = []
    for store, ids in (
        (stores.episode, episodes_written),
        (stores.scene, scenes_written),
    ):
        taken = await store.withdraw(scope, ids)
        if taken.entries:
            await eraser.erase(
                scope, {"episode": [_withdrawn_ref(scope, e) for e in taken.entries]}
            )
            retired += [e.entry_id for e in taken.entries]
    stash = undo.get("withdrawn") or {}
    restored: dict[str, list[MemoryRef]] = {}
    kinds = {
        "episode": stores.episode,
        "image_memory": stores.image_memory,
        "scene": stores.scene,
        "group_profile": stores.group_profile,
    }
    for kind, store in kinds.items():
        if kind not in stash:
            continue
        withdrawal = _withdrawal_from(stash[kind])
        restored[kind] = await store.reinstate(scope, withdrawal)
        paths = {e.md_path for e in withdrawal.entries} | {
            md_path for md_path, _, _ in withdrawal.released
        }
        if paths:
            await eraser.requeue(kind, paths)
    members_back: list[str] = []
    for removed in undo.get("members_removed") or []:
        member_id = str(removed["member_id"])
        async with profile_lock(scope, member_id):
            if await stores.profile.read(scope, member_id) is None:
                await stores.profile.write(
                    scope, member_from_record(removed["profile"])
                )
                members_back.append(member_id)
    items_back: list[str] = []
    for change in undo.get("member_items") or []:
        member_id = str(change["member_id"])
        async with profile_lock(scope, member_id):
            held = await stores.profile.read(scope, member_id)
            if held is None:
                continue
            back, ids = profile_items.put_back(
                held,
                profile_items.Taken(
                    items=tuple(
                        (bucket, int(n), dict(item))
                        for bucket in profile_items.BUCKETS
                        for n, item in change.get(bucket) or []
                    ),
                    history=tuple(
                        (int(n), dict(h)) for n, h in change.get("history") or []
                    ),
                    summary=change.get("summary"),
                ),
            )
            await stores.profile.write(
                scope, _unqueue(back, [str(m) for m in change.get("queued") or []])
            )
        items_back += ids
    queued = {str(m) for m in undo.get("group_queued") or []}
    if queued:
        async with group_profile_lock(scope.app_id, scope.project_id, scope.owner_id):
            pending = await stores.group_profile.pending(scope)
            left = [m for m in pending if m not in queued]
            if left != pending:
                await stores.group_profile.set_pending(scope, left)
    changes = undo.get("clusters") or []
    if changes:
        async with scenes_lock(scope):
            for change in reversed(changes):
                cluster = Cluster.model_validate(change["cluster"])
                if cluster.id is None:
                    continue
                await stores.cluster.delete(cluster.id)
                await stores.cluster.upsert(
                    scope, scene_cluster_kind(scope), cluster, member_type="episode"
                )
    for image_id in undo.get("files") or []:
        await stores.images.restore(image_id)
    await stores.message_forget.undo(entry.forget_id)
    await stores.forget_log.mark_undone(entry.forget_id)
    return UndoResult(
        forget_id=entry.forget_id,
        applied=True,
        target=target,
        messages=tuple(
            (str(session_id), str(message_id))
            for session_id, ids in undo.get("messages") or []
            for message_id in ids
        ),
        episodes=_rows(scope, restored.get("episode", [])),
        image_memories=_rows(scope, restored.get("image_memory", [])),
        images=tuple(str(i) for i in undo.get("files") or []),
        scenes=_rows(scope, restored.get("scene", [])),
        profiles=tuple(members_back),
        profile_items=tuple(items_back),
        profile_item_count=len(items_back),
        group_profile_items=_rows(scope, restored.get("group_profile", [])),
        retired=tuple(f"{scope.owner_id}_{e}" for e in retired),
    )


def _rows(scope: Scope, refs: Sequence[MemoryRef]) -> tuple[str, ...]:
    return tuple(f"{scope.owner_id}_{ref.entry_id}" for ref in refs)


def _withdrawn_ref(scope: Scope, entry: WithdrawnEntry) -> MemoryRef:
    return MemoryRef(
        id=f"{scope.owner_id}_{entry.entry_id}",
        entry_id=entry.entry_id,
        md_path=entry.md_path,
    )


def _withdrawal_from(data: Mapping[str, Any]) -> Withdrawal:
    return Withdrawal(
        entries=tuple(
            WithdrawnEntry(
                entry_id=str(e["entry_id"]),
                md_path=str(e["md_path"]),
                text=str(e["text"]),
                mark=e.get("mark"),
            )
            for e in data.get("entries") or []
        ),
        released=tuple(
            (str(r[0]), str(r[1]), str(r[2])) for r in data.get("released") or []
        ),
    )
