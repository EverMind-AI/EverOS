"""Carrying out a forget's plan: one write path for every owner type.

docs/api.md and docs/group-memory.md, section 9.4. Everything the run changes
is appended or marked -- nothing leaves markdown -- and recorded as it goes
(:class:`Applied`), which is both the forget log's ``undo_json`` and, when a
step fails, what the run's own reversal reads:

1. the model calls first, before anything is written: each segment that
   keeps text after a message forget is extracted again (the space's own
   episode call, :class:`Reextraction`), and each group scene that changed is
   narrated again over what it keeps;
2. every version written in place of another supersedes it (a re-extracted
   episode, a text edit without a picture, a group's scene narrated again);
3. what goes without a successor is marked ``forgotten``, the mark each entry
   carried before kept for the undo, and its index row erased at once (the
   cascade keeps the mark on the row it projects again);
4. profile documents lose what their sources no longer support, each value
   kept with its place; a group's stale items are queued to be written again;
5. picture files no live memory refers to any more leave circulation.

A text edit needs no model: a picture's contribution to an episode or a scene
is a reference, not prose -- the picture prompt forbids restating a picture's
description in the body (the picture has its own memory for that) -- so the
reference is cut out, and a model call would risk rewriting what nobody asked
to change.
"""

from __future__ import annotations

__all__ = [
    "Applied",
    "ForgetWriter",
    "Reextraction",
    "applied_record",
    "strip_reference",
]

import re
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from everos.core.errors import ConflictError
from everos.memory import group_profile_lock, profile_lock, scenes_lock
from everos.memory.extract.pipeline import profile_items
from everos.memory.extract.pipeline.pictures import memcell_alive
from everos.memory.extract.scene import (
    alive_members,
    current_versions,
    forget_orphaned_scenes,
    scene_cluster_kind,
)
from everos.memory.models import FORGOTTEN_MARK, Episode, Profile, Scene
from everos.memory.ports import NO_SOURCE_CHECK, MemoryRef, Scope, Stored

from ._forget_log import FORMAT, Logs
from ._forget_plan import ForgetPlan, ProfileEdit, member_edit, user_edit

if TYPE_CHECKING:
    from everos.memory.deps import Stores
    from everos.memory.extract.scene import SceneNarrator
    from everos.memory.models import MemCell
    from everos.memory.ports import IndexEraser

_EPISODE_TABLE = "episode"
"""Episodes and scene versions share this index table."""

_TABLES: Mapping[str, str] = {
    "episode": _EPISODE_TABLE,
    "scene": _EPISODE_TABLE,
    "image_memory": "image_memory",
    "group_profile": "group_profile_item",
}
"""The index table each daily-log kind is projected into."""


@runtime_checkable
class Reextraction(Protocol):
    """What a forget needs of the conversation pipeline: a segment's episode call."""

    async def reextract(
        self,
        *,
        scope: Scope,
        session_id: str,
        memcell_id: str,
        cell: MemCell,
        senders: Sequence[str] = (),
    ) -> Episode | None: ...


@dataclass(frozen=True)
class Prepared:
    """The model's answers for one run, made before anything is written.

    Attributes:
        episodes: Per rewrite, in the plan's order, the re-extracted episode;
            ``None`` for a text edit.
        scenes: Per re-narrated scene, its new version; a rewritten member is
            named by its old entry id until the run knows the new one.
    """

    episodes: tuple[Episode | None, ...] = ()
    scenes: Mapping[str, Scene] = field(default_factory=dict)


@dataclass
class Applied:
    """What a run changed so far; it becomes the forget log's ``undo_json``.

    Attributes:
        marked: Per kind, ``[md_path, entry_id, previous mark]`` of every
            entry the run marked forgotten (a picture memory adds its hash).
        orphaned: ``[md_path, entry_id]`` of a user's scenes marked because
            none of their episodes is alive.
        written: Per kind, ``[old md_path, old entry_id, new md_path, new
            entry_id]`` of every version written in place of another.
        clusters: Scene clusters as they were before the run changed them.
        files: Picture files taken out of circulation.
        profiles: Per profile document, what went with each value's place.
        members_removed: Member documents removed whole.
        group_queued: Segments queued on the group profile's next update.
    """

    marked: dict[str, list[list[Any]]] = field(default_factory=dict)
    orphaned: list[list[str]] = field(default_factory=list)
    written: dict[str, list[list[str]]] = field(default_factory=dict)
    clusters: list[dict[str, Any]] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    profiles: list[dict[str, Any]] = field(default_factory=list)
    members_removed: list[dict[str, Any]] = field(default_factory=list)
    group_queued: list[str] = field(default_factory=list)


def applied_record(plan: ForgetPlan, applied: Applied) -> dict[str, Any]:
    """The forget log's ``undo_json`` of a run: the plan's facts and its changes."""
    return {
        "format": FORMAT,
        "target": plan.target.to_json(),
        "sources": sorted(plan.sources),
        "messages": [
            [session.session_id, list(ids)] for session, ids in plan.messages.items()
        ],
        "marked": applied.marked,
        "orphaned": applied.orphaned,
        "written": applied.written,
        "retired": {kind: sorted(ids) for kind, ids in plan.retired.items()},
        "clusters": applied.clusters,
        "files": applied.files,
        "profiles": applied.profiles,
        "members_removed": applied.members_removed,
        "group_queued": applied.group_queued,
        "segments": list(plan.segments),
        # Named apart so the picture-bytes check finds a hash by key.
        "pictures": [
            {"image_id": image_id}
            for image_id in dict.fromkeys(m.item.image_id for m in plan.image_memories)
        ],
    }


class ForgetWriter:
    """Writes one plan's changes, recording each in :class:`Applied`.

    Args:
        stores: Markdown and system state.
        eraser: Erases the index rows of what a run takes out of default reads.
        pipeline: Re-extraction; ``None`` refuses a plan that needs it.
        scenes: Narrates a group's changed scene again; ``None`` writes none
            (the scene's current version is marked forgotten instead).
    """

    def __init__(
        self,
        *,
        stores: Stores,
        eraser: IndexEraser,
        pipeline: Reextraction | None,
        scenes: SceneNarrator | None,
    ) -> None:
        self._stores = stores
        self._eraser = eraser
        self._pipeline = pipeline
        self._narrator = scenes
        self._logs = Logs(stores)

    async def prepare(self, plan: ForgetPlan) -> Prepared:
        """Every model call of the run, in order, before anything is written.

        Each segment that keeps text is extracted again; then each group scene
        that changed is narrated afresh over the episodes it keeps, a
        rewritten one read as its new version. Raises ConflictError when a
        segment lost its text meanwhile (the plan checked it has some).
        """
        episodes: list[Episode | None] = []
        for rewrite in plan.rewrites:
            if rewrite.cell is None:
                episodes.append(None)
                continue
            assert self._pipeline is not None
            episode = await self._pipeline.reextract(
                scope=plan.scope,
                session_id=rewrite.session_id,
                memcell_id=rewrite.memcell_id,
                cell=rewrite.cell,
                senders=rewrite.old.item.sender_ids,
            )
            if episode is None:
                raise ConflictError("a segment changed while it was forgotten")
            episodes.append(episode)
        new = {
            r.old.ref.entry_id: e
            for r, e in zip(plan.rewrites, episodes, strict=True)
            if e is not None
        }
        scenes: dict[str, Scene] = {}
        narrator = self._narrator
        for edit in plan.scenes:
            if not edit.members or edit.current is None or narrator is None:
                continue
            alive = await alive_members(
                self._stores.episode, plan.scope, list(edit.members)
            )
            members = sorted(
                (
                    Stored(
                        ref=m.ref, scope=m.scope, item=new.get(m.ref.entry_id, m.item)
                    )
                    for m in alive
                ),
                key=lambda m: m.item.timestamp,
            )
            if len(members) < narrator.min_members:
                continue
            scene, _ = await narrator.version(
                plan.scope,
                scene_id=edit.scene_id,
                version=edit.current.item.version + 1,
                members=members,
                language=edit.current.item.language,
            )
            scenes[edit.scene_id] = scene
        return Prepared(episodes=tuple(episodes), scenes=scenes)

    async def write(
        self, plan: ForgetPlan, prepared: Prepared, applied: Applied
    ) -> None:
        """Every change of ``plan`` but the profiles."""
        scope = plan.scope
        reached = await self._cluster_members(plan)
        replaced = await self._rewrite_episodes(plan, prepared.episodes, applied)
        await self._mark(
            "episode",
            scope,
            [(e.ref.md_path, e.ref.entry_id, None) for e in plan.episodes],
            applied,
        )
        await self._mark(
            "image_memory",
            scope,
            [
                (m.ref.md_path, m.ref.entry_id, None, m.item.image_id)
                for m in plan.image_memories
            ],
            applied,
        )
        await self._scenes(plan, prepared, replaced, reached, applied)
        await self._group_profile(plan, applied)
        for image_id in plan.files:
            await self._stores.images.forget(image_id)
            applied.files.append(image_id)
        await self._erase(scope, applied)

    # ── episodes ─────────────────────────────────────────────────────────

    async def _rewrite_episodes(
        self,
        plan: ForgetPlan,
        episodes: Sequence[Episode | None],
        applied: Applied,
    ) -> dict[str, MemoryRef]:
        """One superseding version per rewrite: re-extracted, or a text edit."""
        scope = plan.scope
        replaced: dict[str, MemoryRef] = {}
        for rewrite, episode in zip(plan.rewrites, episodes, strict=True):
            old = rewrite.old
            if episode is None:
                assert rewrite.image_id is not None
                ref = await self._stores.episode.append(
                    scope, _without_picture(old.item, rewrite.image_id), only_if=_always
                )
            else:
                ref = await self._stores.episode.append(
                    scope,
                    episode,
                    only_if=memcell_alive(self._stores.memcell, rewrite.memcell_id),
                )
            if ref is None:  # the segment is gone: nothing supersedes the old one
                await self._mark(
                    "episode",
                    scope,
                    [(old.ref.md_path, old.ref.entry_id, None)],
                    applied,
                )
                continue
            await self._stores.episode.deprecate(
                scope,
                entries=[(old.ref.md_path, old.ref.entry_id)],
                replaced_by=ref.entry_id,
                only_if=_always,
            )
            applied.written.setdefault("episode", []).append(
                [old.ref.md_path, old.ref.entry_id, ref.md_path, ref.entry_id]
            )
            replaced[old.ref.entry_id] = ref
        return replaced

    async def _mark(
        self,
        kind: str,
        scope: Scope,
        entries: Sequence[tuple[Any, ...]],
        applied: Applied,
    ) -> None:
        """Mark ``(md_path, entry_id, previous mark[, hash])`` forgotten."""
        if not entries:
            return
        store = self._store(kind)
        await store.deprecate(
            scope,
            entries=[(row[0], row[1]) for row in entries],
            replaced_by=FORGOTTEN_MARK,
            only_if=NO_SOURCE_CHECK,
        )
        applied.marked.setdefault(kind, []).extend(list(row) for row in entries)

    # ── scenes ───────────────────────────────────────────────────────────

    async def _scenes(
        self,
        plan: ForgetPlan,
        prepared: Prepared,
        replaced: Mapping[str, MemoryRef],
        reached: Mapping[str, str],
        applied: Applied,
    ) -> None:
        """Each scene edit; then a user's scenes left without a live episode."""
        scope = plan.scope
        if plan.scenes:
            async with scenes_lock(scope):
                for edit in plan.scenes:
                    current = edit.current
                    if edit.image_id is not None and current is not None:
                        await self._write_scene(
                            scope,
                            current,
                            current.item.model_copy(
                                update={
                                    "version": current.item.version + 1,
                                    "narrative": strip_reference(
                                        current.item.narrative, edit.image_id
                                    ),
                                    "summary": strip_reference(
                                        current.item.summary, edit.image_id
                                    ),
                                }
                            ),
                            applied,
                        )
                        continue
                    scene = prepared.scenes.get(edit.scene_id)
                    if scene is not None and current is not None:
                        ids = [
                            replaced[m].entry_id if m in replaced else m
                            for m in scene.member_ids
                        ]
                        scene = scene.model_copy(update={"member_ids": ids})
                        await self._write_scene(scope, current, scene, applied)
                        current = None
                    if current is not None:
                        await self._mark(
                            "scene",
                            scope,
                            [(current.ref.md_path, current.ref.entry_id, None)],
                            applied,
                        )
                    if edit.whole or plan.is_group:
                        await self._cluster(
                            plan, edit.scene_id, replaced, reached, applied
                        )
        if not plan.is_group and plan.episodes:
            orphaned = await forget_orphaned_scenes(
                self._stores.episode, self._stores.scene, scope
            )
            applied.orphaned += [[s.ref.md_path, s.ref.entry_id] for s in orphaned]

    async def _write_scene(
        self, scope: Scope, current: Stored[Any], scene: Any, applied: Applied
    ) -> None:
        ref = await self._stores.scene.append(
            scope, scene, supersedes=current, only_if=NO_SOURCE_CHECK
        )
        if ref is not None:
            applied.written.setdefault("scene", []).append(
                [current.ref.md_path, current.ref.entry_id, ref.md_path, ref.entry_id]
            )

    async def _cluster_members(self, plan: ForgetPlan) -> dict[str, str]:
        """Each member of the plan's scene clusters, by the version it leads to.

        A member is named by the id it joined with; the forget reaches it
        through the version that id leads to before anything is written.
        """
        members: set[str] = set()
        for edit in plan.scenes:
            cluster = await self._stores.cluster.get(edit.scene_id)
            if cluster is not None:
                members.update(cluster.members)
        if not members:
            return {}
        now = await current_versions(self._stores.episode, plan.scope, members)
        return {m: v.ref.entry_id for m, v in now.items() if v is not None}

    async def _cluster(
        self,
        plan: ForgetPlan,
        scene_id: str,
        replaced: Mapping[str, MemoryRef],
        reached: Mapping[str, str],
        applied: Applied,
    ) -> None:
        """The scene's cluster follows its episodes; a forgotten scene's goes."""
        scope = plan.scope
        cluster = await self._stores.cluster.get(scene_id)
        if cluster is None:
            return
        whole = any(e.scene_id == scene_id and e.whole for e in plan.scenes)
        dropped = {e.ref.entry_id for e in plan.episodes}
        members: list[str] = []
        for m in cluster.members:
            now = reached.get(m, m)
            if m in dropped or now in dropped:
                continue
            members.append(replaced[now].entry_id if now in replaced else m)
        applied.clusters.append(
            {"cluster": cluster.model_dump(mode="json"), "whole": whole}
        )
        await self._stores.cluster.delete(scene_id)
        if not whole and members:
            await self._stores.cluster.upsert(
                scope,
                scene_cluster_kind(scope),
                cluster.model_copy(
                    update={"members": members, "count": len(members), "preview": []}
                ),
                member_type="episode",
            )

    # ── the group profile ────────────────────────────────────────────────

    async def _group_profile(self, plan: ForgetPlan, applied: Applied) -> None:
        """Forgotten and stale items are marked; the stale ones' segments queued."""
        items = [*plan.group_forgotten, *plan.group_stale]
        if not items and not plan.group_queue:
            return
        scope = plan.scope
        async with group_profile_lock(scope.app_id, scope.project_id, scope.owner_id):
            # An item another forget marked is that forget's; a superseded one
            # keeps its successor's id for the undo to put back.
            unique = {s.ref.entry_id: s for s in items if s.item.status != "forgotten"}
            await self._mark(
                "group_profile",
                scope,
                [
                    (s.ref.md_path, s.ref.entry_id, s.item.superseded_by)
                    for s in unique.values()
                ],
                applied,
            )
            if plan.group_queue:
                pending = await self._stores.group_profile.pending(scope)
                added = [m for m in plan.group_queue if m not in pending]
                if added:
                    await self._stores.group_profile.set_pending(
                        scope, [*pending, *added]
                    )
                    applied.group_queued.extend(added)

    # ── profiles ─────────────────────────────────────────────────────────

    def profile_guard(self, plan: ForgetPlan) -> AbstractAsyncContextManager[None]:
        """Held from a user's profile edit until the log says "forgotten": an
        update waiting on it reads none of this, and none in flight writes it
        back. A group's members are each locked for their own edit."""
        return nullcontext() if plan.is_group else profile_lock(plan.scope)

    async def profiles(self, plan: ForgetPlan, applied: Applied) -> None:
        """The source rule over the owner's profile documents, as they are now.

        A user's caller holds :meth:`profile_guard`.
        """
        scope = plan.scope
        if not plan.is_group:
            held = await self._stores.profile.read(scope)
            if held is None:
                return
            edit = user_edit(
                held,
                plan,
                already=await self._logs.forgotten_segments(
                    scope,
                    except_session=plan.target.session_id
                    if plan.target.kind == "session"
                    else None,
                ),
            )
            await self._take_out(scope, held, edit, applied)
            return
        for member_id in plan.members_removed:
            async with profile_lock(scope, member_id):
                removed = await self._stores.profile.remove(scope, member_id)
            if removed is None:
                continue
            await self._eraser.erase(scope, {"user_profile": [removed.ref]})
            applied.members_removed.append(
                {"member_id": member_id, "profile": member_record(scope, removed.item)}
            )
        for planned in plan.profiles:
            member_id = planned.person_id
            async with profile_lock(scope, member_id):
                held = await self._stores.profile.read(scope, member_id)
                if held is None:
                    continue
                await self._take_out(scope, held, member_edit(held, plan), applied)

    async def _take_out(
        self, scope: Scope, held: Profile, edit: ProfileEdit, applied: Applied
    ) -> None:
        if not edit.changes:
            return
        kept, change = take_out_edit(held, edit)
        if change is None:
            return
        await self._stores.profile.write(scope, kept)
        applied.profiles.append(change)

    async def _erase(self, scope: Scope, applied: Applied) -> None:
        """The rows of what left default reads go at once (the cascade, when it
        projects the files again, writes them back with their marks)."""
        erased: dict[str, list[MemoryRef]] = {}
        for kind, rows in applied.marked.items():
            erased.setdefault(_TABLES[kind], []).extend(
                _ref(scope, row[0], row[1]) for row in rows
            )
        for kind, rows in applied.written.items():
            erased.setdefault(_TABLES[kind], []).extend(
                _ref(scope, row[0], row[1]) for row in rows
            )
        erased.setdefault(_EPISODE_TABLE, []).extend(
            _ref(scope, row[0], row[1]) for row in applied.orphaned
        )
        erased = {table: refs for table, refs in erased.items() if refs}
        if erased:
            await self._eraser.erase(scope, erased)

    def _store(self, kind: str) -> Any:
        return {
            "episode": self._stores.episode,
            "image_memory": self._stores.image_memory,
            "scene": self._stores.scene,
            "group_profile": self._stores.group_profile,
        }[kind]


def take_out_edit(
    held: Profile, edit: ProfileEdit
) -> tuple[Profile, dict[str, Any] | None]:
    """``held`` without what ``edit`` takes, its stale segments queued; and the
    log's record of what went, each value with its place (``None``: nothing)."""
    chosen = {id(getattr(held, bucket)[n]) for bucket, n in edit.items}
    positions = set(edit.history)
    whole = edit.whole
    kept, taken = profile_items.take_out(
        held,
        item=lambda i: whole or id(i) in chosen,
        entry=lambda n, _entry: whole or n in positions,
    )
    summary = taken.summary
    if whole and held.summary:
        kept = kept.model_copy(update={"summary": ""})
        summary = held.summary
    if not (taken.items or taken.history or summary or edit.queue):
        return held, None
    if edit.queue:
        kept = kept.model_copy(
            update={"pending_memcell_ids": [*held.pending_memcell_ids, *edit.queue]}
        )
    return kept, {
        "person": held.owner_id,
        "items": [[bucket, n, item] for bucket, n, item in taken.items],
        "history": [[n, entry] for n, entry in taken.history],
        "summary": summary,
        "queued": list(edit.queue),
        "by_segments": [list(place) for place in edit.by_segments],
        "forgotten": list(edit.forgotten),
        "stale": list(edit.stale),
    }


def member_record(scope: Scope, profile: Profile) -> dict[str, Any]:
    """A member's whole profile in the group, as the forget log keeps it."""
    document = profile.model_dump(mode="json", exclude={"pending_memcell_ids"})
    return {
        "group_id": scope.owner_id,
        "member_id": profile.owner_id,
        "profile": {**document, "history": []},
        "pending_memcell_ids": list(profile.pending_memcell_ids),
        "history": list(profile.history),
    }


def member_from_record(record: Mapping[str, Any]) -> Profile:
    """The profile :func:`member_record` kept."""
    return Profile.model_validate(
        {
            **record["profile"],
            "history": list(record.get("history") or []),
            "pending_memcell_ids": list(record.get("pending_memcell_ids") or []),
        }
    )


def _without_picture(episode: Episode, image_id: str) -> Episode:
    """The episode written again without any reference to the picture."""
    return episode.model_copy(
        update={
            "episode": strip_reference(episode.episode, image_id),
            "summary": strip_reference(episode.summary, image_id),
            # The reference list follows the body: a version that no longer
            # names the picture must not claim to refer to it.
            "image_ids": [kept for kept in episode.image_ids if kept != image_id],
        }
    )


def strip_reference(text: str, image_id: str) -> str:
    """Remove every reference to ``image_id``; the rest of ``text`` stays as it is.

    The two spellings in use are ``(img-1: <id>)`` from the picture extractor
    and ``[IMAGE: <id>]`` from a text-only render; each shape is removed
    whole, outermost first, so no ``(img-1: )`` is left inside a sentence.
    A span takes the spaces before it along (``a kite (img-1: <id>).`` reads
    ``a kite.``), or the spaces after it when it opens a line; line breaks,
    paragraphs and every other character of the body are kept.
    """
    if image_id not in text:
        return text
    spans = [
        (found.start(), found.end())
        for found in re.finditer(re.escape(f"[IMAGE: {image_id}]"), text)
    ]
    spans += [
        (start, end)
        for start, end in _parenthesised(text)
        if image_id in text[start:end]
    ]
    kept: list[str] = []
    last = 0
    for start, end in _outermost(spans):
        if start < last:
            continue
        cut = start
        while cut > last and text[cut - 1] in _SPACES:
            cut -= 1
        kept.append(text[last:cut])
        last = end
        if cut == 0 or text[cut - 1] == "\n":
            while last < len(text) and text[last] in _SPACES:
                last += 1
    kept.append(text[last:])
    return "".join(kept)


_SPACES = " \t"
"""The whitespace a removed reference takes along: never a line break."""


def _outermost(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """``spans`` in order, each one inside another dropped."""
    found: list[tuple[int, int]] = []
    for start, end in sorted(spans, key=lambda span: (span[0], -span[1])):
        if found and start < found[-1][1]:
            continue
        found.append((start, end))
    return found


def _parenthesised(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` of every ``(...)`` span in ``text``, outermost and
    balanced (a scan: a description may itself hold parentheses, which a
    greedy pattern eats)."""
    spans: list[tuple[int, int]] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            if depth == 0:
                start = index
            depth += 1
        elif char == ")" and depth:
            depth -= 1
            if depth == 0:
                spans.append((start, index + 1))
    return spans


async def _always() -> bool:
    """``only_if`` of a write whose source is the forget itself."""
    return True


def _ref(scope: Scope, md_path: str, entry_id: str) -> MemoryRef:
    """The index row of one entry, keyed as the derived index keys it."""
    return MemoryRef(
        id=f"{scope.owner_id}_{entry_id}", entry_id=entry_id, md_path=md_path
    )
