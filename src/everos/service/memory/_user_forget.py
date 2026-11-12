"""Forgetting in one user's memory: every target, a dry run, an undo by id.

The use case behind a user's ``POST /memory/forget`` and its undo (a group's
is :mod:`._group_forget`). A forget **marks**: every version it reaches gets
the forget marker in its file's ``deprecated_entries``, its index row is
erased at once, and nothing leaves markdown -- so an undo is a status change,
and a question about the past can still find what was forgotten. Five
targets:

* ``record`` -- records by the ids a read returned, of any kind: episodes,
  picture memories, scene versions (the scene's current version, and its
  cluster, so it does not grow back), profile items and replaced values;
* ``session`` -- one conversation (:mod:`._forget_session`);
* ``owner`` -- everything of the user: every current episode, picture memory
  and scene, and the whole profile (items, replaced values, summary);
* ``condition`` -- the memories of some kinds whose time falls in a range
  (``start`` inclusive, ``end`` exclusive, epoch ms; a profile item by when it
  was observed);
* ``image`` -- one picture by content hash (:mod:`._forget`): its memories, and
  the references to it in episodes and scenes, written again without it.

The forget follows the sources, as a conversation's does: a scene none of
whose episodes is alive any more is forgotten; a picture file no live memory of
the user refers to is taken out of circulation; a profile item goes once every
segment it rests on is forgotten (or every picture it was read off).

``dry_run`` reads only and answers with what the real run would touch. A real
run returns its ``forget_id``, the undo token: ``/memory/forget/undo`` takes it
and brings back exactly what that forget marked -- except an entry another
live forget names too, which stays forgotten. One forget, undo or update of
one user at a time (:func:`~everos.memory.user_forget_lock`).
"""

from __future__ import annotations

__all__ = [
    "UserForgetManager",
    "UserForgetResult",
    "UserForgetTarget",
    "UserUndoResult",
]

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from everos.core.errors import InvalidInputError, NotFoundError, SessionBusyError
from everos.core.locks import LOCKS, session_key
from everos.core.observability.logging import get_logger
from everos.memory import user_forget_lock, user_profile_lock, user_scenes_lock
from everos.memory.extract.pipeline import profile_items
from everos.memory.extract.scene import (
    alive_members,
    forget_orphaned_scenes,
    newest_versions,
    restore_revived_scenes,
)
from everos.memory.models import (
    FORGOTTEN_MARK,
    Cluster,
    Episode,
    ImageMemory,
    Profile,
    Scene,
)
from everos.memory.ports import NO_SOURCE_CHECK, Scope, Stored

from ._forget_common import (
    Named,
    forgotten_segments,
    named_by_others,
    removed_by_others,
    removed_item_ids,
    segments_of,
)
from ._forget_session import current_episodes_of, current_pictures_of
from ._records import (
    episode_end,
    kind_of,
    memory_id,
    picture_end,
    scene_chain,
)

if TYPE_CHECKING:
    from everos.memory.deps import Stores
    from everos.memory.ports import ForgetEntry, IndexEraser

    from ._extraction import ExtractionSupervisor
    from ._forget import ForgetManager, ForgetResult
    from ._forget_session import SessionForgetManager, SessionForgetResult
    from ._undo import SessionUndoResult, UndoManager, UndoResult

logger = get_logger("everos.service.memory")

type UserTargetKind = Literal["record", "session", "owner", "condition", "image"]
type ForgettableKind = Literal["episode", "image_memory", "scene", "profile"]

_SETTLE_SECONDS = 120.0
"""How long a conversation's forget waits for its running extractions."""


@dataclass(frozen=True)
class UserForgetTarget:
    """What a forget in a user's memory addresses.

    Exactly the fields of its ``kind`` are set; the request DTO validates it.
    """

    kind: UserTargetKind
    record_ids: tuple[str, ...] = ()
    session_id: str | None = None
    image_id: str | None = None
    memory_types: tuple[ForgettableKind, ...] = ()
    start: int | None = None
    end: int | None = None

    def label(self, owner_id: str) -> str:
        """The forget log's ``target_id``: what an undo by target finds."""
        if self.kind == "session":
            return self.session_id or ""
        if self.kind == "image":
            return self.image_id or ""
        if self.kind == "owner":
            return owner_id
        if self.kind == "condition":
            return (
                f"{','.join(self.memory_types)}:{self.start or ''}-{self.end or ''}"
            )[:500]
        return ",".join(self.record_ids)[:500]

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "record_ids": list(self.record_ids),
            "session_id": self.session_id,
            "image_id": self.image_id,
            "memory_types": list(self.memory_types),
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> UserForgetTarget:
        return cls(
            kind=data["kind"],
            record_ids=tuple(data.get("record_ids") or ()),
            session_id=data.get("session_id"),
            image_id=data.get("image_id"),
            memory_types=tuple(data.get("memory_types") or ()),
            start=data.get("start"),
            end=data.get("end"),
        )


@dataclass(frozen=True)
class UserForgetResult:
    """What a forget in a user's memory touched (or would, on a dry run).

    Records are named as a read names them (``<user>_<entry>``); profile items
    by ``item_id``.

    Attributes:
        forget_id: The undo token (``fg_<hex>``); ``""`` on a dry run.
        dry_run: Whether nothing was changed.
        target: What the forget addressed.
        episodes: Episode versions taken out of default reads: marked
            forgotten, or (a picture's forget) replaced by a version without
            the picture.
        new_episodes: Those replacements.
        image_memories: Picture memories marked forgotten.
        images: Picture files taken out of circulation.
        scenes: Scene versions taken out of default reads.
        new_scenes: Scene versions written without the picture.
        profile_items: Profile items and replaced values taken out of
            ``user.md``.
    """

    forget_id: str
    dry_run: bool
    target: UserForgetTarget
    episodes: tuple[str, ...] = ()
    new_episodes: tuple[str, ...] = ()
    image_memories: tuple[str, ...] = ()
    images: tuple[str, ...] = ()
    scenes: tuple[str, ...] = ()
    new_scenes: tuple[str, ...] = ()
    profile_items: tuple[str, ...] = ()


@dataclass(frozen=True)
class UserUndoResult:
    """What an undo of a user's forget put back.

    ``applied`` is ``False`` when the forget was undone already.
    """

    forget_id: str
    applied: bool
    target: UserForgetTarget | None = None
    episodes: tuple[str, ...] = ()
    image_memories: tuple[str, ...] = ()
    images: tuple[str, ...] = ()
    scenes: tuple[str, ...] = ()
    profile_items: int = 0


@dataclass
class _Plan:
    """What a forget by records, owner, condition or session marks."""

    episodes: list[Stored[Episode]] = field(default_factory=list)
    pictures: list[Stored[ImageMemory]] = field(default_factory=list)
    scenes: list[Stored[Scene]] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)
    whole_profile: bool = False
    segments: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    orphaned: list[Stored[Scene]] = field(default_factory=list)
    profile_removed: list[str] = field(default_factory=list)

    def add_episode(self, stored: Stored[Episode]) -> None:
        if all(e.ref.entry_id != stored.ref.entry_id for e in self.episodes):
            self.episodes.append(stored)

    def add_picture(self, stored: Stored[ImageMemory]) -> None:
        if all(p.ref.entry_id != stored.ref.entry_id for p in self.pictures):
            self.pictures.append(stored)

    def add_scene(self, stored: Stored[Scene]) -> None:
        if all(s.ref.entry_id != stored.ref.entry_id for s in self.scenes):
            self.scenes.append(stored)


class UserForgetManager:
    """Forgets in a user's memory, their dry runs and their undos.

    Args:
        stores: Markdown and system state.
        eraser: Erases the index rows of what a forget marks; hands what an
            undo puts back to the cascade.
        pictures: A picture's forget; ``None`` without read-side indexes.
        sessions: A conversation's forget.
        undoer: The undo of a picture's and a conversation's forget.
        extractions: The memorize extractions a conversation's forget waits
            for.
        session_wait_seconds: How long a conversation's forget waits for a
            cut in progress before answering 503.
    """

    def __init__(
        self,
        *,
        stores: Stores,
        eraser: IndexEraser,
        pictures: ForgetManager | None,
        sessions: SessionForgetManager,
        undoer: UndoManager,
        extractions: ExtractionSupervisor,
        session_wait_seconds: float,
    ) -> None:
        self._stores = stores
        self._eraser = eraser
        self._pictures = pictures
        self._sessions = sessions
        self._undoer = undoer
        self._extractions = extractions
        self._wait = session_wait_seconds

    # ── forget ───────────────────────────────────────────────────────────

    async def forget(
        self,
        *,
        scope: Scope,
        target: UserForgetTarget,
        reason: str | None = None,
        requested_by: str | None = None,
        dry_run: bool = True,
        strict: bool = True,
    ) -> UserForgetResult:
        """Forget ``target`` in ``scope``'s memory (or plan it, ``dry_run``).

        ``strict`` answers an id that names nothing with 404; the delete-by-id
        alias passes ``False`` and skips it.

        Raises:
            InvalidInputError: A record id of no kind of a user's.
            NotFoundError: A record id that names nothing (``strict``).
            SessionBusyError: The conversation stays busy (503).
            RuntimeError: A picture's forget without read-side indexes.
        """
        if target.kind == "image":
            assert target.image_id is not None
            if dry_run:
                return await self._plan_image(scope, target)
            return _from_picture(
                scope,
                target,
                await self.forget_picture(
                    scope=scope,
                    image_id=target.image_id,
                    reason=reason,
                    requested_by=requested_by,
                ),
            )
        if target.kind == "session":
            assert target.session_id is not None
            if dry_run:
                return await self._plan_session(scope, target)
            return _from_session(
                scope,
                target,
                await self.forget_conversation(
                    scope=scope,
                    session_id=target.session_id,
                    reason=reason,
                    requested_by=requested_by,
                ),
            )
        async with user_forget_lock(scope.app_id, scope.project_id, scope.owner_id):
            plan = await self._plan(scope, target, strict=strict)
            await self._derive(scope, plan, already=await self._already(scope))
            if dry_run:
                return _from_plan(scope, target, plan, forget_id="", dry_run=True)
            return await self._apply(scope, target, plan, reason, requested_by)

    async def forget_picture(
        self,
        *,
        scope: Scope,
        image_id: str,
        reason: str | None = None,
        requested_by: str | None = None,
    ) -> ForgetResult:
        """A picture's forget, under the user's lock (``/memory/forget`` alias)."""
        if self._pictures is None:
            raise RuntimeError("forget needs the read-side indexes")
        async with user_forget_lock(scope.app_id, scope.project_id, scope.owner_id):
            return await self._pictures.forget_image(
                scope=scope,
                image_id=image_id,
                reason=reason,
                requested_by=requested_by,
            )

    async def forget_conversation(
        self,
        *,
        scope: Scope,
        session_id: str,
        reason: str | None = None,
        requested_by: str | None = None,
    ) -> SessionForgetResult:
        """A conversation's forget, under the user's lock, once it has settled.

        A cut in progress finishes first (its lock; 503 past
        ``session_wait_seconds``) and the extractions running for the
        conversation end, so the forget misses nothing they write.

        Raises:
            SessionBusyError: The conversation stays busy.
        """
        await self._settle(scope, session_id)
        async with user_forget_lock(scope.app_id, scope.project_id, scope.owner_id):
            return await self._sessions.forget_session(
                scope=scope,
                session_id=session_id,
                reason=reason,
                requested_by=requested_by,
            )

    async def _settle(self, scope: Scope, session_id: str) -> None:
        key = session_key(scope.app_id, scope.project_id, session_id)
        try:
            async with LOCKS.hold("session", key, max_wait=self._wait):
                pass
        except TimeoutError:
            raise SessionBusyError(
                "the conversation is being memorized; nothing was forgotten, "
                "retry later"
            ) from None
        left = await self._extractions.settle([session_id], _SETTLE_SECONDS)
        if left:
            logger.warning("memory.forget_extractions_running", running=left)

    # ── plan ─────────────────────────────────────────────────────────────

    async def _plan(
        self, scope: Scope, target: UserForgetTarget, *, strict: bool
    ) -> _Plan:
        plan = _Plan()
        if target.kind == "record":
            await self._plan_records(scope, target.record_ids, plan, strict=strict)
        elif target.kind == "owner":
            await self._plan_owner(scope, plan)
        elif target.kind == "condition":
            await self._plan_condition(scope, target, plan)
        else:  # the routes send image and session elsewhere
            raise InvalidInputError(f"no plan for a {target.kind} target")
        plan.segments = list(
            dict.fromkeys(
                e.item.parent_id
                for e in plan.episodes
                if e.item.parent_type == "memcell" and e.item.parent_id
            )
        )
        return plan

    async def _plan_records(
        self,
        scope: Scope,
        record_ids: Sequence[str],
        plan: _Plan,
        *,
        strict: bool,
    ) -> None:
        stores = self._stores
        missing: list[str] = []
        profile: Profile | None = None
        profile_read = False
        forgotten_items: set[str] | None = None
        for record_id in dict.fromkeys(record_ids):
            kind, entry_id = kind_of(scope, record_id)
            if kind == "episode":
                end = await episode_end(stores, scope, entry_id)
                if end is None:
                    missing.append(record_id)
                elif end[1] is None:
                    plan.add_episode(end[0])
            elif kind == "image_memory":
                found = await picture_end(stores, scope, entry_id)
                if found is None:
                    missing.append(record_id)
                elif found[1] is None:
                    plan.add_picture(found[0])
            elif kind == "scene":
                chain = scene_chain(entry_id, await stores.scene.read_all(scope))
                newest = chain.newest
                if newest is None:
                    missing.append(record_id)
                elif newest.mark is None:
                    plan.add_scene(newest.stored)
            else:
                if not profile_read:
                    profile, profile_read = await stores.profile.read(scope), True
                history = profile.history if profile is not None else []
                if profile_items.find_item(profile, history, entry_id) is not None:
                    if entry_id not in plan.item_ids:
                        plan.item_ids.append(entry_id)
                    continue
                if forgotten_items is None:
                    forgotten_items = await self._forgotten_item_ids(scope)
                if entry_id not in forgotten_items:
                    missing.append(record_id)
        if missing and strict:
            raise NotFoundError(f"no record {missing[0]!r} of user {scope.owner_id!r}")

    async def _plan_owner(self, scope: Scope, plan: _Plan) -> None:
        stores = self._stores
        for stored, mark in await stores.episode.read_all(scope):
            if mark is None:
                plan.add_episode(stored)
        for row in await stores.image_memory.read_all(scope):
            if row.item.superseded_by is None:
                plan.add_picture(row)
        for version, mark in await newest_versions(stores.scene, scope):
            if mark is None:
                plan.add_scene(version)
        plan.whole_profile = True

    async def _plan_condition(
        self, scope: Scope, target: UserForgetTarget, plan: _Plan
    ) -> None:
        stores = self._stores
        kinds = set(target.memory_types)

        def within(at: int | None) -> bool:
            if at is None:
                return False
            return (target.start is None or at >= target.start) and (
                target.end is None or at < target.end
            )

        if "episode" in kinds:
            for stored, mark in await stores.episode.read_all(scope):
                if mark is None and within(stored.item.timestamp):
                    plan.add_episode(stored)
        if "image_memory" in kinds:
            for row in await stores.image_memory.read_all(scope):
                if row.item.superseded_by is None and within(row.item.timestamp):
                    plan.add_picture(row)
        if "scene" in kinds:
            for version, mark in await newest_versions(stores.scene, scope):
                if mark is None and within(version.item.timestamp):
                    plan.add_scene(version)
        if "profile" in kinds:
            profile = await stores.profile.read(scope)
            if profile is not None:
                entries: list[Mapping[str, Any]] = [
                    item.model_dump()
                    for bucket in profile_items.BUCKETS
                    for item in getattr(profile, bucket)
                ]
                entries += profile.history
                for entry in entries:
                    at = entry.get("observed_at") or entry.get("source_end")
                    item_id = entry.get("item_id")
                    if (
                        isinstance(item_id, str)
                        and isinstance(at, int)
                        and within(at)
                        and item_id not in plan.item_ids
                    ):
                        plan.item_ids.append(item_id)

    async def _derive(self, scope: Scope, plan: _Plan, *, already: set[str]) -> None:
        """The plan's consequences: orphaned scenes, files, profile items."""
        stores = self._stores
        gone = {e.ref.entry_id for e in plan.episodes}
        targeted = {s.ref.entry_id for s in plan.scenes}
        if gone:
            for version, mark in await newest_versions(stores.scene, scope):
                if mark is not None or version.ref.entry_id in targeted:
                    continue
                alive = await alive_members(
                    stores.episode, scope, version.item.member_ids
                )
                if not [m for m in alive if m.ref.entry_id not in gone]:
                    plan.orphaned.append(version)
        marked = {p.ref.entry_id for p in plan.pictures}
        for image_id in dict.fromkeys(p.item.image_id for p in plan.pictures):
            if not image_id:
                continue
            occurrences = await stores.image_memory.read_by_image(scope, image_id)
            if not [
                o
                for o in occurrences
                if o.item.superseded_by is None and o.ref.entry_id not in marked
            ]:
                plan.files.append(image_id)
        _, removed, _ = _profile_removal(
            await stores.profile.read(scope), plan, already=already
        )
        plan.profile_removed = removed_item_ids(removed)

    async def _already(self, scope: Scope) -> set[str]:
        return await forgotten_segments(self._stores, scope)

    async def _forgotten_item_ids(self, scope: Scope) -> set[str]:
        found: set[str] = set()
        for entry in await self._stores.forget_log.live_entries(
            app_id=scope.app_id, project_id=scope.project_id, owner_id=scope.owner_id
        ):
            found |= set(removed_item_ids(entry.removed_profile_items))
        return found

    async def _plan_session(
        self, scope: Scope, target: UserForgetTarget
    ) -> UserForgetResult:
        """A conversation's forget as a plan: what it would mark."""
        assert target.session_id is not None
        stores = self._stores
        async with user_forget_lock(scope.app_id, scope.project_id, scope.owner_id):
            plan = _Plan()
            episodes = await current_episodes_of(stores, scope, target.session_id)
            plan.episodes = await stores.episode.read_many(
                scope, [e for _, e in episodes]
            )
            pictures = await current_pictures_of(stores, scope, target.session_id)
            plan.pictures = await stores.image_memory.read_many(
                scope, [e for _, e, _ in pictures]
            )
            plan.segments = sorted(
                await segments_of(stores, scope, [target.session_id])
            )
            await self._derive(
                scope,
                plan,
                already=await forgotten_segments(
                    stores, scope, except_session=target.session_id
                ),
            )
        return _from_plan(scope, target, plan, forget_id="", dry_run=True)

    async def _plan_image(
        self, scope: Scope, target: UserForgetTarget
    ) -> UserForgetResult:
        """A picture's forget as a plan: what it would mark and rewrite."""
        image_id = target.image_id
        assert image_id is not None
        stores = self._stores
        async with user_forget_lock(scope.app_id, scope.project_id, scope.owner_id):
            occurrences = [
                row
                for row in await stores.image_memory.read_by_image(scope, image_id)
                if row.item.superseded_by is None
            ]
            referring = await stores.episode.read_referring(scope, image_id)
            scenes = [
                version
                for version, mark in await newest_versions(stores.scene, scope)
                if mark is None
                and (
                    image_id in version.item.narrative
                    or image_id in version.item.summary
                )
            ]
            _, removed, _ = profile_items.forget_images(
                await stores.profile.read(scope), [image_id]
            )
        return UserForgetResult(
            forget_id="",
            dry_run=True,
            target=target,
            episodes=tuple(memory_id(scope, e.ref.entry_id) for e in referring),
            image_memories=tuple(memory_id(scope, o.ref.entry_id) for o in occurrences),
            images=(image_id,),
            scenes=tuple(memory_id(scope, s.ref.entry_id) for s in scenes),
            profile_items=tuple(removed_item_ids(removed)),
        )

    # ── apply ────────────────────────────────────────────────────────────

    async def _apply(
        self,
        scope: Scope,
        target: UserForgetTarget,
        plan: _Plan,
        reason: str | None,
        requested_by: str | None,
    ) -> UserForgetResult:
        stores = self._stores
        forgotten = FORGOTTEN_MARK
        if plan.episodes:
            await stores.episode.deprecate(
                scope,
                entries=[(e.ref.md_path, e.ref.entry_id) for e in plan.episodes],
                replaced_by=forgotten,
                only_if=NO_SOURCE_CHECK,
            )
        if plan.pictures:
            await stores.image_memory.deprecate(
                scope,
                entries=[(p.ref.md_path, p.ref.entry_id) for p in plan.pictures],
                replaced_by=forgotten,
                only_if=NO_SOURCE_CHECK,
            )
        clusters: list[dict[str, Any]] = []
        if plan.scenes:
            async with user_scenes_lock(scope.app_id, scope.project_id, scope.owner_id):
                await stores.scene.deprecate(
                    scope,
                    entries=[(s.ref.md_path, s.ref.entry_id) for s in plan.scenes],
                    replaced_by=forgotten,
                    only_if=NO_SOURCE_CHECK,
                )
                # Its cluster goes too, or the next episode joining it would
                # write the forgotten scene again.
                for scene in plan.scenes:
                    cluster = await stores.cluster.get(scene.item.scene_id)
                    if cluster is not None and cluster.id is not None:
                        clusters.append(cluster.model_dump(mode="json"))
                        await stores.cluster.delete(cluster.id)
        orphaned = (
            await forget_orphaned_scenes(stores.episode, stores.scene, scope)
            if plan.episodes
            else []
        )
        for image_id in plan.files:
            await stores.images.forget(image_id)
        erased: dict[str, list[Any]] = {}
        rows = [e.ref for e in plan.episodes] + [
            s.ref for s in (*plan.scenes, *orphaned)
        ]
        if rows:  # a scene version is a row of the episode table
            erased["episode"] = rows
        if plan.pictures:
            erased["image_memory"] = [p.ref for p in plan.pictures]
        if erased:
            await self._eraser.erase(scope, erased)
        # Under the profile lock until the log says "forgotten": an update
        # waiting on it reads none of this, and none in flight writes it back.
        async with user_profile_lock(scope.app_id, scope.project_id, scope.owner_id):
            existing = await stores.profile.read(scope)
            written, removed, by_segments = _profile_removal(
                existing, plan, already=await self._already(scope)
            )
            if removed and written is not None:
                await stores.profile.write(scope, written)
            entry = await stores.forget_log.record(
                target=target.kind,
                target_id=target.label(scope.owner_id),
                app_id=scope.app_id,
                project_id=scope.project_id,
                owner_id=scope.owner_id,
                owner_type=scope.owner_type,
                reason=reason,
                requested_by=requested_by,
                entry_ids=tuple(p.ref.entry_id for p in plan.pictures),
                episode_ids=tuple((e.ref.entry_id, "") for e in plan.episodes),
                removed_profile_items=tuple(removed),
                undo={
                    "target": target.to_json(),
                    "episodes": [
                        [e.ref.md_path, e.ref.entry_id] for e in plan.episodes
                    ],
                    "image_memories": [
                        [p.ref.md_path, p.ref.entry_id, p.item.image_id]
                        for p in plan.pictures
                    ],
                    "scenes": [[s.ref.md_path, s.ref.entry_id] for s in plan.scenes],
                    "orphaned": [[s.ref.md_path, s.ref.entry_id] for s in orphaned],
                    "clusters": clusters,
                    "files": list(plan.files),
                    "segments": list(plan.segments),
                    "segment_items": by_segments,
                    "profile_summary": existing.summary
                    if plan.whole_profile and existing is not None
                    else None,
                },
            )
        plan.orphaned = list(orphaned)
        plan.profile_removed = removed_item_ids(removed)
        result = _from_plan(
            scope, target, plan, forget_id=entry.forget_id, dry_run=False
        )
        logger.info(
            "memory.user.forgotten",
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            forget_id=entry.forget_id,
            target=target.kind,
            episodes=len(plan.episodes),
            image_memories=len(plan.pictures),
            scenes=len(result.scenes),
            images=len(plan.files),
            profile_items=len(result.profile_items),
            reason=reason,
        )
        return result

    # ── undo ─────────────────────────────────────────────────────────────

    async def undo(self, *, scope: Scope, forget_id: str) -> UserUndoResult:
        """Reverse the user's forget ``forget_id``.

        Raises:
            NotFoundError: The user has no forget of that id.
        """
        async with user_forget_lock(scope.app_id, scope.project_id, scope.owner_id):
            entry = await self._stores.forget_log.get(forget_id)
            if (
                entry is None
                or entry.owner_type != scope.owner_type
                or entry.owner_id != scope.owner_id
                or entry.app_id != scope.app_id
                or entry.project_id != scope.project_id
            ):
                raise NotFoundError(
                    f"no forget {forget_id!r} of user {scope.owner_id!r}"
                )
            target = _target_of(entry)
            if entry.undone:
                return UserUndoResult(forget_id=forget_id, applied=False, target=target)
            keep = await named_by_others(self._stores, scope, except_id=forget_id)
            if entry.target == "image":
                undone = await self._undoer.undo_image_entry(
                    scope=scope, image_id=entry.target_id, entry=entry, keep=keep
                )
                return UserUndoResult(
                    forget_id=forget_id,
                    applied=undone.applied,
                    target=target,
                    episodes=tuple(memory_id(scope, e) for e in undone.episodes),
                    image_memories=tuple(
                        memory_id(scope, p.entry_id) for p in undone.pictures
                    ),
                    images=(entry.target_id,),
                    profile_items=undone.profiles,
                )
            if entry.target == "session":
                session = await self._undoer.undo_session_entry(
                    scope=scope, session_id=entry.target_id, entry=entry, keep=keep
                )
                return UserUndoResult(
                    forget_id=forget_id,
                    applied=session.applied,
                    target=target,
                    episodes=tuple(memory_id(scope, e) for e in session.episodes),
                    images=tuple(session.images),
                    profile_items=session.profiles,
                )
            return await self._undo_marks(scope, entry, target, keep)

    async def undo_picture(self, *, scope: Scope, image_id: str) -> UndoResult:
        """Undo the newest live forget of one picture (the legacy undo body)."""
        async with user_forget_lock(scope.app_id, scope.project_id, scope.owner_id):
            entry = await self._stores.forget_log.latest_live(
                target="image",
                target_id=image_id,
                app_id=scope.app_id,
                project_id=scope.project_id,
                owner_id=scope.owner_id,
            )
            keep = (
                await named_by_others(self._stores, scope, except_id=entry.forget_id)
                if entry is not None
                else Named()
            )
            return await self._undoer.undo_image_entry(
                scope=scope, image_id=image_id, entry=entry, keep=keep
            )

    async def undo_conversation(
        self, *, scope: Scope, session_id: str
    ) -> SessionUndoResult:
        """Undo the newest live forget of one conversation (the legacy undo)."""
        async with user_forget_lock(scope.app_id, scope.project_id, scope.owner_id):
            entry = await self._stores.forget_log.latest_live(
                target="session",
                target_id=session_id,
                app_id=scope.app_id,
                project_id=scope.project_id,
                owner_id=scope.owner_id,
            )
            keep = (
                await named_by_others(self._stores, scope, except_id=entry.forget_id)
                if entry is not None
                else Named()
            )
            return await self._undoer.undo_session_entry(
                scope=scope, session_id=session_id, entry=entry, keep=keep
            )

    async def _undo_marks(
        self,
        scope: Scope,
        entry: ForgetEntry,
        target: UserForgetTarget | None,
        keep: Named,
    ) -> UserUndoResult:
        """Undo a forget by records, owner or condition: exactly its marks."""
        stores = self._stores
        undo = entry.undo or {}
        episodes = await _clear(
            stores.episode,
            scope,
            [
                (str(md), str(e))
                for md, e in undo.get("episodes") or []
                if e not in keep.episodes
            ],
        )
        pictures = await self._unmark_pictures(
            scope,
            [
                (str(row[0]), str(row[1]))
                for row in undo.get("image_memories") or []
                if row[1] not in keep.image_memories
            ],
        )
        async with user_scenes_lock(scope.app_id, scope.project_id, scope.owner_id):
            scenes = await _clear(
                stores.scene,
                scope,
                [
                    (str(md), str(e))
                    for md, e in undo.get("scenes") or []
                    if e not in keep.scenes
                ],
            )
            await self._restore_clusters(scope, undo.get("clusters") or [])
        revived = (
            await restore_revived_scenes(
                stores.episode, stores.scene, scope, keep=keep.scenes
            )
            if episodes
            else []
        )
        files = [str(i) for i in undo.get("files") or []]
        for image_id in files:
            await stores.images.restore(image_id)
        await self._requeue("episode", {md for md, _ in episodes})
        await self._requeue("image_memory", {md for md, _ in pictures})
        await self._requeue(
            "scene", {md for md, _ in scenes} | {s.ref.md_path for s in revived}
        )
        async with user_profile_lock(scope.app_id, scope.project_id, scope.owner_id):
            restored = await self._restore_profile(scope, entry, undo)
            await stores.forget_log.mark_undone(entry.forget_id)
        logger.info(
            "memory.user.undo_completed",
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            forget_id=entry.forget_id,
            episodes=len(episodes),
            image_memories=len(pictures),
            scenes=len(scenes) + len(revived),
            profile_items=restored,
        )
        return UserUndoResult(
            forget_id=entry.forget_id,
            applied=True,
            target=target,
            episodes=tuple(memory_id(scope, e) for _, e in episodes),
            image_memories=tuple(memory_id(scope, e) for _, e in pictures),
            images=tuple(files),
            scenes=tuple(
                dict.fromkeys(
                    [memory_id(scope, e) for _, e in scenes]
                    + [memory_id(scope, s.ref.entry_id) for s in revived]
                )
            ),
            profile_items=restored,
        )

    async def _unmark_pictures(
        self, scope: Scope, entries: Sequence[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Clear the forget marker of these picture memories (only that marker)."""
        if not entries:
            return []
        rows = await self._stores.image_memory.read_many(scope, [e for _, e in entries])
        still = {
            row.ref.entry_id for row in rows if row.item.superseded_by == FORGOTTEN_MARK
        }
        targets = [(md, e) for md, e in entries if e in still]
        if targets:
            await self._stores.image_memory.undeprecate(
                scope, entries=targets, only_if=NO_SOURCE_CHECK
            )
        return targets

    async def _restore_clusters(
        self, scope: Scope, clusters: Sequence[Mapping[str, Any]]
    ) -> None:
        for data in clusters:
            cluster = Cluster.model_validate(data)
            if cluster.id is None or await self._stores.cluster.get(cluster.id):
                continue
            await self._stores.cluster.upsert(
                scope, "user_scene", cluster, member_type="episode"
            )

    async def _restore_profile(
        self, scope: Scope, entry: ForgetEntry, undo: Mapping[str, Any]
    ) -> int:
        """Put back what the forget took from ``user.md``, and what other live
        forgets took because of it; the caller holds the profile lock."""
        stores = self._stores
        removed = list(entry.removed_profile_items)
        removed += await removed_by_others(
            stores,
            scope,
            [str(s) for s in undo.get("segments") or []],
            except_id=entry.forget_id,
        )
        summary = undo.get("profile_summary")
        if not removed and not summary:
            return 0
        existing = await stores.profile.read(scope)
        profile, restored, replaced = profile_items.restore_items(
            existing,
            removed,
            owner_id=scope.owner_id,
            timestamp=_newest_observed(removed),
        )
        if isinstance(summary, str) and summary and not profile.summary:
            profile = profile.model_copy(update={"summary": summary})
        elif not (restored or replaced):
            return 0
        await stores.profile.write(scope, profile)
        return restored

    async def _requeue(self, kind: str, md_paths: Collection[str]) -> None:
        if md_paths:
            await self._eraser.requeue(kind, md_paths)


# ── helpers ──────────────────────────────────────────────────────────────


def _profile_removal(
    existing: Profile | None, plan: _Plan, *, already: Collection[str]
) -> tuple[Profile | None, list[tuple[str, dict[str, Any]]], list[str]]:
    """``existing`` without what ``plan`` takes out of it.

    Returns ``(profile, removed, by_segments)``: the profile to write, the
    removed ``(bucket, item)`` pairs for the undo, and the ids the segment
    rule took (what another forget's undo may bring back).
    """
    if existing is None:
        return None, [], []
    if plan.whole_profile:
        removed = [
            (bucket, item.model_dump(mode="json"))
            for bucket in profile_items.BUCKETS
            for item in getattr(existing, bucket)
        ] + [("history", dict(entry)) for entry in existing.history]
        emptied = existing.model_copy(
            update={
                "explicit_info": [],
                "implicit_traits": [],
                "history": [],
                "summary": "",
            }
        )
        return emptied, removed, []
    profile: Profile | None = existing
    removed: list[tuple[str, dict[str, Any]]] = []
    by_segments: list[str] = []
    if plan.item_ids:
        profile, taken, _ = profile_items.forget_items(profile, plan.item_ids)
        removed += taken
    if plan.segments:
        profile, taken, _ = profile_items.forget_segments(
            profile, plan.segments, already=already
        )
        removed += taken
        by_segments = removed_item_ids(taken)
    if plan.files:
        profile, taken, _ = profile_items.forget_images(profile, plan.files)
        removed += taken
    return profile, removed, by_segments


async def _clear(
    store: Any, scope: Scope, entries: Sequence[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Clear the forget marker of ``entries`` where it still stands; those cleared.

    ``store`` is the episode or the scene store: both read one file's marks
    and drop marks by entry id.
    """
    if not entries:
        return []
    cleared: list[tuple[str, str]] = []
    for md_path in dict.fromkeys(md for md, _ in entries):
        marks = await store.deprecations(scope, md_path)
        cleared += [
            (md, e)
            for md, e in entries
            if md == md_path and marks.get(e) == FORGOTTEN_MARK
        ]
    if cleared:
        await store.clear_deprecations(scope, [e for _, e in cleared])
    return cleared


def _newest_observed(removed: Sequence[tuple[str, Mapping[str, Any]]]) -> int:
    newest = 0
    for _bucket, item in removed:
        value = item.get("observed_at")
        if isinstance(value, int):
            newest = max(newest, value)
    return newest


def _target_of(entry: ForgetEntry) -> UserForgetTarget | None:
    data = (entry.undo or {}).get("target")
    if isinstance(data, Mapping):
        return UserForgetTarget.from_json(data)
    if entry.target == "image":
        return UserForgetTarget(kind="image", image_id=entry.target_id)
    if entry.target == "session":
        return UserForgetTarget(kind="session", session_id=entry.target_id)
    return None


def _from_plan(
    scope: Scope,
    target: UserForgetTarget,
    plan: _Plan,
    *,
    forget_id: str,
    dry_run: bool,
) -> UserForgetResult:
    return UserForgetResult(
        forget_id=forget_id,
        dry_run=dry_run,
        target=target,
        episodes=tuple(memory_id(scope, e.ref.entry_id) for e in plan.episodes),
        image_memories=tuple(memory_id(scope, p.ref.entry_id) for p in plan.pictures),
        images=tuple(plan.files),
        scenes=tuple(
            dict.fromkeys(
                memory_id(scope, s.ref.entry_id) for s in (*plan.scenes, *plan.orphaned)
            )
        ),
        profile_items=tuple(plan.profile_removed),
    )


def _from_picture(
    scope: Scope, target: UserForgetTarget, result: ForgetResult
) -> UserForgetResult:
    return UserForgetResult(
        forget_id=result.forget_id,
        dry_run=False,
        target=target,
        episodes=tuple(memory_id(scope, e) for e in result.replaced_episodes),
        new_episodes=tuple(memory_id(scope, e) for e in result.episodes),
        image_memories=tuple(memory_id(scope, p.entry_id) for p in result.pictures),
        images=(result.image_id,),
        scenes=tuple(memory_id(scope, e) for e in result.replaced_scenes),
        new_scenes=tuple(memory_id(scope, e) for e in result.scenes),
        profile_items=tuple(result.profile_items),
    )


def _from_session(
    scope: Scope, target: UserForgetTarget, result: SessionForgetResult
) -> UserForgetResult:
    return UserForgetResult(
        forget_id=result.forget_id,
        dry_run=False,
        target=target,
        episodes=tuple(memory_id(scope, e) for e in result.episodes),
        image_memories=tuple(memory_id(scope, p) for p in result.pictures),
        images=tuple(result.images),
        scenes=tuple(memory_id(scope, s) for s in result.scenes),
        profile_items=tuple(result.profile_items),
    )
