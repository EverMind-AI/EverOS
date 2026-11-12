"""Undoing a forget: the one status change that brings forgotten material back.

What a forget changed, and what an undo puts back, in reverse order so that a
failure part-way leaves the store consistent with the log entry that still says
"forgotten":

1. **The episode versions.** A forget wrote each referring episode again
   without the picture and marked the original superseded. The undo appends the
   original's *pre-forget text* as a new entry and marks the forget's rewrite
   superseded by it. Appending, not un-marking the old entry: the store keeps
   one current version per episode, and a re-marked original plus a re-marked
   replacement would both claim to be current.
2. **The picture memories.** Every occurrence the forget marked is marked
   current again, so retrieval sees the picture. This is the one place a
   supersession mark is edited rather than superseded — the same edit
   :meth:`Deleter.delete_by_ids` makes when it drops the merge marks of entries
   it deleted — because a "restored" second copy of the same occurrence would
   be a new memory of a picture that never arrived again.
3. **The profile items.** Items (and replaced values) the forget removed
   because the picture alone established them, or because every segment they
   rest on was forgotten, are merged back; an item re-established meanwhile by
   a later conversation is not duplicated, because the merge is by description.
4. **The picture file.** Its ``forgotten_at`` is cleared. The bytes were never
   unlinked, which is what makes this a status change rather than a restore.

Idempotency is per forget: the log entry is stamped only after every step
succeeded, so a crashed undo can be retried, and a second undo of the same
forget finds no live entry and restores nothing.

An entry another live forget also names stays forgotten (``keep``): the
conversation forgotten after one of its episodes was forgotten by id, and
the first undone, leaves that episode forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from everos.core.observability.logging import get_logger
from everos.memory import user_profile_lock
from everos.memory.extract.pipeline import profile_items
from everos.memory.extract.scene import restore_revived_scenes
from everos.memory.models import Episode, ImageMemory
from everos.memory.ports import ForgetEntry, MemoryRef, Scope, Stored
from everos.service.memory._forget import (
    FORGET_REASON,
    ForgottenPicture,
    _always,
)
from everos.service.memory._forget_common import (
    Named,
    removed_by_others,
    segments_of,
)

if TYPE_CHECKING:
    from everos.memory.deps import Stores
    from everos.memory.ports import IndexEraser
    from everos.memory.ports.images import ImageStore

logger = get_logger("everos.service.memory")


@dataclass(frozen=True)
class UndoResult:
    """What one undo put back.

    Attributes:
        image_id: The picture, by content hash.
        forget_id: The log entry the undo reversed; empty when there was none.
        applied: Whether anything was restored. ``False`` is the honest answer
            for a second undo of the same forget, or for a picture that was
            never forgotten — the picture is still current, nothing was undone.
        image_memories: Picture memories marked current again (the picture's
            current occurrences, when nothing needed undoing).
        episodes: Entry ids of the restored episode versions.
        profiles: Profile items re-merged.
        pictures: One entry per picture memory, in entry order, with the name
            each arrival carried.
    """

    image_id: str
    forget_id: str
    applied: bool
    image_memories: int
    episodes: tuple[str, ...]
    profiles: int
    pictures: tuple[ForgottenPicture, ...] = ()


@dataclass(frozen=True)
class SessionUndoResult:
    """What undoing a conversation's forget put back.

    Attributes:
        session_id: The conversation.
        forget_id: The log entry reversed; empty when there was none.
        applied: Whether anything was restored.
        episodes: Entry ids of the episode versions marked current again.
        image_memories: Picture memories marked current again.
        images: Picture files brought back into circulation.
        profiles: Profile items put back into ``user.md``.
    """

    session_id: str
    forget_id: str
    applied: bool
    episodes: tuple[str, ...]
    image_memories: int
    images: tuple[str, ...] = ()
    profiles: int = 0


class UndoManager:
    """Reverses the newest forget of one picture for one owner.

    Args:
        stores: Write-side stores; markdown and the forget log are the truth.
        images: The picture files, whose status this clears.
        eraser: Hands re-written files back to the cascade, so the projection
            sees the marks the undo cleared.
    """

    def __init__(
        self,
        *,
        stores: Stores,
        images: ImageStore,
        eraser: IndexEraser,
    ) -> None:
        self._stores = stores
        self._images = images
        self._eraser = eraser

    async def undo_image(
        self, *, scope: Scope, image_id: str, keep: Named | None = None
    ) -> UndoResult:
        """Bring back everything the newest forget of ``image_id`` removed."""
        entry = await self._stores.forget_log.latest_live(
            target="image",
            target_id=image_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
        )
        return await self.undo_image_entry(
            scope=scope, image_id=image_id, entry=entry, keep=keep
        )

    async def undo_image_entry(
        self,
        *,
        scope: Scope,
        image_id: str,
        entry: ForgetEntry | None,
        keep: Named | None = None,
    ) -> UndoResult:
        """Bring back everything the picture forget ``entry`` removed.

        ``keep`` are the entries other live forgets name: they stay forgotten.
        """
        occurrences = await self._stores.image_memory.read_by_image(scope, image_id)
        if entry is None:
            # Never forgotten, or already undone. Clearing the file's status is
            # still correct (a forget whose log write failed leaves it set) and
            # the current occurrences are the honest report.
            await self._images.restore(image_id)
            current = [row for row in occurrences if row.item.superseded_by is None]
            return UndoResult(
                image_id=image_id,
                forget_id="",
                applied=False,
                image_memories=len(current),
                episodes=(),
                profiles=0,
                pictures=tuple(_picture_of(row, image_id) for row in current),
            )

        restored_episodes = await self._restore_episodes(scope, entry)
        restored_images = await self._restore_images(
            scope, entry, occurrences, keep=(keep or Named()).image_memories
        )
        async with user_profile_lock(scope.app_id, scope.project_id, scope.owner_id):
            restored_profile = await self._restore_profile(scope, entry)
        await self._images.restore(image_id)

        await self._eraser.requeue(
            "image_memory", {row.ref.md_path for row in restored_images}
        )
        if restored_episodes:
            await self._eraser.requeue(
                "episode", {ref.md_path for ref, _ in restored_episodes}
            )
        await self._stores.forget_log.mark_undone(entry.forget_id)

        logger.info(
            "memory.images.undo_completed",
            image_id=image_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            forget_id=entry.forget_id,
            image_memories=len(restored_images),
            episodes=len(restored_episodes),
            profiles=restored_profile,
        )
        return UndoResult(
            image_id=image_id,
            forget_id=entry.forget_id,
            applied=True,
            image_memories=len(restored_images),
            episodes=tuple(ref.entry_id for ref, _ in restored_episodes),
            profiles=restored_profile,
            pictures=tuple(_picture_of(row, image_id) for row in restored_images),
        )

    async def undo_session(
        self, *, scope: Scope, session_id: str, keep: Named | None = None
    ) -> SessionUndoResult:
        """Bring back everything the newest forget of one conversation removed."""
        entry = await self._stores.forget_log.latest_live(
            target="session",
            target_id=session_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
        )
        return await self.undo_session_entry(
            scope=scope, session_id=session_id, entry=entry, keep=keep
        )

    async def undo_session_entry(
        self,
        *,
        scope: Scope,
        session_id: str,
        entry: ForgetEntry | None,
        keep: Named | None = None,
    ) -> SessionUndoResult:
        """Bring back everything the conversation forget ``entry`` removed.

        The same reversal as a picture's, minus the episode *text* rewrite: a
        session forget marks the session's episodes rather than re-writing them
        (a picture never had to be stripped out of these bodies, because the
        bodies never named the conversation), so undoing it is exactly "drop
        those marks and the picture files' status".

        A conversation that was never forgotten answers ``applied=False`` and
        reports the state it found. Only the entries the forget marked come
        back, and of those not the ones another live forget names (``keep``).
        """
        keep = keep or Named()
        if entry is None:
            return SessionUndoResult(
                session_id=session_id,
                forget_id="",
                applied=False,
                episodes=(),
                image_memories=0,
                images=(),
            )

        restored_episodes, episode_files = await self._restore_session_episodes(
            scope,
            session_id,
            keep=keep.episodes,
            marked={old for old, _ in entry.episode_ids},
        )
        restored_images = await self._restore_session_images(
            scope, session_id, keep=keep.image_memories, marked=set(entry.entry_ids)
        )
        # A scene the forget took with its last episode comes back with it.
        stores = self._stores
        scenes = (
            await restore_revived_scenes(
                stores.episode, stores.scene, scope, keep=keep.scenes
            )
            if scope.owner_type == "user" and restored_episodes
            else []
        )
        images = tuple(
            dict.fromkeys(
                row.item.image_id for row in restored_images if row.item.image_id
            )
        )
        for image_id in images:
            await self._images.restore(image_id)

        await self._eraser.requeue(
            "image_memory", {row.ref.md_path for row in restored_images}
        )
        if episode_files:
            await self._eraser.requeue("episode", episode_files)
        if scenes:
            await self._eraser.requeue("scene", {s.ref.md_path for s in scenes})
        # Under the profile lock until the log says "remembered", as the forget.
        async with user_profile_lock(scope.app_id, scope.project_id, scope.owner_id):
            profiles = await self._restore_session_profile(scope, session_id, entry)
            await self._stores.forget_log.mark_undone(entry.forget_id)

        logger.info(
            "memory.session.undo_completed",
            session_id=session_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            forget_id=entry.forget_id,
            image_memories=len(restored_images),
            episodes=len(restored_episodes),
            profiles=profiles,
            scenes=len(scenes),
        )
        return SessionUndoResult(
            session_id=session_id,
            forget_id=entry.forget_id,
            applied=True,
            episodes=tuple(sorted(restored_episodes)),
            image_memories=len(restored_images),
            images=images,
            profiles=profiles,
        )

    async def _restore_session_episodes(
        self,
        scope: Scope,
        session_id: str,
        *,
        keep: set[str],
        marked: set[str],
    ) -> tuple[set[str], set[str]]:
        """Drop this conversation's forget mark from its episodes.

        Returns the entries marked current again and the files they are in:
        the files are what the cascade is handed to project once more (an
        entry id is no path, and queued as one the restored rows waited for
        the next full scan).

        Reads markdown: the forget erased the index rows through the fence, so
        the files are the only record of which entries carried the mark. An
        episode later superseded by a real replacement keeps its mark — only
        the forget's own ``forgotten`` marker is cleared, on the entries the
        log says it marked (every one of the session, for a log written
        before it said), and not on those in ``keep``.
        """
        entries = [
            (md_path, entry_id)
            for md_path, entry_id in await self._stores.episode.entries_for_session(
                scope, session_id
            )
            if entry_id not in keep and (not marked or entry_id in marked)
        ]
        by_file: dict[str, set[str]] = {}
        for md_path, entry_id in entries:
            by_file.setdefault(md_path, set()).add(entry_id)
        restored: set[str] = set()
        files: set[str] = set()
        for md_path, entry_ids in by_file.items():
            marks = await self._stores.episode.deprecations(scope, md_path)
            targets = [
                entry_id
                for entry_id in entry_ids
                if marks.get(entry_id) == FORGET_REASON
            ]
            if not targets:
                continue
            await self._stores.episode.clear_deprecations(scope, targets)
            restored |= set(targets)
            files.add(md_path)
        return restored, files

    async def _restore_session_images(
        self,
        scope: Scope,
        session_id: str,
        *,
        keep: set[str],
        marked: set[str],
    ) -> list[Stored[ImageMemory]]:
        """Mark current again the picture memories this conversation's forget retired.

        Driven by the file ids the session's entries name, then read back per
        file through :meth:`read_by_image`, which is the one reader that returns
        an occurrence together with its mark — a session forget has no per-file
        batch read, and inventing a second one would duplicate that logic.
        """
        entries = await self._stores.image_memory.entries_for_session(scope, session_id)
        wanted = {
            entry_id
            for _, entry_id, _ in entries
            if entry_id not in keep and (not marked or entry_id in marked)
        }
        restored: list[Stored[ImageMemory]] = []
        seen: set[str] = set()
        for image_id in dict.fromkeys(image for _, _, image in entries if image):
            for row in await self._stores.image_memory.read_by_image(scope, image_id):
                if row.ref.entry_id not in wanted or row.ref.entry_id in seen:
                    continue
                if row.item.superseded_by != FORGET_REASON:
                    continue
                seen.add(row.ref.entry_id)
                restored.append(row)
        if restored:
            await self._stores.image_memory.undeprecate(
                scope,
                entries=[(row.ref.md_path, row.ref.entry_id) for row in restored],
                only_if=_always,
            )
            for row in restored:
                row.item.superseded_by = None
                row.item.status = "current"
        return restored

    # ── steps ──────────────────────────────────────────────────────────────

    async def _restore_images(
        self,
        scope: Scope,
        entry: ForgetEntry,
        occurrences: list[Stored[ImageMemory]],
        *,
        keep: set[str],
    ) -> list[Stored[ImageMemory]]:
        """Mark current again the occurrences this forget retired, nothing else.

        Only entries the log recorded, and only while the mark still points at
        a forget (``forgotten`` or this entry's id): a later forget of the same
        picture is a different decision, and undoing an older one must not
        reverse it.
        """
        wanted = set(entry.entry_ids) - keep
        targets = [
            row
            for row in occurrences
            if row.ref.entry_id in wanted
            and row.item.superseded_by in (FORGET_REASON, entry.forget_id)
        ]
        if not targets:
            return []
        await self._stores.image_memory.undeprecate(
            scope,
            entries=[(row.ref.md_path, row.ref.entry_id) for row in targets],
            only_if=_always,
        )
        for row in targets:
            row.item.status = "current"
            row.item.superseded_by = None
        return targets

    async def _restore_episodes(
        self, scope: Scope, entry: ForgetEntry
    ) -> list[tuple[MemoryRef, Episode]]:
        """Append each pre-forget episode body again, superseding the rewrite."""
        if not entry.episode_ids:
            return []
        originals = await self._stores.episode.read_many(
            scope, [original for original, _ in entry.episode_ids]
        )
        by_id = {row.ref.entry_id: row for row in originals}
        pairs = [
            (by_id[original], replacement)
            for original, replacement in entry.episode_ids
            if original in by_id
        ]
        if not pairs:
            return []
        refs = await self._stores.episode.append_many(
            scope, [row.item for row, _ in pairs], only_if=_always
        )
        replacements = await self._stores.episode.read_many(
            scope, [replacement for _, replacement in pairs]
        )
        replacement_paths = {row.ref.entry_id: row.ref.md_path for row in replacements}
        written: list[tuple[MemoryRef, Episode]] = []
        for (original, replacement_id), ref in zip(pairs, refs, strict=True):
            md_path = replacement_paths.get(replacement_id)
            if md_path is None:
                # The rewrite is gone (deleted since); there is nothing left to
                # supersede, and the restored version already reads as current.
                written.append((ref, original.item))
                continue
            await self._stores.episode.deprecate(
                scope,
                entries=[(md_path, replacement_id)],
                replaced_by=ref.entry_id,
                only_if=_always,
            )
            written.append((ref, original.item))
        return written

    async def _restore_session_profile(
        self, scope: Scope, session_id: str, entry: ForgetEntry
    ) -> int:
        """Put back what the forget took from ``user.md`` and what it made others take.

        The forget's own items come back: each rests on a segment of this
        conversation, remembered again. So do the items another forget of the
        owner's, still live, took out while this one stood, that rest on one
        of this conversation's segments: not every segment they rest on is
        forgotten any more. The merge is by description, so an item the
        profile holds again is not doubled. The caller holds the profile lock.

        Returns the number of items; a replaced value is history, not an item.
        """
        removed = list(entry.removed_profile_items)
        mine = await segments_of(self._stores, scope, [session_id])
        removed += await removed_by_others(
            self._stores, scope, mine, except_id=entry.forget_id
        )
        if not removed:
            return 0
        existing = await self._stores.profile.read(scope)
        profile, restored, replaced = profile_items.restore_items(
            existing,
            removed,
            owner_id=scope.owner_id,
            timestamp=_newest_observed_at(tuple(removed)),
        )
        if not (restored or replaced):
            return 0
        await self._stores.profile.write(scope, profile)
        return restored

    async def _restore_profile(self, scope: Scope, entry: ForgetEntry) -> int:
        """Merge the items and replaced values the forget removed back into ``user.md``.

        Returns the number of items; a replaced value is history, not an item.
        """
        if not entry.removed_profile_items:
            return 0
        existing = await self._stores.profile.read(scope)
        profile, restored, replaced = profile_items.restore_items(
            existing,
            list(entry.removed_profile_items),
            owner_id=scope.owner_id,
            timestamp=_newest_observed_at(entry.removed_profile_items),
        )
        if not (restored or replaced):
            return 0
        await self._stores.profile.write(scope, profile)
        return restored


def _picture_of(stored: Stored[ImageMemory], image_id: str) -> ForgottenPicture:
    """One occurrence as the response's picture record."""
    return ForgottenPicture(
        entry_id=stored.ref.entry_id,
        image_id=image_id,
        name=stored.item.name,
    )


def _newest_observed_at(
    items: tuple[tuple[str, dict[str, object]], ...],
) -> int:
    """The newest ``observed_at`` of a forget's removed items.

    Only used when the profile had to be created by the undo (its owner had
    none), which in practice cannot happen — a profile item cannot be removed
    from a profile that does not exist — but must still have a value.
    """
    newest = 0
    for _bucket, item in items:
        value = item.get("observed_at")
        if isinstance(value, int):
            newest = max(newest, value)
    return newest
