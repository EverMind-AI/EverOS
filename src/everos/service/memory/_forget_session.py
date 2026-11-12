"""Forgetting one conversation: its episodes and picture memories, reversibly.

A conversation is not a file to delete — it is a set of memories that came from
one session, and the design's rule for removing material is the same here as for
one picture: **mark, do not delete**. This marks every episode and every picture
memory that arrived in the session as forgotten, through the same forget log the
picture path writes, so an undo reverses it the same way.

The picture files are the one place with a reference rule rather than a mark:

- an original **stays current while any live picture memory still references
  it**, because the same picture may have arrived in another conversation;
- it is marked forgotten only when the last live reference to it goes.

That is decided from markdown, per file, with the entries this call marked
excluded by id. The SQLite reference count is not used: it counts a file's
*arrivals* and is released only by a forget of the file, so it does not answer
"does anything still point here".

Only the owner's side is forgotten. A conversation with two participants has one
set of episodes per participant, and one participant forgetting it must not take
the other's memory of it — the same separation a picture forget already keeps.

The profile (``user.md``) is reached through its items' sources: an item, or a
replaced value in its history, goes once every segment it rests on is forgotten,
this conversation's or one the owner forgot before. The forget log keeps what
went, for the undo, and the profile update never reads a segment of a forgotten
conversation.

The forget follows the episodes into the owner's scenes (docs/scenes.md): a
scene none of whose episodes is alive any more is marked forgotten too, and its
index row erased; a scene that still has a live episode stays.

Only the conversation's **current** versions are marked: a version a newer
one already replaced (a person's edit, a picture forget's rewrite) keeps the
mark that names its successor, so the chain of versions survives the forget
and its undo.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from everos.core.observability.logging import get_logger
from everos.memory import profile_lock
from everos.memory.extract.pipeline import profile_items
from everos.memory.extract.scene import forget_orphaned_scenes
from everos.memory.ports import MemoryRef, Scope
from everos.service.memory._forget import FORGET_REASON, _always
from everos.service.memory._forget_common import (
    forgotten_segments,
    removed_item_ids,
    segments_of,
)

if TYPE_CHECKING:
    from everos.memory.deps import Stores
    from everos.memory.ports import IndexEraser
    from everos.memory.ports.images import ImageStore

logger = get_logger("everos.service.memory")


@dataclass(frozen=True)
class SessionForgetResult:
    """What forgetting one conversation touched.

    Attributes:
        session_id: The conversation.
        episodes: Entry ids of the episode versions marked forgotten.
        image_memories: Picture-memory entries marked forgotten.
        images: Picture *files* taken out of circulation, because this session
            held their last live reference.
        forget_id: The log entry written (``fg_<hex>``), which an undo of the
            same session reverses.
        profiles: Profile items taken out of ``user.md`` because every segment
            they rest on is now forgotten.
        scenes: Entry ids of the scene versions marked forgotten because none
            of their episodes is alive any more.
        pictures: Entry ids of the picture memories marked forgotten.
        profile_items: Ids of the profile items and replaced values taken out.
    """

    session_id: str
    episodes: Sequence[str]
    image_memories: int
    images: Sequence[str]
    forget_id: str
    profiles: int = 0
    scenes: Sequence[str] = ()
    pictures: Sequence[str] = ()
    profile_items: Sequence[str] = ()


class SessionForgetManager:
    """Marks one conversation's memories forgotten, and the files it alone held.

    Args:
        stores: Write-side stores; markdown and the forget log are the truth.
        images: The picture files, whose status this changes.
        eraser: Removes the marked entries' index rows at once.
    """

    def __init__(
        self, *, stores: Stores, images: ImageStore, eraser: IndexEraser
    ) -> None:
        self._stores = stores
        self._images = images
        self._eraser = eraser

    async def forget_session(
        self,
        *,
        scope: Scope,
        session_id: str,
        reason: str | None = None,
        requested_by: str | None = None,
    ) -> SessionForgetResult:
        """Mark everything one conversation established for ``scope`` as forgotten."""
        episodes = await current_episodes_of(self._stores, scope, session_id)
        pictures = await current_pictures_of(self._stores, scope, session_id)
        if episodes:
            await self._stores.episode.deprecate(
                scope,
                entries=episodes,
                replaced_by=FORGET_REASON,
                only_if=_always,
            )
        if pictures:
            await self._stores.image_memory.deprecate(
                scope,
                entries=[(md_path, entry_id) for md_path, entry_id, _ in pictures],
                replaced_by=FORGET_REASON,
                only_if=_always,
            )
        files = await self._release_files(scope, pictures)
        stores = self._stores
        scenes = (
            await forget_orphaned_scenes(stores.episode, stores.scene, scope)
            if scope.owner_type == "user" and episodes
            else []
        )

        erased: dict[str, list[MemoryRef]] = {}
        if episodes or scenes:
            # A scene version is a row of the episode table.
            erased["episode"] = [
                *(_ref(scope, path, entry) for path, entry in episodes),
                *(scene.ref for scene in scenes),
            ]
        if pictures:
            erased["image_memory"] = [
                _ref(scope, path, entry) for path, entry, _ in pictures
            ]
        if erased:
            await self._eraser.erase(scope, erased)

        # The profile lock is held until the log says "forgotten": an update
        # waiting on it then reads none of the conversation, and none in
        # flight writes back what this takes out.
        async with profile_lock(scope):
            profile_removed = await self._forget_profile(scope, session_id)
            entry = await self._stores.forget_log.record(
                target="session",
                target_id=session_id,
                app_id=scope.app_id,
                project_id=scope.project_id,
                owner_id=scope.owner_id,
                owner_type=scope.owner_type,
                reason=reason,
                requested_by=requested_by,
                entry_ids=tuple(item[1] for item in pictures),
                episode_ids=tuple((entry_id, "") for _, entry_id in episodes),
                removed_profile_items=profile_removed,
            )
        profiles = sum(
            1 for bucket, _ in profile_removed if bucket in profile_items.BUCKETS
        )
        logger.info(
            "memory.session.forgotten",
            session_id=session_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            episodes=len(episodes),
            image_memories=len(pictures),
            images=len(files),
            profiles=profiles,
            scenes=len(scenes),
            reason=reason,
        )
        return SessionForgetResult(
            session_id=session_id,
            episodes=[entry_id for _, entry_id in episodes],
            image_memories=len(pictures),
            images=files,
            forget_id=entry.forget_id,
            profiles=profiles,
            scenes=[scene.ref.entry_id for scene in scenes],
            pictures=[entry_id for _, entry_id, _ in pictures],
            profile_items=removed_item_ids(profile_removed),
        )

    async def _forget_profile(
        self, scope: Scope, session_id: str
    ) -> tuple[tuple[str, dict[str, object]], ...]:
        """Take out of ``user.md`` what no remembered segment supports any more.

        An item, or a replaced value in the history, goes once every segment it
        rests on is forgotten -- this conversation's, or one the owner forgot
        before and no undo reversed -- and at least one is this conversation's
        (:func:`profile_items.forget_segments`). An item a remembered segment
        still supports stays, and so does one stored before items recorded
        their segments. The caller holds the profile lock and logs what this
        returns, for the undo.
        """
        existing = await self._stores.profile.read(scope)
        if existing is None:
            return ()
        next_profile, removed, count = profile_items.forget_segments(
            existing,
            await segments_of(self._stores, scope, [session_id]),
            already=await forgotten_segments(
                self._stores, scope, except_session=session_id
            ),
        )
        if not removed or next_profile is None:
            return ()
        await self._stores.profile.write(scope, next_profile)
        logger.info(
            "memory.session.profile_forgotten",
            session_id=session_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            removed=count,
            history=sum(1 for bucket, _ in removed if bucket == "history"),
        )
        return tuple(removed)

    async def _release_files(
        self, scope: Scope, marked: Sequence[tuple[str, str, str]]
    ) -> list[str]:
        """Forget each original whose last live reference this session held.

        ``marked`` is the set this call retired; those entries are excluded by
        id rather than by re-reading the frontmatter mark, because the mark is
        written per file and a later entry of the same file may already hold a
        different value in the copy this pass read.
        """
        gone = {entry_id for _, entry_id, _ in marked}
        released: list[str] = []
        for image_id in dict.fromkeys(image for _, _, image in marked if image):
            occurrences = await self._stores.image_memory.read_by_image(scope, image_id)
            live = [
                row
                for row in occurrences
                if row.item.superseded_by is None and row.ref.entry_id not in gone
            ]
            if live:
                continue
            await self._images.forget(image_id)
            released.append(image_id)
        return released


async def current_episodes_of(
    stores: Stores, scope: Scope, session_id: str
) -> list[tuple[str, str]]:
    """``(md_path, entry_id)`` of the conversation's episode versions no mark
    retires: a superseded one keeps the mark that names its successor."""
    entries = await stores.episode.entries_for_session(scope, session_id)
    marks: dict[str, dict[str, str]] = {}
    for md_path in {md_path for md_path, _ in entries}:
        marks[md_path] = await stores.episode.deprecations(scope, md_path)
    return [(md, e) for md, e in entries if not marks[md].get(e)]


async def current_pictures_of(
    stores: Stores, scope: Scope, session_id: str
) -> list[tuple[str, str, str]]:
    """``(md_path, entry_id, image_id)`` of the conversation's current picture
    memories."""
    entries = await stores.image_memory.entries_for_session(scope, session_id)
    rows = await stores.image_memory.read_many(scope, [e for _, e, _ in entries])
    marked = {row.ref.entry_id for row in rows if row.item.superseded_by}
    return [row for row in entries if row[1] not in marked]


def _ref(scope: Scope, md_path: str, entry_id: str) -> MemoryRef:
    """The index-row handle of one entry, keyed as the derived index keys it."""
    return MemoryRef(
        id=f"{scope.owner_id}_{entry_id}",
        entry_id=entry_id,
        md_path=md_path,
    )
