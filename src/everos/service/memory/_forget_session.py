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
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from everos.core.observability.logging import get_logger
from everos.memory.ports import MemoryRef, Scope
from everos.service.memory._forget import FORGET_REASON, _always

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
    """

    session_id: str
    episodes: Sequence[str]
    image_memories: int
    images: Sequence[str]
    forget_id: str


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
        self, *, scope: Scope, session_id: str, reason: str | None = None
    ) -> SessionForgetResult:
        """Mark everything one conversation established for ``scope`` as forgotten."""
        episodes = await self._stores.episode.entries_for_session(scope, session_id)
        pictures = await self._stores.image_memory.entries_for_session(
            scope, session_id
        )
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

        erased: dict[str, list[MemoryRef]] = {}
        if episodes:
            erased["episode"] = [_ref(scope, path, entry) for path, entry in episodes]
        if pictures:
            erased["image_memory"] = [
                _ref(scope, path, entry) for path, entry, _ in pictures
            ]
        if erased:
            await self._eraser.erase(scope, erased)

        entry = await self._stores.forget_log.record(
            target="session",
            target_id=session_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            owner_type=scope.owner_type,
            reason=reason,
            entry_ids=tuple(item[1] for item in pictures),
            episode_ids=tuple((entry_id, "") for _, entry_id in episodes),
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
            reason=reason,
        )
        return SessionForgetResult(
            session_id=session_id,
            episodes=[entry_id for _, entry_id in episodes],
            image_memories=len(pictures),
            images=files,
            forget_id=entry.forget_id,
        )

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


def _ref(scope: Scope, md_path: str, entry_id: str) -> MemoryRef:
    """The index-row handle of one entry, keyed as the derived index keys it."""
    return MemoryRef(
        id=f"{scope.owner_id}_{entry_id}",
        entry_id=entry_id,
        md_path=md_path,
    )
