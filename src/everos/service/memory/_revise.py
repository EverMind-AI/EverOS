"""Revising a picture memory: a new version linked to the old one.

A description written from a glance is sometimes wrong — the tag says
"Biscuit", not "Biscuit's collar", the sign is a different branch. Correcting
it is a *change*, not a forget and not a second memory: the picture arrived
once, and two memories about it would compete with each other for the same
question.

So a revision writes a new markdown entry carrying the revised text and the
same picture, then marks the old entry superseded by the new one. The old text
stays on disk, which is what makes "what did this memory say before" — and an
undo of a revision, later — answerable. The cascade then recomputes the new
entry's keyword tokens and vector from the new text, because the entry is new
rather than edited; nothing recomputes anything by hand.

What a revision may touch is deliberately narrow: the description and the text
visible in the picture. The identity (content hash, path, dimensions, time,
provenance) belongs to the arrival and would be a different picture if it
changed — that is a forget plus a new add, not an edit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from everos.core.errors import NotFoundError
from everos.core.observability.logging import get_logger
from everos.memory.models import ImageMemory
from everos.memory.ports import Scope, Stored

if TYPE_CHECKING:
    from everos.memory.deps import Stores

logger = get_logger("everos.service.memory")


@dataclass(frozen=True)
class RevisionResult:
    """What one revision wrote.

    Attributes:
        image_id: The picture the memory is about.
        entry_id: The new version's entry id.
        supersedes: The entry id the new version replaced.
        description: The description as now stored.
        ocr: The in-picture text as now stored.
    """

    image_id: str
    entry_id: str
    supersedes: str
    description: str
    ocr: str


class RevisionManager:
    """Writes a superseding version of one picture memory.

    Args:
        stores: Write-side stores; markdown is the truth.
    """

    def __init__(self, *, stores: Stores) -> None:
        self._stores = stores

    async def revise_image(
        self,
        *,
        scope: Scope,
        image_id: str,
        description: str | None = None,
        ocr: str | None = None,
        entry_id: str | None = None,
    ) -> RevisionResult:
        """Write a revised version of one occurrence of ``image_id``.

        Args:
            scope: The owner whose memory this is.
            image_id: Which picture; its occurrences are read from markdown.
            description: The new description, or ``None`` to keep the stored one.
            ocr: The new in-picture text, or ``None`` to keep the stored one.
            entry_id: Which occurrence, when the picture has several. ``None``
                picks the current one — the only defensible default, since a
                superseded occurrence is history and two calls could otherwise
                each pick a different occurrence.

        Raises:
            NotFoundError: No current occurrence of this picture for this owner.
            ValueError: Both fields are ``None`` (nothing to change).
        """
        if description is None and ocr is None:
            raise ValueError("a revision needs a new description or a new ocr")
        occurrences = await self._stores.image_memory.read_by_image(scope, image_id)
        target = _pick(occurrences, entry_id=entry_id)
        if target is None:
            raise NotFoundError(
                f"no current picture memory for image {image_id!r}"
                + (f" and entry {entry_id!r}" if entry_id else "")
            )
        revised = target.item.model_copy(
            update={
                "description": (
                    target.item.description if description is None else description
                ),
                "ocr": target.item.ocr if ocr is None else ocr,
                "entry_id": None,
                "md_path": None,
                "status": "current",
                "superseded_by": None,
            }
        )
        ref = await self._stores.image_memory.append(scope, revised, only_if=_always)
        if ref is None:
            # The store only skips a write whose ``only_if`` failed, and this
            # one has none: unreachable short of a store bug.
            raise NotFoundError("the picture memory could not be written")
        await self._stores.image_memory.deprecate(
            scope,
            entries=[(target.ref.md_path, target.ref.entry_id)],
            replaced_by=ref.entry_id,
            only_if=_always,
        )
        logger.info(
            "memory.images.revised",
            image_id=image_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            supersedes=target.ref.entry_id,
            entry_id=ref.entry_id,
        )
        return RevisionResult(
            image_id=image_id,
            entry_id=ref.entry_id,
            supersedes=target.ref.entry_id,
            description=revised.description,
            ocr=revised.ocr,
        )


def _pick(
    occurrences: Sequence[Stored[ImageMemory]], *, entry_id: str | None
) -> Stored[ImageMemory] | None:
    """The occurrence to revise: the named one, else the current one."""
    for stored in occurrences:
        if entry_id is not None:
            if stored.ref.entry_id == entry_id:
                return stored
            continue
        if stored.item.superseded_by is None:
            return stored
    return None


async def _always() -> bool:
    """``only_if`` for a write whose source is the revision itself."""
    return True


__all__ = ["RevisionManager", "RevisionResult"]
