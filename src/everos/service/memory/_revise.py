"""Revising a picture memory: a new version linked to the old one.

A description written from a glance is sometimes wrong — the tag says
"Biscuit", not "Biscuit's collar", the sign is a different branch. Correcting
it is a *change*, not a forget and not a second memory: the picture arrived
once, and two memories about it would compete with each other for the same
question.

So a revision writes a new markdown entry carrying the revised text and the
same picture, then marks the old entry superseded by the new one -- the
general ``POST /memory/update`` (:mod:`._update`) does it; this module keeps
the picture-shaped body. The old text stays on disk, which is what makes "what
did this memory say before" answerable (``GET /memory/history``). The cascade
then recomputes the new entry's keyword tokens and vector from the new text,
because the entry is new rather than edited; nothing recomputes anything by
hand.

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
from everos.memory.access import Identity
from everos.memory.models import ImageMemory
from everos.memory.ports import Scope, Stored

from ._records import entry_id_of
from ._update import MemoryChanges, UpdateManager

if TYPE_CHECKING:
    from everos.memory.deps import Stores


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
    """Revises one occurrence of a picture: ``POST /memory/images/revise``.

    A thin alias of ``POST /memory/update`` for a picture memory, kept for the
    body it takes (a picture's hash, optionally which occurrence): it picks
    the occurrence, and the update writes the new version -- a person's edit,
    held, refused with 409 when the occurrence was replaced or forgotten.

    Args:
        stores: Write-side stores; markdown is the truth.
        updater: Writes the new version.
    """

    def __init__(self, *, stores: Stores, updater: UpdateManager) -> None:
        self._stores = stores
        self._updater = updater

    async def revise_image(
        self,
        *,
        scope: Scope,
        image_id: str,
        description: str | None = None,
        ocr: str | None = None,
        entry_id: str | None = None,
        identity: Identity | None = None,
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
            identity: Who asks; the owner when nobody was asserted.

        Raises:
            NotFoundError: No current occurrence of this picture for this owner.
            ValueError: Both fields are ``None`` (nothing to change).
            VersionConflictError: The named occurrence was replaced (409).
            ForgottenMemoryError: The named occurrence is forgotten (409).
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
        result = await self._updater.update(
            scope=scope,
            identity=identity or Identity(),
            record_id=target.ref.entry_id,
            changes=MemoryChanges(description=description, ocr=ocr),
        )
        return RevisionResult(
            image_id=image_id,
            entry_id=entry_id_of(scope.owner_id, result.id),
            supersedes=target.ref.entry_id,
            description=target.item.description if description is None else description,
            ocr=target.item.ocr if ocr is None else ocr,
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


__all__ = ["RevisionManager", "RevisionResult"]
