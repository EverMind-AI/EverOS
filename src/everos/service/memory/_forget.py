"""Forgetting one picture: the use case behind ``POST /memory/forget``.

A picture is not one memory — it is a file, one or more memories written about
it, their index rows, and the episode text that referred to it by id.
Forgetting it has to reach all four, or the content survives somewhere:

1. **The file** keeps its bytes but gains ``forgotten_at`` (SQLite). It is not
   unlinked: a forget is reversible, and undoing it must not need a restore.
2. **Every memory written about it** is marked forgotten — a ``deprecated_by``
   marker pointing at the forget, the same append-only mechanism the cascade
   already turns into the index row's status.
3. **The index rows** are erased through the fence, so the cascade cannot write
   them back and the picture stops competing for retrieval immediately rather
   than one scan interval later.
4. **What referred to it** is re-extracted without it. An episode whose body
   says ``[IMAGE: <id>]`` is written again as a new version whose body drops
   that reference, and the new version supersedes the old one. The old version
   is **not** deleted: the store keeps superseded memories so "what did I say
   before" stays answerable, and the forget marks are what keep them out of a
   default result.

Why re-extraction is deterministic here, with no model call: the picture's
contribution to an episode body is a *reference*, not prose. The picture
prompt forbids restating a picture's description in the body (the picture has
its own memory for that), so removing the reference is a text edit that needs
no model — and a model call would risk rewriting facts the user never asked to
change, which is exactly what a forget must not do.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from everos.core.observability.logging import get_logger
from everos.memory.extract.pipeline import profile_delta
from everos.memory.models import Episode
from everos.memory.ports import Filter, MemoryRef, Scope, Stored, all_of, eq, is_null

if TYPE_CHECKING:
    from everos.memory.deps import Indexes, Stores
    from everos.memory.ports import IndexEraser
    from everos.memory.ports.images import ImageStore

logger = get_logger("everos.service.memory")

FORGET_REASON = "forgotten"
"""The ``deprecated_by`` marker a forget writes.

Deliberately not the entry id of a successor: a superseding *memory* and a
*forget* are different states, and collapsing them would make "what replaced
this" unanswerable for a forgotten picture.
"""

_EPISODE_SCAN_CAP = 5_000
"""Episodes one forget reads to find the ones referring to the picture.

A cap rather than the whole history: an owner with a very long history should
not make one forget call unbounded. Entries past it keep their reference and
the caller sees fewer rewritten episodes than the picture had.
"""

_OCCURRENCE_CAP = 1_000
"""Picture memories one forget marks. One picture is normally one to a handful."""


@dataclass(frozen=True)
class ForgottenPicture:
    """One picture memory a forget (or its undo) touched.

    Returned so a caller can see *which* memories the picture had, not just
    how many: a harness mapping a memory back to its dataset entry needs the
    occurrence's own identity and the name it arrived under.

    Attributes:
        entry_id: ``im_<YYYYMMDD>_<NNNN>`` — this occurrence's id in markdown.
        image_id: The file, by content hash.
        name: The client's file name for this arrival, ``""`` when it sent none.
    """

    entry_id: str
    image_id: str
    name: str


@dataclass(frozen=True)
class ForgetResult:
    """What one forget touched.

    Attributes:
        image_id: The picture, by content hash.
        image_memories: How many picture memories were marked forgotten.
        episodes: Entry ids of the episode versions written without the
            picture. The superseded versions are still on disk.
        profiles: How many profile items were removed because this picture
            alone established them. ``user.md`` is one document, so what a
            forget reports is items, not versions.
        pictures: One entry per picture memory touched, in entry order.
        forget_id: The forget log entry this call wrote (``fg_<hex>``), or
            ``""`` when the log store is not configured. An explicit undo can
            name it; the default undo finds it by target.
    """

    image_id: str
    image_memories: int
    episodes: Sequence[str]
    profiles: int
    pictures: Sequence[ForgottenPicture] = ()
    forget_id: str = ""


class ForgetManager:
    """Marks a picture and everything that referred to it as forgotten.

    Args:
        stores: Write-side stores (markdown is the truth).
        indexes: Read-side indexes; used to find what refers to the picture.
        images: The picture files, whose metadata row records the forget.
        eraser: Removes the index rows at once instead of waiting for the
            cascade, and fences them against it writing them back.
    """

    def __init__(
        self,
        *,
        stores: Stores,
        indexes: Indexes,
        images: ImageStore,
        eraser: IndexEraser,
    ) -> None:
        self._stores = stores
        self._indexes = indexes
        self._images = images
        self._eraser = eraser

    async def forget_image(
        self, *, scope: Scope, image_id: str, reason: str | None = None
    ) -> ForgetResult:
        """Forget ``image_id`` for ``scope``'s owner.

        Steps run in the order the module docstring lists, and each is safe to
        repeat: the marks are idempotent and a re-written version is a new
        entry, so a retry after a crash finishes the job rather than doubling
        it.
        """
        occurrences = await self._indexes.image_memory.find(
            _by_image(scope, image_id), limit=_OCCURRENCE_CAP
        )
        await self._mark_memories(scope, [hit.stored for hit in occurrences])
        await self._images.forget(image_id)
        await self._eraser.erase(
            scope, {"image_memory": [hit.stored.ref for hit in occurrences]}
        )

        referring = await self._referring_episodes(scope, image_id)
        written = await self._rewrite_episodes(scope, referring, image_id)
        profile_removed = await self._forget_profile(scope, image_id)
        profiles = len(profile_removed)
        entry = await self._stores.forget_log.record(
            target="image",
            target_id=image_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            owner_type=scope.owner_type,
            reason=reason,
            entry_ids=tuple(hit.stored.ref.entry_id for hit in occurrences),
            episode_ids=tuple(
                (stored.ref.entry_id, ref.entry_id)
                for stored, ref in zip(referring, written, strict=False)
            ),
            removed_profile_items=profile_removed,
        )

        logger.info(
            "memory.images.forgotten",
            image_id=image_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            occurrences=len(occurrences),
            episodes=len(written),
            profiles=profiles,
            reason=reason,
        )
        return ForgetResult(
            image_id=image_id,
            image_memories=len(occurrences),
            episodes=[ref.entry_id for ref in written],
            profiles=profiles,
            pictures=[
                ForgottenPicture(
                    entry_id=hit.stored.ref.entry_id,
                    image_id=image_id,
                    name=getattr(hit.stored.item, "name", "") or "",
                )
                for hit in occurrences
            ],
            forget_id=entry.forget_id,
        )

    # ── steps ──────────────────────────────────────────────────────────────

    async def _mark_memories(
        self, scope: Scope, memories: Sequence[Stored[object]]
    ) -> int:
        """Set ``deprecated_by`` on every picture memory about ``image_id``.

        The marker is always :data:`FORGET_REASON`, never the caller's free-text
        reason: ``deprecated_by`` answers "what replaced this", and a reason
        string in that column would be indistinguishable from a successor's
        entry id — the reader would look for an entry that does not exist. The
        caller's reason belongs in the log line, where it is.

        One call for all occurrences: the store groups them by file, so a day
        with ten pictures in it costs one rewrite rather than ten.
        """
        entries = [(stored.ref.md_path, stored.ref.entry_id) for stored in memories]
        if not entries:
            return 0
        await self._stores.image_memory.deprecate(
            scope,
            entries=entries,
            replaced_by=FORGET_REASON,
            only_if=_always,
        )
        return len(entries)

    async def _forget_profile(
        self, scope: Scope, image_id: str
    ) -> tuple[tuple[str, dict[str, object]], ...]:
        """Take the profile items that rested on this picture alone out of ``user.md``.

        A profile item records the pictures that established it. The item goes
        only when every one of them is forgotten (see
        :func:`profile_delta.forget_images`); a fact a photo and a sentence
        both established keeps the sentence's ground and stays. An item with no
        picture source is never touched, which is why a forget cannot quietly
        erase a text-derived fact.

        The profile document is rewritten rather than versioned: ``user.md`` is
        everalgo's output shape, the whole profile is replaced by every
        synthesis, and there is no per-item markdown log to append to. What
        makes the removal reversible is the caller's undo, which re-merges the
        items this returned — so the item list is logged here at info level and
        the caller's own response carries the count.
        """
        existing = await self._stores.profile.read(scope)
        next_profile, removed, count = profile_delta.forget_images(existing, [image_id])
        if not count or next_profile is None:
            return ()
        await self._stores.profile.write(scope, next_profile)
        logger.info(
            "memory.images.profile_forgotten",
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            image_id=image_id,
            removed=count,
            buckets=[bucket for bucket, _ in removed],
        )
        return tuple((bucket, item.model_dump(mode="json")) for bucket, item in removed)

    async def _referring_episodes(
        self, scope: Scope, image_id: str
    ) -> list[Stored[Episode]]:
        """The owner's current episodes whose body references ``image_id``.

        The index has no full-text predicate for "mentions this id", so the
        owner's current episodes are read and filtered in process. That is
        linear in the owner's episode count, which is what the scan cap bounds.
        """
        hits = await self._indexes.episode.find(
            _current(scope), limit=_EPISODE_SCAN_CAP
        )
        # Match on the *id*, not on one spelling of the reference: the picture
        # extractor writes ``(img-1: <id>)`` and the text route writes
        # ``[IMAGE: <id>]``, and both are the same picture. The hash is 64 hex
        # characters of an identity, so a substring hit is a reference, not a
        # coincidence.
        return [
            hit.stored
            for hit in hits
            if image_id in hit.stored.item.episode
            or image_id in hit.stored.item.summary
        ]

    async def _rewrite_episodes(
        self, scope: Scope, referring: Sequence[Stored[Episode]], image_id: str
    ) -> list[MemoryRef]:
        """Write one superseding version per referring episode, without the picture.

        The old version stays in markdown and its index row stays too -- marked
        ``deprecated_by`` the new entry -- because the store keeps superseded
        memories on purpose. What changes is which one a default search returns.
        """
        rewrites = [
            stored.item.model_copy(
                update={
                    "episode": _strip_reference(stored.item.episode, image_id),
                    "summary": _strip_reference(stored.item.summary, image_id),
                    # The reference list follows the body: a version that no
                    # longer names the picture must not claim to refer to it,
                    # or a reader would try to resolve a picture the episode
                    # was deliberately rewritten without.
                    "image_ids": [
                        kept for kept in stored.item.image_ids if kept != image_id
                    ],
                }
            )
            for stored in referring
        ]
        if not rewrites:
            return []
        refs = await self._stores.episode.append_many(scope, rewrites, only_if=_always)
        # One deprecation call per old entry: each old version is superseded by
        # its own new one, not by a single shared entry, so the chain of
        # versions stays readable.
        for stored, ref in zip(referring, refs, strict=True):
            await self._stores.episode.deprecate(
                scope,
                entries=[(stored.ref.md_path, stored.ref.entry_id)],
                replaced_by=ref.entry_id,
                only_if=_always,
            )
        return refs


def _strip_reference(text: str, image_id: str) -> str:
    """Remove every reference to ``image_id`` and the empty brackets it leaves.

    The two spellings in use are ``(img-1: <id>)`` from the picture extractor
    and ``[IMAGE: <id>]`` from a text-only render. Stripping only the id would
    leave ``(img-1: )`` or ``[IMAGE: ]`` inside a sentence, so each shape is
    removed whole, outermost first.
    """
    if image_id not in text:
        return text
    cleaned = text.replace(f"[IMAGE: {image_id}]", "")
    for span in _parenthesised(text):
        if image_id in span:
            cleaned = cleaned.replace(span, "")
    # A reference that stood alone in its own parentheses leaves ``()``; drop
    # the empty pair and the space that separated it, so the sentence reads as
    # if the picture were never named. The same pass closes up the space the
    # removed reference left before the punctuation that followed it
    # ("the signboard ." -> "the signboard."), which is otherwise visible to
    # anyone reading the rewritten episode.
    spaced = " ".join(cleaned.replace("()", "").split())
    return _before_punctuation.sub(r"\1", spaced)


_before_punctuation = re.compile(r"\s+([.,;:!?])")
"""A space immediately before sentence punctuation, left by a removal."""


def _parenthesised(text: str) -> list[str]:
    """Every ``(...)`` span in ``text``, outermost and balanced.

    A plain scan, not a regular expression: references nest nothing today, but
    a description may itself contain parentheses, and a greedy match would eat
    from the first ``(`` in the paragraph to the reference's ``)``.
    """
    spans: list[str] = []
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
                spans.append(text[start : index + 1])
    return spans


async def _always() -> bool:
    """``only_if`` for a write with no deletable source: the forget is the source."""
    return True


def _by_image(scope: Scope, image_id: str) -> Filter:
    """The owner's picture memories about one file."""
    return all_of(
        eq("owner_id", scope.owner_id),
        eq("app_id", scope.app_id),
        eq("project_id", scope.project_id),
        eq("image_id", image_id),
    )


def _current(scope: Scope) -> Filter:
    """The owner's live episodes, across every session."""
    return all_of(
        eq("owner_id", scope.owner_id),
        eq("owner_type", scope.owner_type),
        eq("app_id", scope.app_id),
        eq("project_id", scope.project_id),
        is_null("deprecated_by"),
    )
