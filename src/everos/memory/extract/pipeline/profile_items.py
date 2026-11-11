"""Profile items: provenance on every item, versions linked, replaced items kept.

The one writer of both profiles -- the one-to-one ``user.md`` and a member's
profile in a group (docs/group-memory.md, section 7.4) -- is everalgo's
``ProfileExtractor.aupdate``. Its result says how every item came to be and
which stored items it took out (the lineage), and :func:`stamp_update` turns
that into what EverOS keeps on the items:

* every item carries ``item_id``, the segments it was read from
  (``source_memcell_ids``) and their span (``source_start`` / ``source_end``,
  epoch ms), and, read off a picture, the pictures (:data:`SOURCE_IMAGE_KEYS`).
  A kept item keeps its own; an added one gets the segments its operation
  named; fresh evidence for an item adds its segments to the item's; an item
  a consolidation wrote takes the union of the items it came from;
* a fact (``explicit_info``) given a new value by a ``modify`` is a new item
  that names the old one (``supersedes``) and the time of the update
  (``since``); the old one's history entry names the new one
  (``superseded_by``). everalgo applies a ``modify`` in place, so the item
  still carries the old ``item_id``: that, not a comparison of texts, is the
  link. A rewrite that only re-punctuates is not a new value, and an implicit
  trait is an inferred pattern, not a stated value: neither is linked;
* every stored item the update took out moves to the history with the time of
  the update (``replaced_at``) and what replaced it: ``superseded_by`` (a new
  value), ``deprecated_by`` and ``reason`` (the segments that showed it no
  longer true), ``merged_into`` (the items a consolidation merged it into) or
  ``reason`` alone (a consolidation dropped it). :func:`bounded_history` keeps
  :data:`HISTORY_LIMIT` entries.

A forgotten picture reaches the items read off it alone (:func:`forget_images`),
and an undo puts them back (:func:`restore_items`).
"""

from __future__ import annotations

__all__ = [
    "HISTORY_LIMIT",
    "SOURCE_IMAGE_KEYS",
    "bounded_history",
    "forget_images",
    "item_source_images",
    "mint_item_id",
    "normalised",
    "restore_items",
    "stamp_update",
]

import re
import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from everos.memory.models import (
    MEMBER_VERSION_KEYS,
    ExplicitInfoItem,
    ImplicitTraitItem,
    MemCell,
    Profile,
    replaced_chains,
)

if TYPE_CHECKING:
    from everalgo.user_memory.profile import ItemOrigin, ProfileUpdate, RetiredItem

HISTORY_LIMIT = 200
"""Replaced items a profile keeps; past it the oldest drop off first, except
the values a current item still shows as replaced."""

SOURCE_IMAGE_KEYS = "source_image_ids"
"""The key an item read off pictures carries: the content hashes of those
pictures. A list, because two photos of the same cat establish one fact between
them, and the fact survives forgetting either one. An item with no picture
source is never touched by a picture forget."""

_PROVENANCE = (
    "item_id",
    "source_memcell_ids",
    "source_start",
    "source_end",
    SOURCE_IMAGE_KEYS,
    *MEMBER_VERSION_KEYS,
)
"""The keys :func:`stamp_update` owns on an item."""

_NOT_WORDS = re.compile(r"[\W_]+")
"""What a rewrite that only re-punctuates a description changes."""

_Item = ExplicitInfoItem | ImplicitTraitItem


def mint_item_id() -> str:
    """A fresh, stable id for a new profile item (``pi_<12 hex>``)."""
    return f"pi_{uuid.uuid4().hex[:12]}"


def normalised(text: str) -> str:
    """The identity of a profile item: its description, case- and space-folded."""
    return " ".join(text.split()).casefold()


def item_source_images(item: BaseModel | Mapping[str, Any]) -> list[str]:
    """The picture hashes an item was read from; empty for a text-derived one."""
    extra = item if isinstance(item, Mapping) else (item.model_extra or {})
    raw = extra.get(SOURCE_IMAGE_KEYS)
    return [str(value) for value in raw] if isinstance(raw, list) else []


def stamp_update(
    old: Profile | None,
    update: ProfileUpdate,
    *,
    cells: Mapping[str, MemCell],
    replaced_at: int,
    primary_language: str | None,
) -> tuple[Profile, list[dict[str, Any]]]:
    """The profile ``update`` describes, every item stamped, and its history entries.

    Args:
        old: The profile the update was given, as stored (``None`` for none).
        update: What ``ProfileExtractor.aupdate`` returned for it.
        cells: The segments the update read, by memcell id: an item's span is
            read off its own segments.
        replaced_at: Epoch ms of the update -- the end of what it read.
        primary_language: The language the profile is frozen to.

    Returns:
        ``(profile, replaced)`` -- ``replaced`` are the history entries of the
        stored items the update took out, newest first.
    """
    stamper = _Stamper(old, update, cells=cells, replaced_at=replaced_at)
    algo = update.profile
    profile = Profile(
        owner_id=algo.owner_id,
        summary=algo.summary,
        timestamp=algo.timestamp,
        explicit_info=[
            ExplicitInfoItem.model_validate(
                stamper.stamp("explicit_info", i, item, origin)
            )
            for i, (item, origin) in enumerate(
                zip(algo.explicit_info, update.explicit_info, strict=True)
            )
        ],
        implicit_traits=[
            ImplicitTraitItem.model_validate(
                stamper.stamp("implicit_traits", i, item, origin)
            )
            for i, (item, origin) in enumerate(
                zip(algo.implicit_traits, update.implicit_traits, strict=True)
            )
        ],
        primary_language=primary_language,
    )
    return profile, stamper.history()


class _Stamper:
    """Provenance and links for each item of one update's result."""

    def __init__(
        self,
        old: Profile | None,
        update: ProfileUpdate,
        *,
        cells: Mapping[str, MemCell],
        replaced_at: int,
    ) -> None:
        self._old: dict[str, list[_Item]] = {
            "explicit_info": list(old.explicit_info) if old else [],
            "implicit_traits": list(old.implicit_traits) if old else [],
        }
        self._cells = cells
        self._replaced_at = replaced_at
        self._retired = update.retired
        # The value-changing modify that put each item where it is.
        self._new_value: dict[tuple[str, int], RetiredItem] = {
            place: retired
            for retired in update.retired
            if retired.change == "modified"
            for place in retired.replaced_by
        }
        self._ids: dict[tuple[str, int], str] = {}
        self._successors: dict[str, str] = {}

    def stamp(
        self, bucket: str, index: int, item: BaseModel, origin: ItemOrigin
    ) -> dict[str, Any]:
        """The item's fields and extras with its provenance set."""
        fields = item.model_dump()
        extra = dict(item.model_extra or {})
        if origin.change == "kept" or (
            origin.change == "modified" and (bucket, index) not in self._new_value
        ):
            stamped = self._kept(extra, origin)
        elif origin.change == "modified":
            stamped = self._new_value_of(bucket, index, fields, extra, origin)
        elif origin.change == "consolidated":
            stamped = self._consolidated(extra, origin)
        else:
            stamped = self._fresh(extra, origin.sources, origin.pictures)
        self._ids[bucket, index] = str(stamped["item_id"])
        return {**fields, **stamped}

    def history(self) -> list[dict[str, Any]]:
        """The history entries of the retired items, newest first, links resolved."""
        entries: list[dict[str, Any]] = []
        for retired in reversed(self._retired):
            entry = {**retired.item.model_dump(), "replaced_at": self._replaced_at}
            old_id = str((retired.item.model_extra or {}).get("item_id") or "")
            if retired.change == "modified" and old_id in self._successors:
                entry["superseded_by"] = self._successors[old_id]
            elif retired.change == "deprecated":
                entry |= {
                    "deprecated_by": list(retired.sources),
                    "reason": retired.reason,
                }
            elif retired.change == "consolidated" and retired.replaced_by:
                entry["merged_into"] = [
                    self._ids[place]
                    for place in retired.replaced_by
                    if place in self._ids
                ]
            elif retired.change == "consolidated":
                entry["reason"] = retired.reason
            entries.append(entry)
        return entries

    def _kept(self, extra: dict[str, Any], origin: ItemOrigin) -> dict[str, Any]:
        """A stored item: its own provenance, widened by fresh evidence's segments."""
        stamped = {**extra}
        if not stamped.get("item_id"):
            stamped["item_id"] = mint_item_id()  # stored before items had ids
        if origin.sources:
            sources = _merged(extra.get("source_memcell_ids"), origin.sources)
            stamped |= {"source_memcell_ids": sources, **self._span(sources, extra)}
        if origin.pictures:
            stamped[SOURCE_IMAGE_KEYS] = _merged(
                extra.get(SOURCE_IMAGE_KEYS), origin.pictures
            )
        return stamped

    def _new_value_of(
        self,
        bucket: str,
        index: int,
        fields: Mapping[str, Any],
        extra: dict[str, Any],
        origin: ItemOrigin,
    ) -> dict[str, Any]:
        """A new value: a new item, linked to the old one when it is a stated fact."""
        retired = self._new_value[bucket, index]
        old_id = str(extra.get("item_id") or "")
        stamped = self._fresh(extra, origin.sources, origin.pictures)
        if (
            bucket == "explicit_info"
            and old_id
            and not _same_words(
                retired.item.description, str(fields.get("description") or "")
            )
        ):
            stamped |= {"supersedes": old_id, "since": self._replaced_at}
            self._successors[old_id] = str(stamped["item_id"])
        return stamped

    def _consolidated(
        self, extra: dict[str, Any], origin: ItemOrigin
    ) -> dict[str, Any]:
        """A consolidated item: the union of the provenance of what it came from."""
        came_from = [self._old[bucket][index] for bucket, index in origin.from_items]
        stored = [item.model_extra or {} for item in came_from]
        sources = _merged(
            *(e.get("source_memcell_ids") for e in stored), origin.sources
        )
        stamped = {k: v for k, v in extra.items() if k not in _PROVENANCE}
        stamped |= {"item_id": mint_item_id(), "source_memcell_ids": sources}
        stamped |= self._span(sources, *stored)
        # Read off pictures only when every part of it was.
        if stored and all(item_source_images(e) for e in stored):
            stamped[SOURCE_IMAGE_KEYS] = _merged(
                *(item_source_images(e) for e in stored), origin.pictures
            )
        elif not stored and origin.pictures:
            stamped[SOURCE_IMAGE_KEYS] = list(origin.pictures)
        return stamped

    def _fresh(
        self, extra: dict[str, Any], sources: Sequence[str], pictures: Sequence[str]
    ) -> dict[str, Any]:
        """A new item's provenance: a fresh id, the segments and pictures behind it."""
        stamped = {k: v for k, v in extra.items() if k not in _PROVENANCE}
        stamped |= {
            "item_id": mint_item_id(),
            "source_memcell_ids": list(sources),
            **self._span(sources),
        }
        if pictures:
            stamped[SOURCE_IMAGE_KEYS] = list(pictures)
        return stamped

    def _span(
        self, sources: Sequence[str], *stored: Mapping[str, Any]
    ) -> dict[str, int]:
        """The span of ``sources`` among the segments read, widened by ``stored``'s."""
        starts = [_first_message(self._cells[s]) for s in sources if s in self._cells]
        ends = [self._cells[s].timestamp for s in sources if s in self._cells]
        for extra in stored:
            if isinstance(extra.get("source_start"), int) and isinstance(
                extra.get("source_end"), int
            ):
                starts.append(extra["source_start"])
                ends.append(extra["source_end"])
        return {"source_start": min(starts), "source_end": max(ends)} if starts else {}


def bounded_history(
    replaced: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    profile: Profile,
    *,
    limit: int = HISTORY_LIMIT,
) -> list[dict[str, Any]]:
    """The history after an update: ``replaced`` before ``history``, ``limit`` kept.

    The oldest entries drop off first, except the values an item of
    ``profile`` still shows as replaced (:func:`replaced_chains`): those stay,
    so an old value does not lose its successor's "previously" to churn.
    """
    entries = [*replaced, *history]
    if len(entries) <= limit:
        return [dict(entry) for entry in entries]
    current = [
        item.model_dump() for item in (*profile.explicit_info, *profile.implicit_traits)
    ]
    shown = {id(e) for chain in replaced_chains(current, entries) for e in chain}
    room = max(0, limit - len(shown))
    kept: list[dict[str, Any]] = []
    for entry in entries:
        if id(entry) in shown:
            kept.append(dict(entry))
        elif room:
            kept.append(dict(entry))
            room -= 1
    return kept[:limit]


def forget_images(
    existing: Profile | None, image_ids: Sequence[str]
) -> tuple[Profile | None, list[tuple[str, BaseModel]], int]:
    """Drop every profile item that rests on forgotten pictures alone.

    An item records the pictures it was read from. Forgetting one of them
    retires the item only when **every** picture behind it is gone: a fact two
    photos established between them survives forgetting either one, and an
    item with no picture source (a text-derived one) is never touched.

    Returns:
        ``(profile, removed, count)`` -- the profile to write back (``None``
        when the owner had none), the removed ``(bucket, item)`` pairs for an
        undo, and how many items went.
    """
    if existing is None:
        return None, [], 0
    gone = set(image_ids)

    def survives(item: BaseModel) -> bool:
        sources = item_source_images(item)
        return not sources or bool(set(sources) - gone)

    removed: list[tuple[str, BaseModel]] = [
        *(
            ("explicit_info", item)
            for item in existing.explicit_info
            if not survives(item)
        ),
        *(
            ("implicit_traits", item)
            for item in existing.implicit_traits
            if not survives(item)
        ),
    ]
    if not removed:
        return existing, [], 0
    kept = existing.model_copy(
        update={
            "explicit_info": [i for i in existing.explicit_info if survives(i)],
            "implicit_traits": [i for i in existing.implicit_traits if survives(i)],
        }
    )
    return kept, removed, len(removed)


def restore_items(
    existing: Profile | None,
    removed: Sequence[tuple[str, BaseModel | Mapping[str, Any]]],
    *,
    owner_id: str,
    timestamp: int,
) -> tuple[Profile, int]:
    """Put items a forget removed back into the profile, once.

    An item is identified by its folded description, the identity the merge
    uses, so an undo after the item was re-established from another source
    does not duplicate it. The item comes back as it was stored, with its
    ``observed_at`` and its provenance intact.

    Returns:
        ``(profile, restored_count)``.
    """
    profile = (
        existing.model_copy(deep=True)
        if existing is not None
        else Profile(owner_id=owner_id, summary="", timestamp=timestamp)
    )
    known = {
        normalised(item.description)
        for item in (*profile.explicit_info, *profile.implicit_traits)
        if item.description
    }
    restored = 0
    for bucket, item in removed:
        data = dict(item) if isinstance(item, Mapping) else item.model_dump()
        identity = normalised(str(data.get("description") or ""))
        if not identity or identity in known:
            continue
        known.add(identity)
        if bucket == "implicit_traits":
            profile.implicit_traits.append(ImplicitTraitItem.model_validate(data))
        else:
            profile.explicit_info.append(ExplicitInfoItem.model_validate(data))
        restored += 1
    return profile, restored


def _merged(*groups: object) -> list[str]:
    """The values of every list in ``groups``, once each, in first-seen order."""
    values: dict[str, None] = {}
    for group in groups:
        if isinstance(group, list | tuple):
            values.update((str(value), None) for value in group)
    return list(values)


def _same_words(before: str, after: str) -> bool:
    """Whether two descriptions differ in punctuation, spacing or case only."""
    return _NOT_WORDS.sub("", before.casefold()) == _NOT_WORDS.sub("", after.casefold())


def _first_message(cell: MemCell) -> int:
    times = [t for t in (getattr(i, "timestamp", 0) for i in cell.items) if t]
    return min(times) if times else cell.timestamp
