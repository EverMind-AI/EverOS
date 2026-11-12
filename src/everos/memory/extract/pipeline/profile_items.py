"""Profile items: provenance on every item, versions linked, replaced items kept.

The one writer of a person's profile in either space -- the one-to-one
``user.md`` and a member's profile in a group (docs/group-memory.md, section
7.4) -- is everalgo's
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

A forget takes items and replaced values out of either profile one way
(:func:`take_out`): what it takes keeps its place, and taking an item out
drops the summary too -- it is written over every item, so it may repeat the
forgotten one. The next update writes it again from the items that remain
(everalgo derives it on every update), and an undo puts it back while no
update has. A picture forget takes what rests on forgotten pictures alone in
either space (:func:`rests_on_pictures`). Which segments a forget reaches
differs by space: a one-to-one forget takes what every segment of which is
forgotten (:func:`forget_segments`), a forget inside a group what no segment
of the member's own words supports any more (the group forget's plan). The
undo follows what each kept: a one-to-one forget logged its items without
their places, so they come back by description, at the end
(:func:`restore_items`); a group's logged their places, so they come back by
id, where they were (:func:`put_back`).

A person changes an item through ``POST /memory/update`` (:func:`edit_item`):
a new value is a new item linked to the old one, as a model's is, and carries
who wrote it; it is **held** unless the caller says otherwise. The next update
keeps a held item as it is (:func:`keep_held`): the model reads it, and
whatever its reply did to it -- a new value, a retirement, a merge -- leaves
the held item standing where it was; a new value the model wrote for it stays
beside it as an item of its own, linked to nothing.
"""

from __future__ import annotations

__all__ = [
    "BUCKETS",
    "HISTORY_LIMIT",
    "SOURCE_IMAGE_KEYS",
    "SUMMARY",
    "Taken",
    "already_read",
    "bounded_history",
    "edit_item",
    "find_item",
    "forget_images",
    "forget_items",
    "forget_segments",
    "item_source_images",
    "keep_held",
    "mint_item_id",
    "normalised",
    "put_back",
    "restore_items",
    "rests_on",
    "rests_on_pictures",
    "stamp_update",
    "take_out",
]

import re
import uuid
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from everos.memory.models import (
    EDIT_KEYS,
    MEMBER_VERSION_KEYS,
    ExplicitInfoItem,
    ImplicitTraitItem,
    MemCell,
    Profile,
    is_held,
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
    *EDIT_KEYS,
)
"""The keys :func:`stamp_update` owns on an item. A new item the model wrote
never inherits a person's edit marks from the item it came from."""

BUCKETS: tuple[str, str] = ("explicit_info", "implicit_traits")
"""The two lists of profile items."""

SUMMARY = "summary"
"""The part a forget's removed list names the dropped summary by:
``("summary", {"summary": <text>})``, beside the ``(bucket, item)`` and
``("history", entry)`` pairs, so the undo can put it back."""

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
        """The item's fields and extras with its provenance set.

        A person's edit marks ride only on an item kept as it was; an item the
        reply wrote (a new value, a merge, an addition) is the model's.
        """
        fields = {k: v for k, v in item.model_dump().items() if k not in EDIT_KEYS}
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
) -> tuple[Profile | None, list[tuple[str, dict[str, Any]]], int]:
    """Drop every item and replaced value that rests on forgotten pictures alone.

    An item records the pictures it was read from. Forgetting one of them
    retires the item only when **every** picture behind it is gone: a fact two
    photos established between them survives forgetting either one, and an
    item with no picture source (a text-derived one) is never touched. A
    replaced value in the profile's history goes by the same rule, so what the
    picture showed surfaces from neither.

    Returns:
        ``(profile, removed, count)`` -- the profile to write back (``None``
        when the owner had none), the removed ``(bucket, entry)`` pairs for an
        undo (bucket ``history`` for a replaced value), and how many items
        went.
    """
    if existing is None:
        return None, [], 0
    gone = set(image_ids)
    return _without(existing, lambda entry: not rests_on_pictures(entry, gone))


def rests_on_pictures(
    entry: BaseModel | Mapping[str, Any], pictures: Collection[str]
) -> bool:
    """Whether every picture an item or replaced value was read off is gone.

    A fact two photos established between them survives forgetting either
    one, and an entry with no picture source (a text-derived one) is never a
    picture forget's.
    """
    sources = item_source_images(entry)
    return bool(sources) and set(sources) <= set(pictures)


@dataclass(frozen=True)
class Taken:
    """What a forget took out of a profile, each part with its place.

    Attributes:
        items: ``(bucket, position, item)`` per item taken out, bucket by
            bucket in list order; ``item`` as stored, JSON-ready.
        history: ``(position, entry)`` per replaced value taken out.
        summary: The summary taken out with the items; ``None`` when no item
            went.
    """

    items: tuple[tuple[str, int, dict[str, Any]], ...] = ()
    history: tuple[tuple[int, dict[str, Any]], ...] = ()
    summary: str | None = None


def take_out(
    profile: Profile,
    *,
    item: Callable[[_Item], bool],
    entry: Callable[[int, Mapping[str, Any]], bool],
) -> tuple[Profile, Taken]:
    """``profile`` without the items ``item`` names and the history ``entry`` names.

    ``entry`` is given each replaced value with its position. Taking an item
    out drops the summary as well (:data:`SUMMARY`). Nothing taken returns
    ``profile`` itself.
    """
    items: list[tuple[str, int, dict[str, Any]]] = []
    kept: dict[str, Any] = {}
    for bucket in BUCKETS:
        current: list[_Item] = getattr(profile, bucket)
        kept[bucket] = [i for i in current if not item(i)]
        items += [
            (bucket, n, i.model_dump(mode="json"))
            for n, i in enumerate(current)
            if item(i)
        ]
    history = [(n, dict(e)) for n, e in enumerate(profile.history) if entry(n, e)]
    if not items and not history:
        return profile, Taken()
    kept["history"] = [e for n, e in enumerate(profile.history) if not entry(n, e)]
    if items:
        kept["summary"] = ""
    return profile.model_copy(update=kept), Taken(
        items=tuple(items),
        history=tuple(history),
        summary=profile.summary if items else None,
    )


def put_back(profile: Profile, taken: Taken) -> tuple[Profile, list[str]]:
    """What :func:`take_out` took, back in its place; and the items' ids.

    An item comes back where it was unless the profile holds its id again; a
    replaced value comes back at its position in the history; the summary
    comes back while the profile has none (an update since has written one
    over the items it had).
    """
    present = {
        (i.model_extra or {}).get("item_id")
        for i in (*profile.explicit_info, *profile.implicit_traits)
    }
    back: dict[str, list[tuple[int, _Item]]] = {bucket: [] for bucket in BUCKETS}
    for bucket, n, data in taken.items:
        if data.get("item_id") in present:
            continue
        model = ImplicitTraitItem if bucket == "implicit_traits" else ExplicitInfoItem
        back[bucket].append((n, model.model_validate(data)))
    restored = profile.model_copy(
        update={
            **{
                bucket: _in_place(getattr(profile, bucket), back[bucket])
                for bucket in BUCKETS
            },
            "summary": taken.summary
            if taken.summary is not None and not profile.summary
            else profile.summary,
            "history": _in_place(
                profile.history, [(n, dict(e)) for n, e in taken.history]
            ),
        }
    )
    ids = [
        str((i.model_extra or {}).get("item_id") or "")
        for bucket in BUCKETS
        for _, i in back[bucket]
    ]
    return restored, ids


def _in_place[T](items: Sequence[T], placed: Sequence[tuple[int, T]]) -> list[T]:
    """``items`` with each ``(place, item)`` inserted where it was, in order."""
    out = list(items)
    for n, item in sorted(placed, key=lambda pair: pair[0]):
        out.insert(min(n, len(out)), item)
    return out


def _without(
    existing: Profile, survives: Callable[[BaseModel | Mapping[str, Any]], bool]
) -> tuple[Profile, list[tuple[str, dict[str, Any]]], int]:
    """``existing`` without the items and replaced values ``survives`` refuses.

    :func:`take_out`, in the one-to-one forget log's form: ``(bucket, item)``
    pairs, then ``("history", entry)``, then the dropped summary under
    :data:`SUMMARY`, all without their places.
    """
    kept, taken = take_out(
        existing,
        item=lambda i: not survives(i),
        entry=lambda _n, e: not survives(e),
    )
    removed: list[tuple[str, dict[str, Any]]] = [
        (bucket, item) for bucket, _n, item in taken.items
    ]
    removed += [("history", entry) for _n, entry in taken.history]
    if taken.summary:
        removed.append((SUMMARY, {"summary": taken.summary}))
    return kept, removed, len(taken.items)


def forget_segments(
    existing: Profile | None,
    segments: Collection[str],
    *,
    already: Collection[str] = (),
) -> tuple[Profile | None, list[tuple[str, dict[str, Any]]], int]:
    """Drop every item and replaced value that no remembered segment supports.

    ``segments`` are the forgotten conversation's, ``already`` those of the
    owner's earlier forgets no undo reversed. An entry goes when every segment
    it rests on is among them and at least one is among ``segments``: what an
    earlier forget left standing is that forget's to take. An entry with no
    recorded segments (stored before entries had them) is never touched, nor
    is one a remembered segment still supports.

    Returns:
        ``(profile, removed, count)`` as :func:`forget_images` returns them.
    """
    if existing is None:
        return None, [], 0
    these, gone = set(segments), {*segments, *already}

    def survives(entry: BaseModel | Mapping[str, Any]) -> bool:
        extra = entry if isinstance(entry, Mapping) else (entry.model_extra or {})
        raw = extra.get("source_memcell_ids")
        sources = {str(s) for s in raw} if isinstance(raw, list) else set()
        return not sources or not sources <= gone or not sources & these

    return _without(existing, survives)


def already_read(profile: Profile | None, memcell_id: str) -> bool:
    """Whether an item of ``profile`` already rests on the segment.

    Every item an update writes names the segments it was read from, so a
    segment event delivered again after its update ran (a replay) finds its id
    on the profile: both profiles ask this before they read a segment, so no
    segment is read twice.
    """
    if profile is None:
        return False
    return any(
        memcell_id in ((item.model_extra or {}).get("source_memcell_ids") or [])
        for item in (*profile.explicit_info, *profile.implicit_traits)
    )


def rests_on(entry: Mapping[str, Any], segments: Collection[str]) -> bool:
    """Whether a removed item or replaced value names one of ``segments``."""
    raw = entry.get("source_memcell_ids")
    return isinstance(raw, list) and bool({str(s) for s in raw} & set(segments))


def restore_items(
    existing: Profile | None,
    removed: Sequence[tuple[str, BaseModel | Mapping[str, Any]]],
    *,
    owner_id: str,
    timestamp: int,
) -> tuple[Profile, int, int]:
    """Put what a forget removed back into the profile, once.

    An item is identified by its folded description, the identity the merge
    uses, so an undo after the item was re-established from another source
    does not duplicate it. The item comes back as it was stored, with its
    ``observed_at`` and its provenance intact. A replaced value (bucket
    ``history``) goes back into the history in its place by ``replaced_at``,
    newest first, unless the history still holds it. The summary the forget
    dropped (:data:`SUMMARY`) comes back while the profile has none: an
    update since has written one over the items it had.

    Returns:
        ``(profile, items, replaced)`` -- how many items and how many replaced
        values came back.
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
    held = {_history_key(entry) for entry in profile.history}
    restored = replaced = 0
    for bucket, item in removed:
        data = dict(item) if isinstance(item, Mapping) else item.model_dump()
        if bucket == SUMMARY:
            if not profile.summary:
                profile.summary = str(data.get("summary") or "")
            continue
        if bucket == "history":
            if _history_key(data) not in held:
                held.add(_history_key(data))
                profile.history.append(data)
                replaced += 1
            continue
        identity = normalised(str(data.get("description") or ""))
        if not identity or identity in known:
            continue
        known.add(identity)
        if bucket == "implicit_traits":
            profile.implicit_traits.append(ImplicitTraitItem.model_validate(data))
        else:
            profile.explicit_info.append(ExplicitInfoItem.model_validate(data))
        restored += 1
    if replaced:  # newest first; a stable sort keeps the order of equal times
        profile.history.sort(key=_replaced_at, reverse=True)
    return profile, restored, replaced


def keep_held(
    old: Profile | None,
    profile: Profile,
    replaced: Sequence[Mapping[str, Any]],
) -> tuple[Profile, list[dict[str, Any]], list[str]]:
    """``profile`` with every item a person holds back as it stood in ``old``.

    The plan's "an item a person changed is not changed by the model again":
    the update reads a held item like any other (so it does not add the fact a
    second time), and this undoes whatever its reply did to it. The held item
    stands where the reply left it, or where its new value stands, or -- when
    the reply retired or merged it -- at the end of its list. Its history
    entry from this update is dropped. A new value the model wrote for it is
    kept as an item of its own with its ``supersedes`` / ``since`` link
    removed: the person's value is not replaced, and what the conversation
    said is not lost either. A merge that took a held item keeps the merged
    item.

    Returns:
        ``(profile, replaced, kept)`` -- the profile and the history entries
        to write, and the ids of the held items the reply had changed.
    """
    entries = [dict(entry) for entry in replaced]
    if old is None:
        return profile, entries, []
    held: dict[str, tuple[str, _Item]] = {}
    for bucket in BUCKETS:
        for item in getattr(old, bucket):
            item_id = (item.model_extra or {}).get("item_id")
            if isinstance(item_id, str) and is_held(item):
                held[item_id] = (bucket, item)
    if not held:
        return profile, entries, []
    unchanged = {
        str((item.model_extra or {}).get("item_id"))
        for bucket in BUCKETS
        for item in getattr(profile, bucket)
        if (item.model_extra or {}).get("item_id") in held
        and item.description == held[str(item.model_extra["item_id"])][1].description
    }
    placed: set[str] = set()
    lists: dict[str, list[_Item]] = {bucket: [] for bucket in BUCKETS}
    for bucket in BUCKETS:
        for item in getattr(profile, bucket):
            extra = item.model_extra or {}
            own, replaces = extra.get("item_id"), extra.get("supersedes")
            if own in held:
                if own not in placed:
                    lists[held[own][0]].append(held[own][1])
                    placed.add(str(own))
                continue
            if replaces in held:
                if replaces not in placed:
                    lists[held[replaces][0]].append(held[replaces][1])
                    placed.add(str(replaces))
                item = type(item).model_validate(
                    {
                        k: v
                        for k, v in item.model_dump().items()
                        if k not in MEMBER_VERSION_KEYS
                    }
                )
            lists[bucket].append(item)
    for item_id, (bucket, item) in held.items():
        if item_id not in placed:
            lists[bucket].append(item)
    kept_back = [entry for entry in entries if entry.get("item_id") not in held]
    changed = [item_id for item_id in held if item_id not in unchanged]
    return profile.model_copy(update=lists), kept_back, changed


def find_item(
    profile: Profile | None,
    history: Sequence[Mapping[str, Any]],
    item_id: str,
) -> tuple[str, Mapping[str, Any]] | None:
    """Where ``item_id`` is: ``(bucket, item)`` for a current item, ``("history",
    entry)`` for a replaced value (its newest entry), ``None`` when neither."""
    if profile is not None:
        for bucket in BUCKETS:
            for item in getattr(profile, bucket):
                if (item.model_extra or {}).get("item_id") == item_id:
                    return bucket, item.model_dump()
    for entry in history:
        if isinstance(entry, Mapping) and entry.get("item_id") == item_id:
            return "history", entry
    return None


def edit_item(
    profile: Profile,
    history: Sequence[Mapping[str, Any]],
    item_id: str,
    *,
    description: str | None,
    hold: bool,
    edited_by: str,
    at: int,
) -> tuple[Profile, list[dict[str, Any]], str]:
    """A person's change of the current item ``item_id``.

    A new ``description`` is a new value: a new item (a fresh id) that keeps
    the old one's provenance -- its segments, span, pictures and category --
    names it in ``supersedes``, and records who wrote it, when (``since`` and
    ``edited_at``) and whether it is held; the old item moves to the history
    with ``replaced_at`` and the new id in ``superseded_by``, as a model's new
    value does. Without a new description only the hold changes, in place: a
    release is no new value.

    Returns:
        ``(profile, history, item_id)`` -- what to write, and the id of the
        item that now holds the value.

    Raises:
        KeyError: ``item_id`` is no current item of ``profile``.
    """
    for bucket in BUCKETS:
        items: list[_Item] = list(getattr(profile, bucket))
        for index, item in enumerate(items):
            if (item.model_extra or {}).get("item_id") != item_id:
                continue
            old = item.model_dump()
            if description is None or description == item.description:
                data = {**old, "held": hold}
                if hold:
                    data |= {"edited_by": edited_by, "edited_at": at}
                items[index] = type(item).model_validate(data)
                return (
                    profile.model_copy(update={bucket: items}),
                    list(history),
                    item_id,
                )
            new_id = mint_item_id()
            data = {
                **{k: v for k, v in old.items() if k not in (*EDIT_KEYS, "previous")},
                "description": description,
                "item_id": new_id,
                "supersedes": item_id,
                "since": at,
                "edited_by": edited_by,
                "edited_at": at,
                "held": hold,
            }
            items[index] = type(item).model_validate(data)
            entry = {**old, "replaced_at": at, "superseded_by": new_id}
            return (
                profile.model_copy(update={bucket: items}),
                [entry, *(dict(e) for e in history)],
                new_id,
            )
    raise KeyError(item_id)


def forget_items(
    existing: Profile | None, item_ids: Collection[str]
) -> tuple[Profile | None, list[tuple[str, dict[str, Any]]], int]:
    """Drop the items and replaced values named by ``item_ids`` (a forget by id).

    Returns:
        ``(profile, removed, count)`` as :func:`forget_images` returns them.
    """
    if existing is None:
        return None, [], 0
    wanted = set(item_ids)

    def survives(entry: BaseModel | Mapping[str, Any]) -> bool:
        extra = entry if isinstance(entry, Mapping) else (entry.model_extra or {})
        return extra.get("item_id") not in wanted

    return _without(existing, survives)


def _history_key(entry: Mapping[str, Any]) -> tuple[str, str, object]:
    """What tells one replaced value from another: its item, text and time."""
    return (
        str(entry.get("item_id") or ""),
        normalised(str(entry.get("description") or "")),
        entry.get("replaced_at"),
    )


def _replaced_at(entry: Mapping[str, Any]) -> int:
    value = entry.get("replaced_at")
    return value if isinstance(value, int) else 0


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
