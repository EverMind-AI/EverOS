"""Member-in-group profiles: provenance on every item, replaced items kept.

docs/group-memory.md, section 7.4. The profile operator (everalgo
``ProfileExtractor``, the picture call's add-only items) writes the document;
this module keeps what the member profile adds around it:

* every item carries ``item_id`` (stable while its description is), the
  segments it was read from (``source_memcell_ids``) and their span
  (``source_start`` / ``source_end``, epoch ms) -- what a read compares with
  the asker's windows. An item an update kept keeps its provenance; one it
  added or rewrote gets that update's;
* an item an update replaced or dropped moves to ``history`` with the time
  of that update (:func:`bounded_history` keeps :data:`HISTORY_LIMIT`);
* a fact (``explicit_info``) the UPDATE rewrote by its own index is a new
  value of it: the new item names the old one (``supersedes``) and the time
  of the update (``since``), and the old one's history entry names the new
  one (``superseded_by``). everalgo keeps a host's keys on the item an UPDATE
  op patches, so the rewritten item still carries the old ``item_id``: that,
  not a comparison of texts, is the link. COMPACT and REGROUP write new
  items without host keys and a dedupe keeps an item already on file, so
  none of them links anything, and neither does a rewritten implicit trait
  (an inference sharpened, not a value replaced);
* :func:`own_segments` is the source check: only the segments holding a
  message by the member are read, so no item can rest on others' words alone.

The picture route's poster-only items (D17) merge through
:func:`write_member_picture_profile`, under the per-member lock the profile
strategy holds, so the two writers never lose each other's items.
"""

from __future__ import annotations

__all__ = [
    "HISTORY_LIMIT",
    "bounded_history",
    "mint_item_id",
    "own_segments",
    "with_provenance",
    "write_member_picture_profile",
    "wrote_in",
]

import re
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import member_profile_lock
from everos.memory.extract.pipeline import profile_delta
from everos.memory.models import (
    MEMBER_VERSION_KEYS,
    MemberProfile,
    MemCell,
    Profile,
    replaced_chains,
)
from everos.memory.ports import MemberProfileStore, Scope

logger = get_logger(__name__)

HISTORY_LIMIT = 200
"""Replaced items a member profile keeps; past it the oldest drop off first,
except the values a current item still shows as replaced."""

_CARRIED = (
    "item_id",
    "source_memcell_ids",
    "source_start",
    "source_end",
    profile_delta.SOURCE_IMAGE_KEYS,
    *MEMBER_VERSION_KEYS,
)
"""Keys an unchanged item keeps from its stored self."""

_NOT_WORDS = re.compile(r"[\W_]+")
"""What a rewrite that only re-punctuates a description changes."""


def mint_item_id() -> str:
    """A fresh, stable id for a new profile item (``pi_<12 hex>``)."""
    return f"pi_{uuid.uuid4().hex[:12]}"


def wrote_in(cell: MemCell, member_id: str) -> bool:
    """Whether ``member_id`` wrote a message of their own in the segment."""
    return any(
        getattr(item, "sender_id", None) == member_id
        and getattr(item, "role", None) == "user"
        for item in cell.items
    )


def own_segments(memcells: Sequence[MemCell], member_id: str) -> list[MemCell]:
    """The segments in which ``member_id`` wrote a message of their own."""
    return [cell for cell in memcells if wrote_in(cell, member_id)]


def with_provenance(
    old: Profile | None,
    new: Profile,
    *,
    sources: Sequence[str],
    span: tuple[int, int],
    replaced_at: int,
) -> tuple[Profile, list[dict[str, Any]]]:
    """``new`` with every item's provenance, and the old items it no longer holds.

    Items are matched by their folded description, the identity everalgo's
    own merge uses: a match keeps its stored provenance and links, anything
    else is new and is stamped with ``sources`` and ``span``. A new fact
    (``explicit_info``) that still carries the ``item_id`` of an old one this
    update drops is that fact rewritten by an UPDATE op: it names it
    (``supersedes``) with ``since = replaced_at``, unless the rewrite only
    re-punctuated it or two items carry the same old id. An implicit trait is
    an inferred pattern, not a stated value: a rewrite of one sharpens the
    inference and links nothing.

    Returns:
        ``(profile, replaced)`` -- ``replaced`` are the old items' dumps with
        ``replaced_at`` (and ``superseded_by``, the id of the new value, for
        an item rewritten into one), for the document's history.
    """
    stored = (
        {
            profile_delta.normalised(item.description): item
            for item in (*old.explicit_info, *old.implicit_traits)
        }
        if old is not None
        else {}
    )
    items = (*new.explicit_info, *new.implicit_traits)
    held = {profile_delta.normalised(item.description) for item in items}
    leaving = {
        item_id: item
        for key, item in stored.items()
        if key not in held and (item_id := _item_id(item))
    }
    # An UPDATE op patches its item in place, so the rewritten item still
    # carries the stored item's id; COMPACT / REGROUP items carry none.
    rewrites = Counter(
        carried
        for item in new.explicit_info
        if profile_delta.normalised(item.description) not in stored
        and (carried := _item_id(item)) in leaving
    )
    fresh = {
        "source_memcell_ids": list(dict.fromkeys(sources)),
        "source_start": span[0],
        "source_end": span[1],
    }
    successors: dict[str, str] = {}

    def stamp[ItemT: BaseModel](item: ItemT, *, fact: bool) -> ItemT:
        description = getattr(item, "description", "")
        prior = stored.get(profile_delta.normalised(description))
        extra = {k: v for k, v in (item.model_extra or {}).items() if k not in _CARRIED}
        if prior is not None and _item_id(prior):
            kept = prior.model_extra or {}
            extra.update({k: kept[k] for k in _CARRIED if k in kept})
        else:
            carried = (item.model_extra or {}).get(profile_delta.SOURCE_IMAGE_KEYS)
            if carried is not None:
                extra[profile_delta.SOURCE_IMAGE_KEYS] = carried
            extra.update(fresh, item_id=mint_item_id())
            replaced_id = _item_id(item)
            if (
                fact
                and prior is None
                and rewrites.get(replaced_id) == 1
                and not _same_words(leaving[replaced_id].description, description)
            ):
                extra.update(supersedes=replaced_id, since=replaced_at)
                successors[replaced_id] = extra["item_id"]
        declared = type(item).model_fields
        fields = {k: v for k, v in item.model_dump().items() if k in declared}
        return type(item).model_validate(fields | extra)

    explicit = [stamp(item, fact=True) for item in new.explicit_info]
    traits = [stamp(item, fact=False) for item in new.implicit_traits]
    replaced = [
        {
            **item.model_dump(),
            "replaced_at": replaced_at,
            **(
                {"superseded_by": successors[item_id]}
                if (item_id := _item_id(item)) in successors
                else {}
            ),
        }
        for key, item in stored.items()
        if key not in held
    ]
    profile = new.model_copy(
        update={"explicit_info": explicit, "implicit_traits": traits}
    )
    return profile, replaced


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


def _item_id(item: BaseModel) -> str:
    return str((item.model_extra or {}).get("item_id") or "")


def _same_words(before: str, after: str) -> bool:
    """Whether two descriptions differ in punctuation, spacing or case only."""
    return _NOT_WORDS.sub("", before.casefold()) == _NOT_WORDS.sub("", after.casefold())


async def write_member_picture_profile(
    *,
    store: MemberProfileStore,
    scope: Scope,
    member_id: str,
    ops: Sequence[profile_delta.ProfileItemOp],
    observed_at: int,
    image_ids: Sequence[str],
    memcell_id: str,
    span: tuple[int, int],
    language: str | None,
) -> int:
    """Merge a poster's picture items into their profile in this group.

    The add-only merge of :func:`profile_delta.apply_ops`, then the new items
    get their provenance: the picture's segment and the time it was posted.

    Returns:
        How many items were added (0 when the call found nothing new).
    """
    if not profile_delta.adds_anything(ops):
        return 0
    async with member_profile_lock(
        scope.app_id, scope.project_id, scope.owner_id, member_id
    ):
        existing = await store.read(scope, member_id)
        merged, added = profile_delta.apply_ops(
            existing.profile if existing is not None else None,
            ops,
            owner_id=member_id,
            observed_at=observed_at,
            image_ids=image_ids,
        )
        if not added:
            return 0
        merged, _ = with_provenance(
            existing.profile if existing is not None else None,
            merged,
            sources=[memcell_id],
            span=span,
            replaced_at=observed_at,
        )
        if merged.primary_language is None and language is not None:
            merged.primary_language = str(language)
        await store.write(
            scope,
            existing.model_copy(update={"profile": merged})
            if existing is not None
            else MemberProfile(
                group_id=scope.owner_id, member_id=member_id, profile=merged
            ),
        )
    logger.info(
        "memory.images.member_profile_updated",
        app_id=scope.app_id,
        project_id=scope.project_id,
        group_id=scope.owner_id,
        added=added,
        photos=len(image_ids),
    )
    return added
