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
  of that update (:data:`HISTORY_LIMIT` newest kept);
* :func:`own_segments` is the source check: only the segments holding a
  message by the member are read, so no item can rest on others' words alone.

The picture route's poster-only items (D17) merge through
:func:`write_member_picture_profile`, under the per-member lock the profile
strategy holds, so the two writers never lose each other's items.
"""

from __future__ import annotations

__all__ = [
    "HISTORY_LIMIT",
    "mint_item_id",
    "own_segments",
    "with_provenance",
    "write_member_picture_profile",
]

import uuid
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import member_profile_lock
from everos.memory.extract.pipeline import profile_delta
from everos.memory.models import MemberProfile, MemCell, Profile
from everos.memory.ports import MemberProfileStore, Scope

logger = get_logger(__name__)

HISTORY_LIMIT = 200
"""Replaced items a member profile keeps, newest first; older ones drop off."""

_CARRIED = (
    "item_id",
    "source_memcell_ids",
    "source_start",
    "source_end",
    profile_delta.SOURCE_IMAGE_KEYS,
)
"""Keys an unchanged item keeps from its stored self."""


def mint_item_id() -> str:
    """A fresh, stable id for a new profile item (``pi_<12 hex>``)."""
    return f"pi_{uuid.uuid4().hex[:12]}"


def own_segments(memcells: Sequence[MemCell], member_id: str) -> list[MemCell]:
    """The segments in which ``member_id`` wrote a message of their own."""
    return [
        cell
        for cell in memcells
        if any(
            getattr(item, "sender_id", None) == member_id
            and getattr(item, "role", None) == "user"
            for item in cell.items
        )
    ]


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
    own merge uses: a match keeps its stored provenance, anything else is
    new and is stamped with ``sources`` and ``span``.

    Returns:
        ``(profile, replaced)`` -- ``replaced`` are the old items' dumps with
        ``replaced_at``, for the document's history.
    """
    stored = (
        {
            profile_delta.normalised(item.description): item
            for item in (*old.explicit_info, *old.implicit_traits)
        }
        if old is not None
        else {}
    )
    fresh = {
        "source_memcell_ids": list(dict.fromkeys(sources)),
        "source_start": span[0],
        "source_end": span[1],
    }

    def stamp[ItemT: BaseModel](item: ItemT) -> ItemT:
        prior = stored.get(profile_delta.normalised(getattr(item, "description", "")))
        extra = dict(item.model_extra or {})
        if prior is not None and (prior.model_extra or {}).get("item_id"):
            kept = prior.model_extra or {}
            extra.update({k: kept[k] for k in _CARRIED if k in kept})
        else:
            extra.update(fresh, item_id=mint_item_id())
        return type(item).model_validate({**item.model_dump(), **extra})

    explicit = [stamp(item) for item in new.explicit_info]
    traits = [stamp(item) for item in new.implicit_traits]
    held = {profile_delta.normalised(i.description) for i in (*explicit, *traits)}
    replaced = [
        {**item.model_dump(), "replaced_at": replaced_at}
        for key, item in stored.items()
        if key not in held
    ]
    profile = new.model_copy(
        update={"explicit_info": explicit, "implicit_traits": traits}
    )
    return profile, replaced


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
