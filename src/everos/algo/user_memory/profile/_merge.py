"""Merging an UPDATE reply's ops into the stored profile, deduplication, and the profile built from its buckets.

Reply items become stored items here and take their ``observed_at``; a host's own keys on a stored item ride
through every merge untouched.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

from everos.algo.types import ExplicitInfoItem, ImplicitTraitItem, Profile
from everos.algo.user_memory.profile._items import (
    _ITEM_WIDTH_BACKSTOP,
    _LABEL_FIELDS,
    _OBSERVED_AT,
    _PROFILE_MAX_PER_CATEGORY,
    _dated_buckets,
    _Item,
    _item_label,
    _label,
    _normalize,
)
from everos.algo.utils import ascii_width

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

# The package's name rather than this module's: hosts collect the log contract under it.
logger = logging.getLogger("everos.algo.user_memory.profile")

_ITEM_MODELS: dict[str, type[_Item]] = {
    "explicit_info": ExplicitInfoItem,
    "implicit_traits": ImplicitTraitItem,
}

# What an update does to its item's description: nothing, earlier history added to it, or a new current value.
_DescriptionChange = Literal["kept", "history", "rewrite"]
# A stored description may end a sentence; history goes between its words and that stop, so the stop is not compared.
_SENTENCE_STOPS = ".!?\u3002\uff01\uff1f"  # ASCII and fullwidth
_HISTORY_OPENERS = ("(", "\uff08")  # ASCII and fullwidth


def _stored_items(
    bucket: str, replies: Iterable[Mapping[str, Any]], observed_at: int
) -> list[_Item]:
    """The bucket's stored items built from reply items, each dated ``observed_at`` whatever the reply held."""
    model = _ITEM_MODELS[bucket]
    return [
        model.model_validate({**reply, _OBSERVED_AT: observed_at}) for reply in replies
    ]


def _stale_since(item: _Item, observed_at: int) -> int | None:
    """The date that established ``item`` when ``observed_at`` is earlier — older knowledge, no rewrites — else None."""
    established = item.observed_at
    return (
        established if established is not None and observed_at < established else None
    )


def _changes_description(item: _Item, patch: Mapping[str, Any]) -> bool:
    """True when ``patch`` states a different fact than ``item``; grounding-only patches are not rewrites."""
    proposed = _identity_key(patch.get("description"))
    return proposed is not None and proposed != _identity_key(item.description)


def _appends_history(item: _Item, patch: Mapping[str, Any]) -> bool:
    """True when ``patch``'s description is ``item``'s, closing stop dropped, followed by a parenthesis.

    Both are folded by ``_normalize``. A mechanical test: it refuses a negation put before the stored words and a
    contradiction around them, and cannot tell history from a contradiction written inside the parenthesis.
    """
    current = _identity_key(item.description)
    proposed = _identity_key(patch.get("description"))
    if current is None or proposed is None:
        return False
    stem = current.rstrip(_SENTENCE_STOPS)
    added = proposed.removeprefix(stem)
    return (
        bool(stem) and added != proposed and added.lstrip().startswith(_HISTORY_OPENERS)
    )


def _description_change(
    item: _Item, patch: Mapping[str, Any], *, observed_at: int
) -> _DescriptionChange:
    """What ``patch``, from a batch observed at ``observed_at``, does to ``item``'s description.

    ``"history"`` is a batch older than the item keeping the stored description as it opens and adding the earlier
    value in a parenthesis after it: the later source still holds the current value. Any other change is a
    ``"rewrite"``, a newer batch extending a description included — that states a new current value.
    """
    if not _changes_description(item, patch):
        return "kept"
    if _stale_since(item, observed_at) is not None and _appends_history(item, patch):
        return "history"
    return "rewrite"


def _patched(item: _Item, patch: Mapping[str, Any]) -> _Item:
    """``item`` with ``patch`` merged over the fields it holds, revalidated."""
    return item.model_validate({**item.model_dump(exclude_unset=True), **patch})


@dataclass
class _BucketOps:
    """One bucket's ops after validation, keyed by the index numbering the LLM was shown."""

    updates: dict[int, dict[str, Any]] = field(default_factory=dict)
    deletes: set[int] = field(default_factory=set)
    adds: list[dict[str, Any]] = field(default_factory=list)
    canonicalised: int = 0


@dataclass
class _CollectedOps:
    """An UPDATE reply's ops after validation: each bucket's queue, and the counts the update log line reports.

    ``actions`` counts the reply's ops by action as proposed; ``rejected`` the ops refused, the updates to an index
    the reply also deletes counting once; ``rewrote_description`` the queued updates that restate their item's fact,
    ``history_appended`` those that add earlier history to it.
    """

    slots: dict[str, _BucketOps]
    actions: Counter[str] = field(default_factory=Counter[str])
    rejected: int = 0
    rewrote_description: int = 0
    history_appended: int = 0


def _apply_ops(
    old_profile: Profile, ops: list[dict[str, Any]], *, timestamp: int, pass_name: str
) -> Profile:
    """Apply an UPDATE reply's ops to ``old_profile`` and return a new Profile, leaving ``old_profile`` untouched.

    Every ``index`` resolves against ``old_profile`` as rendered for the model; a refused op is logged and dropped.
    ``timestamp``, the source batch's observation date, dates added items and rewritten descriptions, refuses as
    stale a delete or a rewrite of an item established after it — earlier history added to its description is not
    one — and never moves the profile's timestamp back.
    ``pass_name`` only labels the log line.
    """
    explicit_items, implicit_items = _dated_buckets(old_profile)
    buckets = {"explicit_info": explicit_items, "implicit_traits": implicit_items}
    collected = _collect_ops(ops, buckets, observed_at=timestamp)
    merged = _profile_from_buckets(
        owner_id=old_profile.owner_id,
        timestamp=max(old_profile.timestamp, timestamp),
        explicit_info=_apply_bucket_ops(
            "explicit_info",
            explicit_items,
            collected.slots["explicit_info"],
            observed_at=timestamp,
        ),
        implicit_traits=_apply_bucket_ops(
            "implicit_traits",
            implicit_items,
            collected.slots["implicit_traits"],
            observed_at=timestamp,
        ),
    )
    _log_update_applied(
        collected,
        pass_name=pass_name,
        observed_at=timestamp,
        before=old_profile,
        after=merged,
    )
    return merged


def _collect_ops(
    ops: list[dict[str, Any]],
    buckets: Mapping[str, Sequence[_Item]],
    *,
    observed_at: int,
) -> _CollectedOps:
    """Group ops by bucket, validating each against ``buckets``, whose keys are the only bucket names accepted."""
    collected = _CollectedOps(
        slots={name: _BucketOps() for name in buckets},
        actions=Counter(str(op.get("action")) for op in ops),
    )
    for op in ops:
        if (
            op.get("action") == "none"
        ):  # documented no-op: the conversation carried no user info
            continue
        if not _collect_op(op, collected.slots, buckets, observed_at=observed_at):
            collected.rejected += 1

    for slot in collected.slots.values():
        collected.rejected += _drop_updates_superseded_by_delete(slot)
    changes = _count_description_changes(
        buckets, collected.slots, observed_at=observed_at
    )
    collected.rewrote_description = changes["rewrite"]
    collected.history_appended = changes["history"]
    return collected


def _collect_op(
    op: dict[str, Any],
    slots: dict[str, _BucketOps],
    buckets: Mapping[str, Sequence[_Item]],
    *,
    observed_at: int,
) -> bool:
    """Queue one op in its bucket's slot; False, with a warning, when it is refused."""
    action = op.get("action")
    op_type = op.get("type")
    if op_type not in slots:
        _log_rejected_op(op, "unknown type")
        return False
    if action == "add":
        return _collect_add(
            op,
            slots[op_type],
            items=buckets[op_type],
            label_field=_LABEL_FIELDS[op_type],
        )
    if action in ("update", "delete"):
        return _collect_indexed(
            op, slots[op_type], items=buckets[op_type], observed_at=observed_at
        )
    _log_rejected_op(op, "unknown action")
    return False


def _collect_add(
    op: dict[str, Any], slot: _BucketOps, *, items: Sequence[_Item], label_field: str
) -> bool:
    """Queue an add as a sibling item, its label respelled to the spelling on file; False, logged, if refused."""
    data = op.get("data")
    if not _is_object(data):
        _log_rejected_op(op, "data is not an object")
        return False
    if _identity_key(data.get("description")) is None:
        _log_rejected_op(op, "missing description")
        return False

    label = data.get(label_field)
    if (
        isinstance(label, str)
        and (on_file := _label_on_file(items, label)) is not None
        and label != on_file
    ):
        logger.debug(
            "profile add label canonicalised: bucket=%s from_len=%d to_len=%d",
            op.get("type"),
            len(label),
            len(on_file),
        )
        data = {**data, label_field: on_file}
        slot.canonicalised += 1
    slot.adds.append(data)
    return True


def _label_on_file(items: Sequence[_Item], label: str) -> str | None:
    """The stored spelling of the first item whose label matches ``label`` by ``_normalize``; None if free."""
    if not label.strip():
        return None
    wanted = _normalize(label)
    return next(
        (
            current
            for item in items
            if (current := _label(item)) is not None and _normalize(current) == wanted
        ),
        None,
    )


def _overcrowded_labels(
    explicit_info: Sequence[_Item], implicit_traits: Sequence[_Item]
) -> list[tuple[str, str]]:
    """``(bucket, label)`` pairs needing a regroup pass: too many items, or one item past the width backstop.

    Each label is its first-seen stripped spelling; matching is by ``_normalize``. Order is explicit_info first,
    then first-seen label order.
    """
    flagged: list[tuple[str, str]] = []
    for bucket, items in (
        ("explicit_info", explicit_info),
        ("implicit_traits", implicit_traits),
    ):
        counts: Counter[str] = Counter()
        first_spelling: dict[str, str] = {}
        oversized: set[str] = set()
        for item in items:
            label = _item_label(item)
            if label is None:
                continue
            key = _normalize(label)
            first_spelling.setdefault(key, label)
            counts[key] += 1
            if ascii_width(item.description) > _ITEM_WIDTH_BACKSTOP:
                oversized.add(key)
        flagged += [
            (bucket, spelling)
            for key, spelling in first_spelling.items()
            if counts[key] > _PROFILE_MAX_PER_CATEGORY or key in oversized
        ]
    return flagged


def _collect_indexed(
    op: dict[str, Any], slot: _BucketOps, *, items: Sequence[_Item], observed_at: int
) -> bool:
    """Queue an update or delete whose index falls inside the original snapshot; False, logged, if refused.

    Against an item established after ``observed_at``, a delete or a description rewrite is stale evidence; an
    update that only adds grounding, or keeps the description and adds earlier history in a parenthesis after it,
    still lands.
    """
    idx = op.get("index")
    if not _is_int(idx) or not 0 <= idx < len(items):
        _log_rejected_op(op, "index out of range")
        return False
    stale_since = _stale_since(items[idx], observed_at)
    if op.get("action") == "delete":
        if stale_since is not None:
            _log_stale_op(
                op, item_observed_at=stale_since, batch_observed_at=observed_at
            )
            return False
        slot.deletes.add(idx)
        return True
    data = op.get("data")
    if not _is_object(data):
        _log_rejected_op(op, "data is not an object")
        return False
    if (
        stale_since is not None
        and _description_change(items[idx], data, observed_at=observed_at) == "rewrite"
    ):
        _log_stale_op(op, item_observed_at=stale_since, batch_observed_at=observed_at)
        return False
    # An update is a partial merge, so two of them on one index accumulate rather than replace.
    slot.updates[idx] = {**slot.updates.get(idx, {}), **data}
    return True


def _is_int(value: object) -> TypeGuard[int]:
    """True for an int; ``bool`` is an int subclass and is excluded."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_object(value: object) -> TypeGuard[dict[str, Any]]:
    """True for a JSON object, the only shape an op's ``data`` can be merged from."""
    return isinstance(value, dict)


def _drop_updates_superseded_by_delete(slot: _BucketOps) -> int:
    """Resolve an index that is both updated and deleted in favour of the delete; return how many updates fell."""
    superseded = sorted(
        slot.updates.keys() & slot.deletes
    )  # sorted so the warnings come out in a fixed order
    for idx in superseded:
        del slot.updates[idx]
        logger.warning(
            "profile update op rejected: action=update index=%d reason=superseded by delete",
            idx,
        )
    return len(superseded)


def _count_description_changes(
    buckets: Mapping[str, Sequence[_Item]],
    slots: dict[str, _BucketOps],
    *,
    observed_at: int,
) -> Counter[_DescriptionChange]:
    """The queued updates by what they do to their item's description, judged against the unpatched snapshot."""
    return Counter(
        _description_change(buckets[name][idx], patch, observed_at=observed_at)
        for name, slot in slots.items()
        for idx, patch in slot.updates.items()
    )


def _apply_bucket_ops(
    bucket: str, items: Sequence[_Item], bucket_ops: _BucketOps, *, observed_at: int
) -> list[_Item]:
    """Apply one bucket's ops: patch by original index, drop deletes in one pass, then append the new items.

    A rewritten description and every added item take ``observed_at``; a patch adding only grounding or earlier
    history keeps the item's date, since the source that set the current value is still the one that dated it.
    """
    patched = list(items)
    for idx, patch in bucket_ops.updates.items():
        rewrites = (
            _description_change(items[idx], patch, observed_at=observed_at) == "rewrite"
        )
        dated = {**patch, _OBSERVED_AT: observed_at} if rewrites else patch
        patched[idx] = _patched(items[idx], dated)
    kept = [item for i, item in enumerate(patched) if i not in bucket_ops.deletes]
    kept.extend(_stored_items(bucket, bucket_ops.adds, observed_at))
    return _dedupe(kept, source="update")


def _dedupe(items: list[_Item], *, source: str) -> list[_Item]:
    """A new list without the items whose identity already appeared; a blank-description item is always kept."""
    kept: list[_Item] = []
    seen: set[str] = set()
    for item in items:
        key = _identity_key(item.description)
        if key is None:
            kept.append(item)
            continue
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)
    if dropped := len(items) - len(kept):
        logger.warning(
            "profile duplicates dropped: source=%s count=%d", source, dropped
        )
    return kept


def _identity_key(description: object) -> str | None:
    """An item's identity — its folded ``description`` — or None when the description is missing or blank."""
    if not isinstance(description, str):
        return None
    return _normalize(description) or None


def _log_rejected_op(op: dict[str, Any], reason: str) -> None:
    """Report a dropped op. ``data`` is never logged — it holds LLM-authored profile content."""
    logger.warning(
        "profile update op rejected: action=%s type=%s index=%s reason=%s",
        op.get("action"),
        op.get("type"),
        op.get("index"),
        reason,
    )


def _log_stale_op(
    op: dict[str, Any], *, item_observed_at: int, batch_observed_at: int
) -> None:
    """Report an op dropped as stale evidence, with the two dates that decided it."""
    logger.warning(
        "profile update op rejected: action=%s type=%s index=%s reason=stale evidence "
        "item_observed_at_ms=%d batch_observed_at_ms=%d",
        op.get("action"),
        op.get("type"),
        op.get("index"),
        item_observed_at,
        batch_observed_at,
    )


def _log_update_applied(
    collected: _CollectedOps,
    *,
    pass_name: str,
    observed_at: int,
    before: Profile,
    after: Profile,
) -> None:
    """Report one UPDATE pass: the reply's ops by action and by fate, and the profile's date and size after it."""
    landed_updates = sum(len(slot.updates) for slot in collected.slots.values())
    logger.info(
        "profile update applied: pass=%s observed_at_ms=%d add=%d update=%d delete=%d none=%d rejected=%d "
        "rewrote_description=%d history_appended=%d grounding_only=%d canonicalised=%d profile_timestamp_ms=%d->%d "
        "explicit=%d implicit=%d",
        pass_name,
        observed_at,
        collected.actions["add"],
        collected.actions["update"],
        collected.actions["delete"],
        collected.actions["none"],
        collected.rejected,
        collected.rewrote_description,
        collected.history_appended,
        landed_updates - collected.rewrote_description - collected.history_appended,
        sum(slot.canonicalised for slot in collected.slots.values()),
        before.timestamp,
        after.timestamp,
        len(after.explicit_info),
        len(after.implicit_traits),
    )


def _profile_from_buckets(
    *,
    owner_id: str,
    timestamp: int,
    explicit_info: Sequence[_Item],
    implicit_traits: Sequence[_Item],
) -> Profile:
    """A Profile holding the two buckets, its ``summary`` derived from them."""
    return Profile.model_validate(
        {
            "owner_id": owner_id,
            "summary": _build_summary(explicit_info, implicit_traits),
            "timestamp": timestamp,
            "explicit_info": explicit_info,
            "implicit_traits": implicit_traits,
        }
    )


def _build_summary(
    explicit_info: Sequence[_Item], implicit_traits: Sequence[_Item]
) -> str:
    """The first non-blank description, explicit_info first, a trait standing in for a blank trait description."""
    lines = (_summary_line(item) for item in (*explicit_info, *implicit_traits))
    return next((line for line in lines if line), "(no summary)")


def _summary_line(item: _Item) -> str:
    """The item's stripped description; an implicit trait with an empty one falls back to its label."""
    if isinstance(item, ImplicitTraitItem):
        return (item.description or item.trait or "").strip()
    return item.description.strip()
