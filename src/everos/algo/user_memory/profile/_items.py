"""Stored-item vocabulary shared by rendering, merging and the extractor.

The item union, the label field of each bucket, the caps, label matching and dating.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everos.algo.types import ExplicitInfoItem, ImplicitTraitItem, Profile

if TYPE_CHECKING:
    from collections.abc import Sequence

_Item = ExplicitInfoItem | ImplicitTraitItem

# The field naming each bucket's group: a key many atomic items share, never a unique key.
_LABEL_FIELDS = {"explicit_info": "category", "implicit_traits": "trait"}
_PROFILE_MAX_ITEMS = 60
_PROFILE_MAX_PER_CATEGORY = 8
# In ascii_width units (East Asian wide characters count 2): a guard against runaway concatenation, not a style rule.
_ITEM_WIDTH_BACKSTOP = 250
_OBSERVED_AT = "observed_at"


def _dated_buckets(profile: Profile) -> tuple[list[_Item], list[_Item]]:
    """The profile's two buckets with every item dated, exactly as both the renderers and ``_apply_ops`` see them."""
    return (
        _fill_observed_at(profile.explicit_info, profile.timestamp),
        _fill_observed_at(profile.implicit_traits, profile.timestamp),
    )


def _fill_observed_at(items: Sequence[_Item], observed_at: int) -> list[_Item]:
    """The items, each one without a date given ``observed_at`` — a profile's own date for items stored undated."""
    return [
        item
        if item.observed_at is not None
        else item.model_copy(update={_OBSERVED_AT: observed_at})
        for item in items
    ]


def _label(item: _Item) -> str | None:
    """The item's group label as stored: ``category`` for explicit_info, ``trait`` for implicit_traits."""
    return item.category if isinstance(item, ExplicitInfoItem) else item.trait


def _item_label(item: _Item) -> str | None:
    """The item's stripped label, or None when it carries no usable one."""
    label = _label(item)
    return label.strip() if label is not None and label.strip() else None


def _in_group(item: _Item, wanted: str) -> bool:
    """True when the item's label matches the already-normalised ``wanted``."""
    label = _item_label(item)
    return label is not None and _normalize(label) == wanted


def _normalize(text: str) -> str:
    """Fold whitespace runs and case, so a restated item still matches the copy already stored."""
    return " ".join(text.split()).casefold()
