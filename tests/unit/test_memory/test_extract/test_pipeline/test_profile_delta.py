"""The profile half of the picture call: what a photo says about its sender.

What these pin:

- The extraction reply's profile operations use everalgo's UPDATE shape and are
  add-only: this call never sees the stored profile, so it cannot address an
  index, and a guessed delete on a profile is the worst failure this path has.
- A merged item is stamped with the observation time and records the pictures it
  was read from, which is what lets a forget reach it later.
- Facts already on file are not duplicated, and the same identity rule everalgo's
  merge uses (case- and space-folded description) is the one applied here.
- A forget removes an item only when every picture behind it is gone; a
  text-derived item carries no picture source and is never touched.
- An undo puts a removed item back once, and does not duplicate a fact that a
  later conversation re-established.
"""

from __future__ import annotations

from everos.memory.extract.pipeline.profile_delta import (
    PICTURE_PROFILE_RULES,
    apply_ops,
    forget_images,
    normalised,
    restore_items,
    retarget,
)
from everos.memory.models import ExplicitInfoItem, Profile

_TS = 1_747_216_800_000
_CAT = "1" * 64
_CITY = "2" * 64


def _add(
    description: str,
    *,
    category: str = "pets",
    source: str = "",
    action: str = "add",
) -> dict:
    op: dict = {
        "action": action,
        "type": "explicit_info",
        "data": {"category": category, "description": description, "evidence": "x"},
    }
    if source:
        op["source_image_id"] = source
    return op


def _profile(*descriptions: str, sources: list[str] | None = None) -> Profile:
    return Profile(
        owner_id="u1",
        summary="",
        timestamp=_TS,
        explicit_info=[
            ExplicitInfoItem(
                description=text,
                category="pets",
                evidence="x",
                observed_at=_TS,
                **({"source_image_ids": sources or []} if sources is not None else {}),
            )
            for text in descriptions
        ],
    )


def test_the_rules_say_only_add_is_available() -> None:
    """The prompt must not invite an operation the reply model cannot accept."""
    folded = " ".join(PICTURE_PROFILE_RULES.split())
    assert 'only "add" is available' in folded
    assert "no stored profile" in folded


def test_an_added_item_carries_its_time_and_pictures() -> None:
    profile, added = apply_ops(
        None,
        [_add("Has a cat named Biscuit", source=_CAT)],
        owner_id="u1",
        observed_at=_TS,
        image_ids=[_CAT],
    )

    assert added == 1
    item = profile.explicit_info[0]
    assert item.description == "Has a cat named Biscuit"
    assert item.observed_at == _TS
    assert item.model_extra["source_image_ids"] == [_CAT]


def test_a_fact_already_on_file_is_not_added_twice() -> None:
    existing = _profile("Has a cat named Biscuit")

    profile, added = apply_ops(
        existing,
        [_add("  has a CAT named biscuit ")],
        owner_id="u1",
        observed_at=_TS,
        image_ids=[_CAT],
    )

    assert added == 0
    assert len(profile.explicit_info) == 1


def test_normalisation_is_case_and_space_folding() -> None:
    assert normalised("  Has  a Cat ") == normalised("has a cat")


def test_a_none_operation_changes_nothing() -> None:
    profile, added = apply_ops(
        _profile("Has a cat"),
        [{"action": "none"}],
        owner_id="u1",
        observed_at=_TS,
        image_ids=[_CAT],
    )

    assert added == 0
    assert len(profile.explicit_info) == 1


def test_an_unnamed_picture_still_contributes_its_item() -> None:
    """Without a source the item is valid, it just cannot be forgotten by picture."""
    profile, added = apply_ops(
        None,
        [_add("Lives in Hangzhou")],
        owner_id="u1",
        observed_at=_TS,
        image_ids=[],
    )

    assert added == 1
    assert profile.explicit_info[0].model_extra["source_image_ids"] == []


def test_a_label_is_retargeted_to_the_picture_hash() -> None:
    ops = retarget([_add("Has a cat", source="img-1")], {"img-1": _CAT})

    assert ops[0].source_image_id == _CAT


def test_an_unknown_label_becomes_untraceable_rather_than_wrong() -> None:
    """A label the cell does not define must not become a hash that means nothing."""
    ops = retarget([_add("Has a cat", source="img-9")], {"img-1": _CAT})

    assert ops[0].source_image_id == ""


def test_a_forget_removes_an_item_only_when_every_picture_is_gone() -> None:
    profile, removed, count = forget_images(
        _profile("Has a cat", "Lives in Hangzhou", sources=[_CAT, _CITY]),
        [_CAT],
    )

    # "Has a cat" had two pictures; one is left, so it stays.
    assert count == 0
    assert [item.description for item in profile.explicit_info] == [
        "Has a cat",
        "Lives in Hangzhou",
    ]
    assert removed == []


def test_a_forget_removes_the_item_its_picture_alone_established() -> None:
    profile, removed, count = forget_images(
        _profile("Has a cat", sources=[_CAT]),
        [_CAT],
    )

    assert count == 1
    assert profile.explicit_info == []
    assert removed[0][0] == "explicit_info"


def test_a_text_derived_item_is_never_touched_by_a_forget() -> None:
    """No picture source means the fact came from the conversation, not a photo."""
    profile, removed, count = forget_images(_profile("Works in Python"), [_CAT])

    assert count == 0
    assert removed == []
    assert len(profile.explicit_info) == 1


def test_a_forget_with_no_profile_is_a_no_op() -> None:
    profile, removed, count = forget_images(None, [_CAT])

    assert profile is None
    assert removed == []
    assert count == 0


def test_an_undo_puts_the_item_back_whole() -> None:
    original = _profile("Has a cat named Biscuit", sources=[_CAT])
    _gone, removed, _ = forget_images(original, [_CAT])

    profile, restored = restore_items(
        Profile(owner_id="u1", summary="", timestamp=_TS),
        removed,
        owner_id="u1",
        timestamp=_TS,
    )

    assert restored == 1
    item = profile.explicit_info[0]
    assert item.description == "Has a cat named Biscuit"
    assert item.observed_at == _TS
    assert item.model_extra["source_image_ids"] == [_CAT]


def test_an_undo_does_not_duplicate_a_fact_re_established_meanwhile() -> None:
    _gone, removed, _ = forget_images(_profile("Has a cat", sources=[_CAT]), [_CAT])
    current = _profile("Has a cat")

    profile, restored = restore_items(current, removed, owner_id="u1", timestamp=_TS)

    assert restored == 0
    assert len(profile.explicit_info) == 1
