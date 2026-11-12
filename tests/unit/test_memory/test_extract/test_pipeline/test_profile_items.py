"""Profile items: provenance, versions and history from an update's lineage.

What these pin (``everos.memory.extract.pipeline.profile_items``):

- an added item gets a fresh ``item_id``, the segments its operation named and
  their span, and the pictures it was read off; a kept item keeps its own;
- fresh evidence for an item keeps its id and adds the segments to its own;
- a fact (``explicit_info``) a ``modify`` gave a new value is a new item that
  names the old one (``supersedes``, ``since``), and the old one's history
  entry names it back (``superseded_by``); a re-punctuation and a rewritten
  implicit trait link nothing;
- a deprecation's history entry names its reason and its segments; a
  consolidation's names the items it merged into, or the reason it dropped;
- an item a consolidation wrote takes the union of what it came from, and is
  read off pictures only when every part of it was;
- the history bound keeps the values a current item still shows as replaced;
- a forgotten picture takes the items and the replaced values resting on it
  alone, and an undo puts them back once, history in its place;
- a forgotten conversation takes those every segment of which is forgotten,
  one at least its own, and never an item with no recorded segments;
- a person's edit is a new item linked to the old one, held, with the old
  one's provenance; a release changes the hold only;
- an update's reply never changes a held item: a new value, a retirement or
  a merge of it is undone, and the model's new value stays beside it, linked
  to nothing; a model-written item never inherits a person's marks.
"""

from __future__ import annotations

from typing import Any

from everalgo.types import ChatMessage, MemCell
from everalgo.types import Profile as AlgoProfile
from everalgo.user_memory.profile import ItemOrigin, ProfileUpdate, RetiredItem

from everos.memory.extract.pipeline.profile_items import (
    HISTORY_LIMIT,
    SOURCE_IMAGE_KEYS,
    bounded_history,
    edit_item,
    forget_images,
    forget_items,
    forget_segments,
    keep_held,
    restore_items,
    rests_on,
    stamp_update,
)
from everos.memory.mappers import profile as profile_mapper
from everos.memory.models import (
    ExplicitInfoItem,
    ImplicitTraitItem,
    Profile,
    replaced_chains,
)

_CAT = "1" * 64
_CITY = "2" * 64


def _cell(memcell_ts: int, *message_ts: int) -> MemCell:
    items = [
        ChatMessage(id=f"m{t}", role="user", sender_id="u1", content="x", timestamp=t)
        for t in message_ts or (memcell_ts,)
    ]
    return MemCell(items=items, timestamp=memcell_ts)


_CELLS = {"mc_2": _cell(50, 30, 50), "mc_3": _cell(70, 60, 70)}


def _item(description: str, **extra: object) -> ExplicitInfoItem:
    return ExplicitInfoItem(description=description, category="plans", **extra)


def _stored(description: str, item_id: str, **extra: object) -> ExplicitInfoItem:
    return _item(
        description,
        item_id=item_id,
        source_memcell_ids=["mc_1"],
        source_start=10,
        source_end=20,
        **extra,
    )


def _profile(*items: ExplicitInfoItem, traits: tuple[Any, ...] = ()) -> Profile:
    return Profile(
        owner_id="u1",
        summary="",
        timestamp=1,
        explicit_info=[*items],
        implicit_traits=[*traits],
    )


def _algo(*items: ExplicitInfoItem, traits: tuple[Any, ...] = ()) -> AlgoProfile:
    return profile_mapper.to_algo(_profile(*items, traits=traits))


def _extra(item: object) -> dict[str, Any]:
    return dict(getattr(item, "model_extra", None) or {})


def _stamp(old: Profile | None, update: ProfileUpdate) -> tuple[Profile, list[dict]]:
    return stamp_update(
        old, update, cells=_CELLS, replaced_at=70, primary_language="English"
    )


def test_an_added_item_gets_its_operations_sources_a_kept_one_keeps_its_own() -> None:
    old = _profile(_stored("Books the hotel", "pi_old"))
    update = ProfileUpdate(
        profile=_algo(old.explicit_info[0], _item("Has a cat named Biscuit")),
        explicit_info=(
            ItemOrigin(change="kept", from_items=(("explicit_info", 0),)),
            ItemOrigin(change="added", sources=("mc_2",), pictures=(_CAT,)),
        ),
        implicit_traits=(),
    )

    profile, replaced = _stamp(old, update)

    kept, added = (_extra(i) for i in profile.explicit_info)
    assert (kept["item_id"], kept["source_memcell_ids"]) == ("pi_old", ["mc_1"])
    assert added["item_id"].startswith("pi_") and added["item_id"] != "pi_old"
    assert added["source_memcell_ids"] == ["mc_2"]
    assert (added["source_start"], added["source_end"]) == (30, 50)
    assert added[SOURCE_IMAGE_KEYS] == [_CAT]
    assert replaced == []
    assert profile.primary_language == "English"


def test_fresh_evidence_keeps_the_id_and_adds_its_segments() -> None:
    old = _profile(_stored("Books the hotel", "pi_old"))
    patched = _item("Books the hotel", evidence="again", **_extra(old.explicit_info[0]))
    update = ProfileUpdate(
        profile=_algo(patched),
        explicit_info=(
            ItemOrigin(
                change="modified", from_items=(("explicit_info", 0),), sources=("mc_3",)
            ),
        ),
        implicit_traits=(),
    )

    profile, replaced = _stamp(old, update)

    extra = _extra(profile.explicit_info[0])
    assert extra["item_id"] == "pi_old"
    assert extra["source_memcell_ids"] == ["mc_1", "mc_3"]
    assert (extra["source_start"], extra["source_end"]) == (10, 70)
    assert replaced == []


def _new_value(
    old: Profile, description: str, *, bucket: str = "explicit_info"
) -> ProfileUpdate:
    """What everalgo returns for a modify of item 0: patched in place, keys kept."""
    stored = (old.explicit_info if bucket == "explicit_info" else old.implicit_traits)[
        0
    ]
    patched = type(stored).model_validate(
        {**stored.model_dump(), "description": description}
    )
    profile = (
        _algo(patched, *old.explicit_info[1:])
        if bucket == "explicit_info"
        else _algo(*old.explicit_info, traits=(patched,))
    )
    origin = ItemOrigin(change="modified", from_items=((bucket, 0),), sources=("mc_3",))
    kept = tuple(
        ItemOrigin(change="kept", from_items=(("explicit_info", i),))
        for i in range(1 if bucket == "explicit_info" else 0, len(old.explicit_info))
    )
    return ProfileUpdate(
        profile=profile,
        explicit_info=(origin, *kept) if bucket == "explicit_info" else kept,
        implicit_traits=(origin,) if bucket == "implicit_traits" else (),
        retired=(
            RetiredItem(
                ref=(bucket, 0),  # type: ignore[arg-type]
                item=profile_mapper.to_algo(old).explicit_info[0]
                if bucket == "explicit_info"
                else profile_mapper.to_algo(old).implicit_traits[0],
                change="modified",
                replaced_by=((bucket, 0),),  # type: ignore[arg-type]
            ),
        ),
    )


def test_a_new_value_names_the_item_it_replaced_and_the_history_names_it_back() -> None:
    old = _profile(_stored("Lives in Paris", "pi_paris"), _stored("Cycles", "pi_bike"))

    profile, replaced = _stamp(old, _new_value(old, "Lives in Berlin"))

    berlin, bike = (_extra(i) for i in profile.explicit_info)
    assert berlin["item_id"] not in ("pi_paris", "pi_bike")
    assert (berlin["supersedes"], berlin["since"]) == ("pi_paris", 70)
    assert berlin["source_memcell_ids"] == ["mc_3"]
    assert "supersedes" not in bike
    assert replaced == [
        {
            **old.explicit_info[0].model_dump(),
            "replaced_at": 70,
            "superseded_by": berlin["item_id"],
        }
    ]


def test_a_re_punctuation_and_a_rewritten_trait_link_nothing() -> None:
    old = _profile(_stored("Cycles to work", "pi_bike"))
    profile, replaced = _stamp(old, _new_value(old, "cycles to work."))
    assert "supersedes" not in _extra(profile.explicit_info[0])
    assert "superseded_by" not in replaced[0]

    trait = ImplicitTraitItem(
        description="Checks details", trait="careful", basis="b", item_id="pi_t"
    )
    old = _profile(traits=(trait,))
    profile, replaced = _stamp(
        old, _new_value(old, "Checks every detail in person", bucket="implicit_traits")
    )
    assert not {"supersedes", "since"} & set(_extra(profile.implicit_traits[0]))
    assert [("superseded_by" in r) for r in replaced] == [False]


def test_a_deprecation_records_its_reason_and_segments() -> None:
    old = _profile(_stored("Plays chess", "pi_chess"))
    update = ProfileUpdate(
        profile=_algo(),
        explicit_info=(),
        implicit_traits=(),
        retired=(
            RetiredItem(
                ref=("explicit_info", 0),
                item=profile_mapper.to_algo(old).explicit_info[0],
                change="deprecated",
                reason="gave up chess",
                sources=("mc_3",),
            ),
        ),
    )

    _, (entry,) = _stamp(old, update)

    assert entry["item_id"] == "pi_chess"
    assert (entry["reason"], entry["deprecated_by"], entry["replaced_at"]) == (
        "gave up chess",
        ["mc_3"],
        70,
    )


def test_a_consolidated_item_takes_the_union_of_what_it_came_from() -> None:
    old = _profile(
        _stored("Has a cat", "pi_cat", **{SOURCE_IMAGE_KEYS: [_CAT]}),
        _item(
            "Owns a cat named Biscuit",
            item_id="pi_cat2",
            source_memcell_ids=["mc_9"],
            source_start=5,
            source_end=8,
            **{SOURCE_IMAGE_KEYS: [_CITY]},
        ),
        _stored("Lives in Hangzhou", "pi_city"),
    )
    algo_old = profile_mapper.to_algo(old)
    merged = _item("Has a cat named Biscuit", consolidated_at=70)
    update = ProfileUpdate(
        profile=_algo(merged, old.explicit_info[2]),
        explicit_info=(
            ItemOrigin(
                change="consolidated",
                from_items=(("explicit_info", 0), ("explicit_info", 1)),
            ),
            ItemOrigin(change="kept", from_items=(("explicit_info", 2),)),
        ),
        implicit_traits=(),
        retired=tuple(
            RetiredItem(
                ref=("explicit_info", i),
                item=algo_old.explicit_info[i],
                change="consolidated",
                replaced_by=(("explicit_info", 0),),
            )
            for i in (0, 1)
        ),
    )

    profile, replaced = _stamp(old, update)

    extra = _extra(profile.explicit_info[0])
    assert extra["item_id"] not in ("pi_cat", "pi_cat2")
    assert extra["source_memcell_ids"] == ["mc_1", "mc_9"]
    assert (extra["source_start"], extra["source_end"]) == (5, 20)
    assert extra[SOURCE_IMAGE_KEYS] == [
        _CAT,
        _CITY,
    ]  # every part was read off a picture
    assert extra["consolidated_at"] == 70
    assert "supersedes" not in extra  # merging restatements is not a new value
    assert [r["merged_into"] for r in replaced] == [[extra["item_id"]]] * 2


def test_a_consolidated_item_with_a_text_part_is_no_longer_a_picture_item() -> None:
    old = _profile(
        _stored("Has a cat", "pi_cat", **{SOURCE_IMAGE_KEYS: [_CAT]}),
        _stored("Owns a cat", "pi_t"),
    )
    update = ProfileUpdate(
        profile=_algo(_item("Has a cat")),
        explicit_info=(
            ItemOrigin(
                change="consolidated",
                from_items=(("explicit_info", 0), ("explicit_info", 1)),
            ),
        ),
        implicit_traits=(),
    )

    profile, _ = _stamp(old, update)

    assert SOURCE_IMAGE_KEYS not in _extra(profile.explicit_info[0])


def test_a_consolidation_drop_records_its_reason() -> None:
    old = _profile(_stored("Is tired today", "pi_tired"))
    update = ProfileUpdate(
        profile=_algo(),
        explicit_info=(),
        implicit_traits=(),
        retired=(
            RetiredItem(
                ref=("explicit_info", 0),
                item=profile_mapper.to_algo(old).explicit_info[0],
                change="consolidated",
                reason="a passing state",
            ),
        ),
    )

    _, (entry,) = _stamp(old, update)

    assert entry["reason"] == "a passing state"
    assert "merged_into" not in entry


def test_an_item_stored_without_an_id_gets_one_and_keeps_it_unknown_sources() -> None:
    old = _profile(_item("Lives in Paris"))
    update = ProfileUpdate(
        profile=_algo(old.explicit_info[0]),
        explicit_info=(ItemOrigin(change="kept", from_items=(("explicit_info", 0),)),),
        implicit_traits=(),
    )

    profile, _ = _stamp(old, update)

    extra = _extra(profile.explicit_info[0])
    assert extra["item_id"].startswith("pi_")
    assert "source_memcell_ids" not in extra


def test_the_history_bound_keeps_the_values_still_shown() -> None:
    current = _profile(_stored("Lives in Lisbon", "pi_c", supersedes="pi_b", since=300))
    chain = [
        {"item_id": "pi_b", "description": "Lives in Berlin", "replaced_at": 300,
         "superseded_by": "pi_c", "supersedes": "pi_a"},
        {"item_id": "pi_a", "description": "Lives in Paris", "replaced_at": 200,
         "superseded_by": "pi_b", "supersedes": "pi_0"},
        {"item_id": "pi_0", "description": "Lives in Rome", "replaced_at": 100,
         "superseded_by": "pi_a"},
    ]  # fmt: skip
    churn = [
        {"item_id": f"pi_x{n}", "description": f"Fact {n}", "replaced_at": 400 + n}
        for n in range(HISTORY_LIMIT + 5)
    ]

    history = bounded_history(churn, chain, current)

    assert len(history) == HISTORY_LIMIT
    kept = {entry["item_id"] for entry in history}
    assert {"pi_b", "pi_a"} <= kept and "pi_0" not in kept
    (shown,) = replaced_chains([_extra(current.explicit_info[0])], history)
    assert [e["description"] for e in shown] == ["Lives in Berlin", "Lives in Paris"]
    assert bounded_history(churn[:3], chain, current) == [*churn[:3], *chain]


def _pictured(*descriptions: str, sources: list[str] | None = None) -> Profile:
    return Profile(
        owner_id="u1",
        summary="",
        timestamp=1,
        explicit_info=[
            ExplicitInfoItem(
                description=text,
                category="pets",
                observed_at=1,
                **({SOURCE_IMAGE_KEYS: sources} if sources is not None else {}),
            )
            for text in descriptions
        ],
    )


def test_a_forget_removes_an_item_only_when_every_picture_is_gone() -> None:
    profile, removed, count = forget_images(
        _pictured("Has a cat", sources=[_CAT, _CITY]), [_CAT]
    )

    assert (count, removed) == (0, [])
    assert [item.description for item in profile.explicit_info] == ["Has a cat"]  # type: ignore[union-attr]


def test_a_forget_removes_what_its_picture_alone_established_never_a_text_fact() -> (
    None
):
    profile, removed, count = forget_images(
        Profile(
            owner_id="u1",
            summary="",
            timestamp=1,
            explicit_info=[
                *_pictured("Has a cat", sources=[_CAT]).explicit_info,
                _item("Works in Python"),
            ],
        ),
        [_CAT],
    )

    assert count == 1
    assert [item.description for item in profile.explicit_info] == ["Works in Python"]  # type: ignore[union-attr]
    assert removed[0][0] == "explicit_info"
    assert forget_images(None, [_CAT]) == (None, [], 0)


def test_an_undo_puts_the_item_back_whole_and_once() -> None:
    _gone, removed, _ = forget_images(
        _pictured("Has a cat named Biscuit", sources=[_CAT]), [_CAT]
    )

    profile, restored, replaced = restore_items(
        None, removed, owner_id="u1", timestamp=1
    )
    assert (restored, replaced) == (1, 0)
    assert _extra(profile.explicit_info[0])[SOURCE_IMAGE_KEYS] == [_CAT]

    again, restored, replaced = restore_items(
        profile, removed, owner_id="u1", timestamp=1
    )
    assert (restored, replaced) == (0, 0)
    assert len(again.explicit_info) == 1


def _replaced(description: str, replaced_at: int, **extra: object) -> dict[str, Any]:
    return {"description": description, "replaced_at": replaced_at, **extra}


def test_a_forget_takes_a_replaced_value_its_picture_alone_established() -> None:
    profile = _profile(_item("Works in Python"))
    profile.history = [
        _replaced("Had a black cat", 30, **{SOURCE_IMAGE_KEYS: [_CAT]}),
        _replaced("Lived in Paris", 20, source_memcell_ids=["mc_1"]),
        _replaced("Had a grey cat", 10, **{SOURCE_IMAGE_KEYS: [_CAT, _CITY]}),
    ]

    after, removed, count = forget_images(profile, [_CAT])

    assert after is not None
    assert count == 0  # a replaced value is history, not an item
    assert [entry["description"] for entry in after.history] == [
        "Lived in Paris",
        "Had a grey cat",
    ]
    assert removed == [("history", profile.history[0])]
    assert [i.description for i in after.explicit_info] == ["Works in Python"]


def test_an_undo_puts_a_replaced_value_back_in_its_place_once() -> None:
    profile = _profile(_item("Works in Python"))
    profile.history = [
        _replaced("Lived in Lyon", 30),
        _replaced("Had a black cat", 20, **{SOURCE_IMAGE_KEYS: [_CAT]}),
        _replaced("Lived in Paris", 10),
    ]
    after, removed, _ = forget_images(profile, [_CAT])

    back, restored, replaced = restore_items(after, removed, owner_id="u1", timestamp=1)
    assert (restored, replaced) == (0, 1)
    assert [entry["description"] for entry in back.history] == [
        "Lived in Lyon",
        "Had a black cat",
        "Lived in Paris",
    ]

    again, _, replaced = restore_items(back, removed, owner_id="u1", timestamp=1)
    assert replaced == 0
    assert len(again.history) == 3


def _sourced(description: str, *sources: str) -> ExplicitInfoItem:
    return _item(description, source_memcell_ids=list(sources))


def test_a_forgotten_conversation_takes_what_no_remembered_segment_supports() -> None:
    profile = _profile(
        _sourced("Lives in Paris", "mc_1"),
        _sourced("Works in Python", "mc_1", "mc_9"),  # mc_9 is remembered
        _sourced("Has a cat", "mc_1", "mc_5"),  # mc_5 an earlier forget's
        _sourced("Plays chess", "mc_5"),  # the earlier forget's to take
        _item("Likes tea"),  # stored before items had segments
    )
    profile.history = [
        _replaced("Lived in Lyon", 30, source_memcell_ids=["mc_1"]),
        _replaced("Lived in Rome", 20, source_memcell_ids=["mc_9"]),
    ]

    after, removed, count = forget_segments(profile, ["mc_1", "mc_2"], already=["mc_5"])

    assert after is not None
    assert [i.description for i in after.explicit_info] == [
        "Works in Python",
        "Plays chess",
        "Likes tea",
    ]
    assert count == 2
    assert [(bucket, entry["description"]) for bucket, entry in removed] == [
        ("explicit_info", "Lives in Paris"),
        ("explicit_info", "Has a cat"),
        ("history", "Lived in Lyon"),
    ]
    assert [entry["description"] for entry in after.history] == ["Lived in Rome"]
    assert forget_segments(None, ["mc_1"]) == (None, [], 0)
    assert forget_segments(after, ["mc_7"]) == (after, [], 0)


def test_rests_on_names_a_removed_entry_by_its_segments() -> None:
    assert rests_on({"source_memcell_ids": ["mc_1", "mc_2"]}, ["mc_2"])
    assert not rests_on({"source_memcell_ids": ["mc_1"]}, ["mc_2"])
    assert not rests_on({"description": "no segments"}, ["mc_2"])


# ── a person's edit, and what an update does to it ───────────────────────


def test_a_persons_edit_is_a_held_new_value_with_the_old_provenance() -> None:
    old = _profile(_stored("Lives in Paris", "pi_paris"))

    profile, history, new_id = edit_item(
        old,
        [],
        "pi_paris",
        description="Lives in Berlin",
        hold=True,
        edited_by="u1",
        at=90,
    )

    item = _extra(profile.explicit_info[0])
    assert profile.explicit_info[0].description == "Lives in Berlin"
    assert new_id == item["item_id"] != "pi_paris"
    assert (item["supersedes"], item["since"], item["held"]) == ("pi_paris", 90, True)
    assert (item["edited_by"], item["edited_at"]) == ("u1", 90)
    assert item["source_memcell_ids"] == ["mc_1"]
    assert history[0]["superseded_by"] == new_id and history[0]["replaced_at"] == 90
    assert [c[0]["item_id"] for c in replaced_chains([item], history)] == ["pi_paris"]


def test_a_release_changes_the_hold_only() -> None:
    held = _profile(_stored("Lives in Paris", "pi_paris", held=True, edited_by="u1"))

    profile, history, item_id = edit_item(
        held, [], "pi_paris", description=None, hold=False, edited_by="u1", at=95
    )

    assert item_id == "pi_paris" and history == []
    assert _extra(profile.explicit_info[0])["held"] is False


def test_an_update_never_changes_a_held_item() -> None:
    old = _profile(
        _stored("Lives in Paris", "pi_paris", held=True, edited_by="u1"),
        _stored("Cycles", "pi_bike"),
    )

    profile, replaced = _stamp(old, _new_value(old, "Lives in Berlin"))
    profile, replaced, kept = keep_held(old, profile, replaced)

    paris, berlin, bike = profile.explicit_info
    assert paris.description == "Lives in Paris" and _extra(paris)["held"] is True
    assert berlin.description == "Lives in Berlin"
    assert "supersedes" not in _extra(berlin) and "since" not in _extra(berlin)
    assert "held" not in _extra(berlin)  # a model's item carries no edit marks
    assert _extra(bike)["item_id"] == "pi_bike"
    assert replaced == [] and kept == ["pi_paris"]


def test_a_retired_or_merged_held_item_comes_back() -> None:
    old = _profile(
        _stored("Lives in Paris", "pi_paris", held=True), _stored("Cycles", "pi_bike")
    )
    update = ProfileUpdate(
        profile=_algo(old.explicit_info[1]),
        explicit_info=(ItemOrigin(change="kept", from_items=(("explicit_info", 1),)),),
        implicit_traits=(),
        retired=(
            RetiredItem(
                ref=("explicit_info", 0),
                item=profile_mapper.to_algo(old).explicit_info[0],
                change="deprecated",
                sources=("mc_3",),
                reason="moved",
            ),
        ),
    )

    profile, replaced = _stamp(old, update)
    profile, replaced, kept = keep_held(old, profile, replaced)

    assert [i.description for i in profile.explicit_info] == [
        "Cycles",
        "Lives in Paris",
    ]
    assert replaced == [] and kept == ["pi_paris"]


def test_a_held_item_the_update_left_alone_is_not_reported() -> None:
    old = _profile(_stored("Lives in Paris", "pi_paris", held=True))
    update = ProfileUpdate(
        profile=_algo(old.explicit_info[0]),
        explicit_info=(ItemOrigin(change="kept", from_items=(("explicit_info", 0),)),),
        implicit_traits=(),
    )

    profile, replaced = _stamp(old, update)
    assert keep_held(old, profile, replaced)[2] == []


def test_a_forget_by_id_takes_the_item_or_the_replaced_value() -> None:
    profile = _profile(_stored("Lives in Berlin", "pi_berlin"))
    profile.history = [{"item_id": "pi_paris", "description": "Lives in Paris"}]

    kept, removed, count = forget_items(profile, ["pi_paris"])
    assert kept is not None and count == 0
    assert removed == [
        ("history", {"item_id": "pi_paris", "description": "Lives in Paris"})
    ]
    kept, removed, count = forget_items(profile, ["pi_berlin"])
    assert kept is not None and kept.explicit_info == [] and count == 1
