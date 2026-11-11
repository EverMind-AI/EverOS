"""Provenance on member-in-group profile items (docs/group-memory.md, 7.4).

What these pin:

- an item an update added gets a fresh ``item_id``, the update's segments and
  their span; an item it kept keeps its stored id and sources;
- an item the update no longer holds is returned for the history, stamped
  with the update's time;
- an item an UPDATE op rewrote (it still carries the old ``item_id``, as
  everalgo leaves a host's keys on a patched item) names the item it replaced
  and the update's time, and the old item's history entry names it back;
- a rewrite that carries no old id (COMPACT, REGROUP), a dedupe into an item
  already on file, a re-punctuation and a rewritten implicit trait link
  nothing, and a kept item keeps its own link and no other;
- the history bound keeps the values a current item still shows as replaced;
- ``own_segments`` keeps only the segments with a message of the member's own.
"""

from __future__ import annotations

from typing import Any

from everalgo.types import ChatMessage, MemCell

from everos.memory.extract.pipeline.member_profile import (
    HISTORY_LIMIT,
    bounded_history,
    own_segments,
    with_provenance,
)
from everos.memory.models import (
    ExplicitInfoItem,
    ImplicitTraitItem,
    Profile,
    replaced_chains,
)


def _item(description: str, **extra: object) -> ExplicitInfoItem:
    return ExplicitInfoItem(description=description, category="plans", **extra)


def test_kept_items_keep_their_provenance_new_ones_get_the_updates() -> None:
    old = Profile(
        owner_id="u_betty",
        summary="",
        timestamp=1,
        explicit_info=[
            _item(
                "Books the hotel",
                item_id="pi_old",
                source_memcell_ids=["mc_1"],
                source_start=10,
                source_end=20,
            ),
            _item(
                "Undecided on dates",
                item_id="pi_gone",
                source_memcell_ids=["mc_1"],
                source_start=10,
                source_end=20,
            ),
        ],
    )
    new = Profile(
        owner_id="u_betty",
        summary="",
        timestamp=2,
        explicit_info=[_item("books the  HOTEL"), _item("Flies on April 30")],
    )
    stamped, replaced = with_provenance(
        old, new, sources=["mc_2", "mc_3"], span=(30, 50), replaced_at=50
    )
    kept, added = stamped.explicit_info
    assert (kept.model_extra or {})["item_id"] == "pi_old"
    assert (kept.model_extra or {})["source_memcell_ids"] == ["mc_1"]
    extra = added.model_extra or {}
    assert extra["item_id"].startswith("pi_") and extra["item_id"] != "pi_old"
    assert extra["source_memcell_ids"] == ["mc_2", "mc_3"]
    assert (extra["source_start"], extra["source_end"]) == (30, 50)
    assert [(r["description"], r["replaced_at"]) for r in replaced] == [
        ("Undecided on dates", 50)
    ]


def test_only_segments_with_the_members_own_messages_are_read() -> None:
    def cell(*senders: tuple[str, str]) -> MemCell:
        items = [
            ChatMessage(
                id=f"m{i}", role=role, sender_id=s, content="x", timestamp=i + 1
            )
            for i, (s, role) in enumerate(senders)
        ]
        return MemCell(items=items, timestamp=len(items))

    mine = cell(("u_betty", "user"), ("u_chen", "user"))
    others = cell(("u_chen", "user"), ("bot_ava", "assistant"))
    assert own_segments([mine, others], "u_betty") == [mine]


def _stored(description: str, item_id: str, **extra: object) -> ExplicitInfoItem:
    return _item(
        description,
        item_id=item_id,
        source_memcell_ids=["mc_1"],
        source_start=10,
        source_end=20,
        **extra,
    )


def _profile(*items: ExplicitInfoItem) -> Profile:
    return Profile(owner_id="u_betty", summary="", timestamp=1, explicit_info=[*items])


def _extra(item: ExplicitInfoItem) -> dict[str, Any]:
    return dict(item.model_extra or {})


def test_a_value_rewritten_by_its_index_names_the_item_it_replaced() -> None:
    old = _profile(_stored("Lives in Paris", "pi_paris"), _stored("Cycles", "pi_bike"))
    # What everalgo returns for {"action": "update", "index": 0}: the patched
    # item keeps every host key of the stored one, the old id included.
    rewritten = _item("Lives in Berlin", **_extra(old.explicit_info[0]))
    new = _profile(rewritten, old.explicit_info[1])

    stamped, replaced = with_provenance(
        old, new, sources=["mc_2"], span=(30, 50), replaced_at=50
    )

    berlin, bike = stamped.explicit_info
    extra = _extra(berlin)
    assert extra["item_id"] not in ("pi_paris", "pi_bike")
    assert (extra["supersedes"], extra["since"]) == ("pi_paris", 50)
    assert extra["source_memcell_ids"] == ["mc_2"]
    assert "supersedes" not in _extra(bike) and "since" not in _extra(bike)
    assert replaced == [
        {
            **old.explicit_info[0].model_dump(),
            "replaced_at": 50,
            "superseded_by": extra["item_id"],
        }
    ]


def test_rewrites_that_are_not_new_values_link_nothing() -> None:
    old = _profile(
        _stored("Lives in Paris", "pi_paris"),
        _stored("Cycles to work", "pi_bike"),
        _stored("Owns a cat", "pi_cat"),
        _stored("Plays chess", "pi_chess"),
    )
    new = _profile(
        # COMPACT / REGROUP: a new item without any host key.
        _item("Lives in Paris, France"),
        # A rewrite that only re-punctuates its item.
        _item("Cycles to work.", **_extra(old.explicit_info[1])),
        # A dedupe: the cat item rewritten into the chess one, which everalgo's
        # dedupe then drops as the later copy.
        _item("Plays chess", **_extra(old.explicit_info[2])),
    )

    stamped, replaced = with_provenance(
        old, new, sources=["mc_2"], span=(30, 50), replaced_at=50
    )

    for item in stamped.explicit_info:
        assert "supersedes" not in _extra(item) and "since" not in _extra(item)
    chess = stamped.explicit_info[2]
    assert _extra(chess)["item_id"] == "pi_chess"  # the item on file, kept
    assert {r["item_id"] for r in replaced} == {"pi_paris", "pi_bike", "pi_cat"}
    assert all("superseded_by" not in r for r in replaced)


def test_a_rewritten_trait_links_nothing() -> None:
    """A trait is an inference: its rewrite sharpens it, it replaces no value."""
    kept = {"item_id": "pi_trait", "source_memcell_ids": ["mc_1"]}
    trait = ImplicitTraitItem(
        description="Checks details", trait="careful", basis="b", **kept
    )
    old = Profile(owner_id="u_betty", summary="", timestamp=1, implicit_traits=[trait])
    sharper = ImplicitTraitItem(
        description="Checks every detail in person", trait="careful", basis="b", **kept
    )
    new = Profile(
        owner_id="u_betty", summary="", timestamp=2, implicit_traits=[sharper]
    )

    stamped, replaced = with_provenance(
        old, new, sources=["mc_2"], span=(30, 50), replaced_at=50
    )

    (item,) = stamped.implicit_traits
    assert not {"supersedes", "since"} & set(item.model_extra or {})
    assert [("superseded_by" in r) for r in replaced] == [False]


def test_a_kept_item_keeps_its_own_link_and_takes_no_other() -> None:
    old = _profile(
        _stored("Lives in Berlin", "pi_berlin", supersedes="pi_paris", since=20),
        _stored("Owns a cat", "pi_cat", supersedes="pi_dog", since=15),
        _stored("Plays chess", "pi_chess"),
    )
    new = _profile(
        # Passed through a REGROUP unchanged: the host keys are gone.
        _item("Lives in Berlin"),
        # The cat item rewritten into the chess one: the dedupe keeps the
        # chess item, so the cat item's link must not ride along onto it.
        _item("plays  CHESS", **_extra(old.explicit_info[1])),
    )

    stamped, _ = with_provenance(
        old, new, sources=["mc_3"], span=(60, 70), replaced_at=70
    )

    berlin, chess = (_extra(i) for i in stamped.explicit_info)
    assert berlin["item_id"] == "pi_berlin"
    assert (berlin["supersedes"], berlin["since"]) == ("pi_paris", 20)
    assert berlin["source_memcell_ids"] == ["mc_1"]
    assert chess["item_id"] == "pi_chess"
    assert "supersedes" not in chess and "since" not in chess


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
    # The two values a read shows for the current item survive the churn;
    # the third one back is past what a read shows and drops off.
    assert {"pi_b", "pi_a"} <= kept and "pi_0" not in kept
    assert [e["item_id"] for e in history[: HISTORY_LIMIT - 2]] == [
        f"pi_x{n}" for n in range(HISTORY_LIMIT - 2)
    ]
    (shown,) = replaced_chains([_extra(current.explicit_info[0])], history)
    assert [e["description"] for e in shown] == ["Lives in Berlin", "Lives in Paris"]
    # Under the bound nothing is dropped.
    assert bounded_history(churn[:3], chain, current) == [*churn[:3], *chain]
