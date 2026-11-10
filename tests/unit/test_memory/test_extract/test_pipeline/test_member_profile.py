"""Provenance on member-in-group profile items (docs/group-memory.md, 7.4).

What these pin:

- an item an update added gets a fresh ``item_id``, the update's segments and
  their span; an item it kept keeps its stored id and sources;
- an item the update no longer holds is returned for the history, stamped
  with the update's time;
- ``own_segments`` keeps only the segments with a message of the member's own.
"""

from __future__ import annotations

from everalgo.types import ChatMessage, MemCell

from everos.memory.extract.pipeline.member_profile import (
    own_segments,
    with_provenance,
)
from everos.memory.models import ExplicitInfoItem, Profile


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
