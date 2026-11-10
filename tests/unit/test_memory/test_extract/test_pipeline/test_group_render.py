"""Group rendering and attribution (docs/group-memory.md, section 7.2).

What these pin -- the group information that reaches the extraction model is
data, never new prompt text, and the attribution fields come from code:

- speakers are labelled by their roster ``display_name`` at the message's
  time, an assistant member as ``"<display_name> (assistant)"``;
- the roster block names the group, the segment's speakers and the members
  it names (with role and title), the assistants, and the attribution rule,
  and is appended after the operator's custom instructions;
- ``sender_ids`` are the segment's senders; ``subject_ids`` the human members
  the record names (a former member included, a later joiner never), falling
  back to the human senders; ``origin`` is ``member`` / ``assistant`` /
  ``mixed`` from the senders' roster kind;
- the block's text is versioned: its fingerprint pins the template.
"""

from __future__ import annotations

from everalgo.types import ChatMessage, MemCell

from everos.memory.extract.pipeline.group_render import (
    ROSTER_BLOCK,
    attribution,
    label_cell,
    roster_block,
    with_roster_block,
)
from everos.memory.models import Group, Roster, RosterEvent, fold_roster

_GROUP = Group(group_id="trip", name="Trip team", created_at=0)


def _roster() -> Roster:
    return Roster(
        intervals=fold_roster(
            [
                RosterEvent(
                    op="join",
                    member_id="u_chen",
                    at=0,
                    display_name="Chen",
                    role="owner",
                    title="Organiser",
                ),
                RosterEvent(op="join", member_id="u_betty", at=0, display_name="Betty"),
                RosterEvent(
                    op="join",
                    member_id="bot_ava",
                    at=0,
                    kind="assistant",
                    role="assistant",
                    display_name="Ava",
                ),
                RosterEvent(op="join", member_id="u_dan", at=0, display_name="Dan"),
                RosterEvent(op="leave", member_id="u_dan", at=50),
                RosterEvent(op="join", member_id="u_evan", at=900, display_name="Evan"),
            ]
        )
    )


def _message(sender: str, at: int, text: str, role: str = "user") -> ChatMessage:
    return ChatMessage(
        id=f"m{at}",
        role=role,
        sender_id=sender,
        sender_name=sender,
        content=text,
        timestamp=at,
    )


def _cell(*messages: ChatMessage) -> MemCell:
    return MemCell(items=list(messages), timestamp=messages[-1].timestamp)


def test_speakers_are_labelled_by_their_roster_names() -> None:
    cell = _cell(
        _message("u_chen", 100, "Flights are sold out"),
        _message("bot_ava", 110, "Two flights left", role="assistant"),
    )
    labelled = label_cell(cell, _roster())
    assert [m.sender_name for m in labelled.items] == ["Chen", "Ava (assistant)"]
    # The cell itself is not touched: the memcell row keeps what was sent.
    assert [m.sender_name for m in cell.items] == ["u_chen", "bot_ava"]


def test_the_roster_block_lists_speakers_named_members_and_assistants() -> None:
    cell = _cell(
        _message("u_chen", 100, "Betty, can you book the hotel?"),
        _message("bot_ava", 110, "Two flights left", role="assistant"),
    )
    block = roster_block(_GROUP, _roster(), cell)
    first, rule = block.split("\n")
    assert first == (
        'This conversation is a group chat in "Trip team". People in this part: '
        'Chen (owner, Organiser), Betty (member). "Ava (assistant)" is the '
        "group's AI assistant."
    )
    assert rule.startswith("Attribute every statement to the person who made it.")
    assert with_roster_block("Write in English.", block) == (
        f"Write in English.\n\n{block}"
    )
    assert with_roster_block(None, block) == block


def test_attribution_of_a_mixed_segment() -> None:
    cell = _cell(
        _message("u_chen", 100, "May 1 is sold out"),
        _message("u_betty", 120, "Fine by me"),
        _message("bot_ava", 130, "Two April 30 flights", role="assistant"),
    )
    found = attribution(
        cell, _roster(), texts=("Departure moved", "Chen moved it; Betty agreed.")
    )
    assert found.sender_ids == ["u_chen", "u_betty", "bot_ava"]
    assert found.subject_ids == ["u_chen", "u_betty"]
    assert found.origin == "mixed"
    assert (found.source_start, found.source_end) == (100, 130)


def test_an_assistant_only_segment_has_assistant_origin_and_no_subject() -> None:
    cell = _cell(_message("bot_ava", 200, "Reminder: check in", role="assistant"))
    found = attribution(cell, _roster(), texts=("Reminder", "Ava posted a reminder."))
    assert found.origin == "assistant"
    assert found.subject_ids == []
    assert found.sender_ids == ["bot_ava"]


def test_a_former_member_named_in_the_text_is_a_subject_a_later_one_is_not() -> None:
    cell = _cell(_message("u_chen", 300, "Dan left his charger here"))
    found = attribution(
        cell, _roster(), texts=("Charger", "Chen said Dan and Evan left chargers.")
    )
    # Dan left at 50 but is a former member; Evan joins only at 900.
    assert found.subject_ids == ["u_chen", "u_dan"]
    assert found.origin == "member"


def test_without_a_named_member_the_subjects_are_the_human_senders() -> None:
    cell = _cell(
        _message("u_betty", 400, "ok"),
        _message("bot_ava", 410, "noted", role="assistant"),
    )
    found = attribution(cell, _roster(), texts=("Ack", "Someone acknowledged."))
    assert found.subject_ids == ["u_betty"]


def test_the_roster_block_is_a_versioned_template() -> None:
    assert ROSTER_BLOCK.id == "chat.group_roster_block"
    assert ROSTER_BLOCK.version == 1
    # Pinned: a change to the block's text is a new version, on purpose.
    assert ROSTER_BLOCK.fingerprint == "a534cf05c899a6a4"
