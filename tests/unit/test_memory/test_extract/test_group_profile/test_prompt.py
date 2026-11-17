"""The group-profile update prompt, version 4 (docs/group-memory.md, 7.5).

What these pin:

- a listed topic shows its summary cut to its first ``TOPIC_SUMMARY_SHOWN``
  characters, the cut marked with an ellipsis; a summary that fits is shown
  whole;
- the system prompt asks for summaries of at most ``TOPIC_SUMMARY_MAX``
  characters and for a ``replaces`` on every ``decision.new``, and shows each
  operation in a shape of its own fields -- not the one shape with every
  field of version 3.
"""

from __future__ import annotations

from everos.memory.extract.group_profile import GroupProfileView, build_messages
from everos.memory.extract.group_profile.prompt import (
    TOPIC_SUMMARY_MAX,
    TOPIC_SUMMARY_SHOWN,
    render_profile,
)
from everos.memory.models import GroupProfileItem, ProfileSource, Roster


def _topic(topic_id: str, summary: str) -> GroupProfileItem:
    return GroupProfileItem(
        owner_id="trip",
        kind="topic",
        topic_id=topic_id,
        name="Travel",
        summary=summary,
        topic_status="exploring",
        sources=[
            ProfileSource(
                memcell_id="mc_0",
                message_id="m-1",
                speaker_id="u_chen",
                at=1,
                quote="q",
            )
        ],
        source_start=1,
        source_end=1,
        observed_at=1,
        item_id=f"gp_{topic_id}",
    )


def test_a_listed_topic_shows_the_start_of_its_summary() -> None:
    long = "a" * (TOPIC_SUMMARY_SHOWN - 2) + " and then the rest of the thread"
    fits = "b" * TOPIC_SUMMARY_SHOWN

    text = render_profile(
        GroupProfileView(topics=[_topic("tp_long", long), _topic("tp_fits", fits)])
    )

    cut = long[:TOPIC_SUMMARY_SHOWN].rstrip()
    assert f"- [tp_long] Travel (exploring): {cut}…\n" in text
    assert f"- [tp_fits] Travel (exploring): {fits}\n" in text
    assert long not in text


def test_the_reply_asks_for_short_summaries_replaces_and_no_empty_fields() -> None:
    system = build_messages(
        roster=Roster(), profile=GroupProfileView(), segments=[], language=None
    )[0].content

    assert (TOPIC_SUMMARY_MAX, TOPIC_SUMMARY_SHOWN) == (300, 200)
    assert f"in at most {TOPIC_SUMMARY_MAX} characters" in system
    assert f"cut to its first {TOPIC_SUMMARY_SHOWN} characters" in system
    assert '{"op": "decision.new", "statement": "", "replaces": "none",' in system
    assert "leave out a field that has no value" in system
    # Version 3's single shape listed every field for every operation.
    assert '"held_by": [], "about": "", "member_id": ""' not in system
    assert "decision.supersede" not in system
