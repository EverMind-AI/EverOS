"""The group-profile verifier: what it drops, what it builds (section 7.5).

docs/group-memory.md, section 7.5 and task 11. Crafted replies, no model.
What these pin:

- an unknown segment id, a paraphrased quote, a speaker the evidence does not
  back, and a decision resting on the assistant alone are each dropped, never
  repaired, and counted by reason -- in the result and in the metric;
- speakers, times, the span and the origin come from the cited messages;
- an id an operation updates or supersedes must be a rendered current item;
- topic.update keeps the topic id and names the version it replaces;
- a topic.new's topic_id is a label the reply's other operations name the new
  topic by, before or after it; code mints the id;
- a link that names nothing (a made-up id, the label of a dropped topic, a
  label two new topics share) is dropped and counted, the item kept; targets
  and the semantic checks stay strict, and a dropped operation counts no link.
"""

from __future__ import annotations

from typing import Any

from everalgo.types import ChatMessage, MemCell

from everos.core.observability.metrics import get_metrics_registry
from everos.memory.extract.group_profile import (
    GroupProfileView,
    Operation,
    Segment,
    drop_summary,
    verify,
)
from everos.memory.models import (
    GroupProfileItem,
    ProfileSource,
    Roster,
    RosterEvent,
    fold_roster,
)

_ROSTER = Roster(
    intervals=fold_roster(
        [
            RosterEvent(op="join", member_id="u_chen", at=0, display_name="Chen"),
            RosterEvent(op="join", member_id="u_betty", at=0, display_name="Betty"),
            RosterEvent(
                op="join",
                member_id="bot_ava",
                at=0,
                display_name="Ava",
                kind="assistant",
            ),
        ]
    )
)


def _message(mid: str, sender: str, text: str, at: int) -> ChatMessage:
    role = "assistant" if sender.startswith("bot") else "user"
    return ChatMessage(
        id=mid,
        role=role,
        sender_id=sender,
        sender_name=sender,
        content=text,
        timestamp=at,
    )


_SEGMENT = Segment(
    memcell_id="mc_1",
    cell=MemCell(
        items=[
            _message(
                "m-102",
                "u_chen",
                "May 1 flights are sold out. Let's leave on April 30 instead.",
                1_000,
            ),
            _message(
                "m-103",
                "u_betty",
                "Fine by me.  I'll book the hotel from April 30.",
                2_000,
            ),
            _message(
                "m-104",
                "bot_ava",
                "There are two April 30 flights: 08:10 and 13:40.",
                3_000,
            ),
        ],
        timestamp=3_000,
    ),
)


def _op(op: str, *evidence: tuple[str, str, str], **fields: Any) -> Operation:
    blank: dict[str, Any] = {
        "target": "",
        "topic_id": "",
        "name": "",
        "summary": "",
        "status": "",
        "statement": "",
        "decided_by": [],
        "agreed_by": [],
        "assignees": [],
        "applies_to": [],
        "held_by": [],
        "about": "",
        "member_id": "",
        "role": "",
        "confidence": "",
    }
    return Operation.model_validate(
        {
            **blank,
            **fields,
            "op": op,
            "evidence": [
                {"segment": s, "message": m, "quote": q} for s, m, q in evidence
            ],
        }
    )


def _check(*operations: Operation, profile: GroupProfileView | None = None) -> Any:
    return verify(
        operations,
        group_id="trip",
        segments=[_SEGMENT],
        roster=_ROSTER,
        profile=profile or GroupProfileView(),
        observed_at=3_000,
    )


def _dropped(reason: str) -> float:
    value = get_metrics_registry().get_sample_value(
        "everos_group_profile_ops_dropped_total", {"reason": reason}
    )
    return value or 0.0


def _links_dropped(reason: str) -> float:
    value = get_metrics_registry().get_sample_value(
        "everos_group_profile_links_dropped_total", {"reason": reason}
    )
    return value or 0.0


_DEPART = ("mc_1", "m-102", "Let's leave on April 30 instead.")
_AGREE = ("mc_1", "m-103", "Fine by me. I'll book the hotel")


def test_a_grounded_decision_becomes_an_item_with_the_messages_speakers() -> None:
    result = _check(
        _op(
            "decision.new",
            _DEPART,
            _AGREE,
            statement="Depart on April 30 instead of May 1.",
            decided_by=["u_chen"],
            agreed_by=["u_betty"],
        )
    )

    assert not result.dropped
    (item,) = result.items
    assert item.kind == "decision" and item.owner_id == "trip"
    assert [(s.speaker_id, s.at) for s in item.sources] == [
        ("u_chen", 1_000),
        ("u_betty", 2_000),
    ]
    # The double space in the message is normalised away, never a mismatch.
    assert item.sources[1].quote == "Fine by me. I'll book the hotel"
    assert (item.source_start, item.source_end, item.observed_at) == (
        1_000,
        2_000,
        3_000,
    )
    assert item.sender_ids == ["u_chen", "u_betty"] and item.origin == "member"
    assert item.subject_ids == ["u_chen", "u_betty"]


def test_each_bad_operation_is_dropped_and_counted_by_reason() -> None:
    reasons = (
        "unknown_segment",
        "quote_not_verbatim",
        "ungrounded_member",
        "assistant_only_decision",
    )
    before = {r: _dropped(r) for r in reasons}
    decision = {"statement": "Depart on April 30.", "decided_by": ["u_chen"]}

    result = _check(
        # An unknown segment id.
        _op("decision.new", ("mc_9", "m-102", "Let's leave"), **decision),
        # A paraphrased quote.
        _op("decision.new", ("mc_1", "m-102", "Let us leave on April 30"), **decision),
        # A wrong speaker: Betty decided, says the model; the message is Chen's.
        _op(
            "decision.new",
            _DEPART,
            statement="Depart on April 30.",
            decided_by=["u_betty"],
        ),
        # The assistant's words alone.
        _op(
            "decision.new",
            ("mc_1", "m-104", "two April 30 flights"),
            statement="Take a flight on April 30.",
        ),
    )

    assert result.items == []
    assert dict(result.dropped) == dict.fromkeys(reasons, 1)
    assert {r: _dropped(r) - before[r] for r in reasons} == dict.fromkeys(reasons, 1.0)


def test_the_rest_of_the_checks() -> None:
    result = _check(
        _op("decision.new", statement="No evidence at all."),
        _op("decision.new", ("mc_1", "m-999", "Fine"), statement="x"),
        _op("decision.new", _DEPART, statement="x", assignees=["u_zoe"]),
        _op("decision.supersede", _DEPART, statement="x", target="gp_unknown"),
        _op("topic.new", _DEPART, name="Travel", status="ongoing"),
        _op("role.add", _DEPART, member_id="u_chen", role="boss", confidence="strong"),
    )

    assert result.items == []
    assert dict(result.dropped) == {
        "no_evidence": 1,
        "unknown_message": 1,
        "unknown_member": 1,
        "unknown_target": 1,
        "invalid_field": 2,
    }


def _stored(item: GroupProfileItem, item_id: str) -> GroupProfileItem:
    return item.read_back(item_id=item_id, mark=None)


_SOURCE = ProfileSource(
    memcell_id="mc_0", message_id="m-1", speaker_id="u_chen", at=500, quote="q"
)
_TOPIC = _stored(
    GroupProfileItem(
        owner_id="trip",
        kind="topic",
        topic_id="tp_trip",
        name="Travel",
        summary="How we get there",
        topic_status="exploring",
        sources=[_SOURCE],
        source_start=500,
        source_end=500,
        observed_at=500,
    ),
    "gp_1",
)
_DECISION = _stored(
    GroupProfileItem(
        owner_id="trip",
        kind="decision",
        statement="Fly out on May 1.",
        topic_id="tp_trip",
        decided_by=["u_chen"],
        sources=[_SOURCE],
        source_start=500,
        source_end=500,
        observed_at=500,
    ),
    "gp_2",
)


def test_updates_name_what_they_replace_and_keep_the_topic() -> None:
    profile = GroupProfileView(topics=[_TOPIC], decisions=[_DECISION])

    result = _check(
        _op("topic.update", _DEPART, target="tp_trip", status="consensus"),
        _op(
            "decision.supersede",
            _DEPART,
            _AGREE,
            target="gp_2",
            statement="Depart on April 30 instead of May 1.",
            decided_by=["u_chen"],
            agreed_by=["u_betty"],
        ),
        # A second operation on the same decision in one reply.
        _op("decision.supersede", _DEPART, target="gp_2", statement="Again."),
        profile=profile,
    )

    topic, decision = result.items
    assert (topic.topic_id, topic.supersedes, topic.topic_status) == (
        "tp_trip",
        "gp_1",
        "consensus",
    )
    assert (topic.name, topic.summary) == ("Travel", "How we get there")
    assert (decision.supersedes, decision.topic_id) == ("gp_2", "tp_trip")
    assert dict(result.dropped) == {"duplicate": 1}


def test_a_dissent_is_resolved_by_a_new_version() -> None:
    dissent = _stored(
        GroupProfileItem(
            owner_id="trip",
            kind="dissent",
            statement="Betty thinks April 30 is too early.",
            held_by=["u_betty"],
            about="gp_2",
            sources=[_SOURCE],
            source_start=500,
            source_end=500,
            observed_at=500,
        ),
        "gp_3",
    )
    result = _check(
        _op("dissent.resolve", _AGREE, target="gp_3", statement="Betty agreed."),
        profile=GroupProfileView(decisions=[_DECISION], dissent=[dissent]),
    )

    (resolved,) = result.items
    assert resolved.resolved and resolved.supersedes == "gp_3"
    assert resolved.held_by == ["u_betty"] and resolved.resolved_by == ["u_betty"]
    assert resolved.own_status() == "resolved" and dissent.own_status() == "open"


def test_curly_and_straight_quotation_marks_match_either_way() -> None:
    segment = Segment(
        memcell_id="mc_q",
        cell=MemCell(
            items=[
                _message(
                    "m-1", "u_chen", "The “delivery timeline” was vague; we’ll fix.", 1
                ),
                _message("m-2", "u_betty", 'He said "maybe" and it\'s fine.', 2),
            ],
            timestamp=2,
        ),
    )
    topic = {"name": "Delivery", "status": "exploring"}

    result = verify(
        [
            _op(
                "topic.new",
                ("mc_q", "m-1", 'The "delivery timeline" was vague; we\'ll fix'),
                **topic,
            ),
            _op(
                "topic.new", ("mc_q", "m-2", "He said “maybe” and it’s fine."), **topic
            ),
            # A changed word is still not verbatim, whatever the marks.
            _op("topic.new", ("mc_q", "m-1", "The “delivery date” was vague"), **topic),
        ],
        group_id="trip",
        segments=[segment],
        roster=_ROSTER,
        profile=GroupProfileView(),
        observed_at=2,
    )

    first, second = result.items
    # The quote kept is the message's own words, with its own marks.
    assert first.sources[0].quote == "The “delivery timeline” was vague; we’ll fix"
    assert second.sources[0].quote == 'He said "maybe" and it\'s fine.'
    assert result.dropped == {"quote_not_verbatim": 1}


# ── topics created in the same reply, and links ───────────────────────────

_TRAVEL = {"name": "Travel", "summary": "Dates and flights", "status": "consensus"}
_HOTEL = {"name": "Hotel", "summary": "Where we stay", "status": "exploring"}
_DECIDED = {"statement": "Depart on April 30.", "decided_by": ["u_chen"]}
_OBJECTS = {"statement": "Betty would rather wait.", "held_by": ["u_betty"]}
_ASSISTANT = ("mc_1", "m-104", "two April 30 flights")


def test_a_decision_names_a_topic_its_reply_creates_after_it() -> None:
    result = _check(
        _op("decision.new", _DEPART, topic_id="new_1", **_DECIDED),
        _op("topic.new", _DEPART, topic_id="new_1", **_TRAVEL),
    )

    decision, topic = result.items  # the reply's order
    assert (decision.kind, topic.kind) == ("decision", "topic")
    assert topic.topic_id and topic.topic_id.startswith("tp_")
    assert decision.topic_id == topic.topic_id
    assert not result.dropped and not result.links_dropped


def test_operations_name_topics_their_reply_created_before_them() -> None:
    profile = GroupProfileView(topics=[_TOPIC], decisions=[_DECISION])

    result = _check(
        _op("topic.new", _DEPART, topic_id="new_1", **_TRAVEL),
        _op("topic.new", _AGREE, topic_id="new_2", **_HOTEL),
        _op("decision.new", _DEPART, topic_id="new_1", **_DECIDED),
        _op("dissent.open", _AGREE, about="new_2", **_OBJECTS),
        # A superseding decision moves to a new topic by its label.
        _op("decision.supersede", _DEPART, target="gp_2", topic_id="new_2", **_DECIDED),
        profile=profile,
    )

    travel, hotel, decision, dissent, moved = result.items
    assert len({travel.topic_id, hotel.topic_id, "tp_trip"}) == 3
    assert decision.topic_id == travel.topic_id
    assert dissent.about == hotel.topic_id
    assert (moved.supersedes, moved.topic_id) == ("gp_2", hotel.topic_id)
    assert not result.dropped and not result.links_dropped


def test_a_link_that_names_nothing_is_dropped_and_the_item_kept() -> None:
    before = _links_dropped("unknown_reference")

    result = _check(
        # The 11-11 smoke: the topic.new gives no label, and the decision and
        # the dissent name a topic id the model made up.
        _op("topic.new", _DEPART, **_TRAVEL),
        _op("decision.new", _DEPART, topic_id="tp_spring_offsite", **_DECIDED),
        _op("dissent.open", _AGREE, about="tp_spring_offsite", **_OBJECTS),
        # A decision id that was never rendered.
        _op("dissent.open", _AGREE, about="gp_x", **_OBJECTS),
    )

    topic, decision, dissent, other = result.items
    assert topic.kind == "topic" and decision.statement == "Depart on April 30."
    assert decision.topic_id is None
    assert dissent.about is None and other.about is None
    assert not result.dropped
    assert drop_summary(result.links_dropped) == {"unknown_reference": 3}
    assert _links_dropped("unknown_reference") - before == 3.0


def test_the_label_of_a_dropped_or_doubled_topic_names_nothing() -> None:
    result = _check(
        # Dropped (a paraphrased quote): its label names nothing.
        _op(
            "topic.new", ("mc_1", "m-102", "Let us leave"), topic_id="new_1", **_TRAVEL
        ),
        _op("decision.new", _DEPART, topic_id="new_1", **_DECIDED),
        # Two new topics under one label: it names neither.
        _op("topic.new", _DEPART, topic_id="new_2", **_TRAVEL),
        _op("topic.new", _AGREE, topic_id="new_2", **_HOTEL),
        _op("dissent.open", _AGREE, about="new_2", **_OBJECTS),
    )

    decision, travel, hotel, dissent = result.items
    assert decision.topic_id is None and dissent.about is None
    assert travel.topic_id != hotel.topic_id
    assert dict(result.dropped) == {"quote_not_verbatim": 1}
    assert drop_summary(result.links_dropped) == {
        "unknown_reference": 1,
        "ambiguous_reference": 1,
    }


def test_a_listed_id_wins_over_a_label_that_repeats_it() -> None:
    profile = GroupProfileView(topics=[_TOPIC], decisions=[_DECISION])

    result = _check(
        _op("topic.new", _DEPART, topic_id="tp_trip", **_HOTEL),
        _op("decision.new", _DEPART, topic_id="tp_trip", **_DECIDED),
        profile=profile,
    )

    topic, decision = result.items
    assert topic.topic_id != "tp_trip" and decision.topic_id == "tp_trip"


def test_a_supersede_whose_link_names_nothing_keeps_its_topic() -> None:
    result = _check(
        _op("decision.supersede", _DEPART, target="gp_2", topic_id="tp_x", **_DECIDED),
        profile=GroupProfileView(topics=[_TOPIC], decisions=[_DECISION]),
    )

    (decision,) = result.items
    assert (decision.supersedes, decision.topic_id) == ("gp_2", "tp_trip")
    assert dict(result.links_dropped) == {"unknown_reference": 1}


def test_targets_and_the_semantic_checks_stay_strict() -> None:
    profile = GroupProfileView(topics=[_TOPIC], decisions=[_DECISION])

    result = _check(
        _op("topic.new", _DEPART, topic_id="new_1", **_TRAVEL),
        # A topic the reply creates is written as it stands, never updated in
        # the same reply; a label is no target, nor is a made-up id.
        _op("topic.update", _AGREE, target="new_1", **_HOTEL),
        _op("decision.supersede", _DEPART, target="new_1", **_DECIDED),
        _op("decision.supersede", _DEPART, target="gp_9", **_DECIDED),
        _op("dissent.resolve", _AGREE, target="gp_9", statement="Settled."),
        _op("role.retire", _DEPART, target="gp_9"),
        # A link never saves an operation that fails a check, and a dropped
        # operation counts no link: the assistant's words alone, a speaker
        # the evidence does not back, a dissent given to the assistant.
        _op("decision.new", _ASSISTANT, topic_id="new_1", statement="Fly 08:10."),
        _op(
            "decision.new",
            _DEPART,
            topic_id="tp_x",
            statement="x",
            decided_by=["u_betty"],
        ),
        _op(
            "dissent.open", _ASSISTANT, about="tp_x", statement="x", held_by=["bot_ava"]
        ),
        profile=profile,
    )

    (topic,) = result.items
    assert topic.kind == "topic"
    assert dict(result.dropped) == {
        "unknown_target": 5,
        "assistant_only_decision": 1,
        "ungrounded_member": 1,
        "unknown_member": 1,
    }
    assert not result.links_dropped
