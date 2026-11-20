"""``message_ids``: what a resend keeps, what two messages never share."""

from __future__ import annotations

from typing import Any

from everos.memory.extract.ingest import message_ids

_TS = 1_700_000_000_000


def _m(text: str, ts: int = _TS, sender: str = "u1") -> dict[str, Any]:
    return {"sender_id": sender, "role": "user", "timestamp": ts, "content": text}


def test_a_message_keeps_its_id_wherever_it_sits_in_the_request() -> None:
    a, b = _m("a", _TS), _m("b", _TS + 1)
    assert message_ids("s", [a, b])[1] == message_ids("s", [b])[0]


def test_one_millisecond_resent_in_parts_keeps_its_ids() -> None:
    """[A, B] in one millisecond, then B alone: B is the same message."""
    a, b = _m("a"), _m("b")
    first = message_ids("s", [a, b])
    assert message_ids("s", [b]) == [first[1]]
    assert message_ids("s", [b, a]) == [first[1], first[0]]


def test_different_messages_in_one_millisecond_never_share_an_id() -> None:
    assert message_ids("s", [_m("a")]) != message_ids("s", [_m("b")])
    assert message_ids("s", [_m("a")]) != message_ids("s", [_m("a", sender="u2")])


def test_a_message_repeated_verbatim_in_one_request_is_kept_twice() -> None:
    ids = message_ids("s", [_m("ok"), _m("ok")])
    assert len(set(ids)) == 2
    assert ids[0] == message_ids("s", [_m("ok")])[0]


def test_the_label_is_not_part_of_the_identity() -> None:
    named = _m("a") | {"sender_name": "Alice"}
    assert message_ids("s", [named]) == message_ids("s", [_m("a")])
