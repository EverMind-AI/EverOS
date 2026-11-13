"""The dedupe key of a batch message (docs/group-memory.md, section 5.2).

What these pin:

- a message with a client ``message_id`` is keyed by it alone;
- one without is keyed by its sender, its time in epoch ms (seconds read as
  ingest reads them) and the SHA-256 of its content as sent -- the same
  content items with their unset fields dropped or not are one content;
- a row a chat memory never stores (a tool row, a tool call) has no key.
"""

from __future__ import annotations

import hashlib
import json

from everos.memory.groups import message_key

T0 = 1_772_323_200_000


def test_a_client_id_is_the_key() -> None:
    message = {"message_id": "m-1", "sender_id": "u", "role": "user", "timestamp": T0}
    assert message_key(message | {"content": "a"}) == "id:m-1"
    assert message_key(message | {"content": "b", "sender_id": "v"}) == "id:m-1"


def test_without_an_id_the_sender_time_and_content_are_the_key() -> None:
    message = {"sender_id": "u_chen", "role": "user", "timestamp": T0, "content": "hi"}
    digest = hashlib.sha256(json.dumps("hi").encode()).hexdigest()
    assert message_key(message) == f"fp:u_chen:{T0}:{digest}"
    # Seconds are the same instant as milliseconds, as ingest reads them.
    assert message_key(message | {"timestamp": T0 // 1000}) == message_key(message)
    assert message_key(message | {"content": "hi!"}) != message_key(message)
    assert message_key(message | {"sender_id": "u_dan"}) != message_key(message)
    assert message_key(message | {"timestamp": T0 + 1}) != message_key(message)


def test_content_items_compare_without_their_unset_fields() -> None:
    sent = [{"type": "text", "text": "hi", "uri": None, "base64": None}]
    plain = [{"type": "text", "text": "hi"}]
    message = {"sender_id": "u", "role": "user", "timestamp": T0}
    assert message_key(message | {"content": sent}) == message_key(
        message | {"content": plain}
    )
    assert message_key(message | {"content": plain}) != message_key(
        message | {"content": "hi"}
    )


def test_rows_a_chat_memory_never_stores_have_no_key() -> None:
    base = {"sender_id": "u", "timestamp": T0, "content": "x"}
    assert message_key(base | {"role": "tool", "tool_call_id": "c1"}) is None
    calls = [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}]
    assert message_key(base | {"role": "assistant", "tool_calls": calls}) is None
