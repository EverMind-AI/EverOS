"""Unit tests for the pure helpers in :mod:`everos.service._boundary`.

Covers the mapping + filter functions that don't touch sqlite — fast,
deterministic, no fixture overhead. The full ``prepare_cells`` flow is
exercised by integration tests under tests/integration/.
"""

from __future__ import annotations

import datetime as _dt

import pytest
from everalgo.types import ChatMessage, ToolCallRequest, ToolCallResult

from everos.memory import CanonicalMessage, ToolCall
from everos.service._boundary import (
    _filter_for_mode,
    _merge_dedupe_sort,
    _slice_tail,
    _split_messages_per_cell,
    _to_conversation_item,
    _unique_all_senders,
)


def _msg(
    mid: str,
    role: str,
    *,
    ts_seconds: int = 1_700_000_000,
    text: str = "hi",
    sender: str = "u1",
    tool_calls: list[ToolCall] | None = None,
    tool_call_id: str | None = None,
) -> CanonicalMessage:
    return CanonicalMessage(
        message_id=mid,
        session_id="s1",
        sender_id=sender,
        sender_name=sender,
        role=role,  # type: ignore[arg-type]
        timestamp=_dt.datetime.fromtimestamp(ts_seconds, tz=_dt.UTC),
        text=text,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
    )


# ── _filter_for_mode ──────────────────────────────────────────────────────


def test_filter_chat_drops_tool_rows() -> None:
    msgs = [
        _msg("m1", "user"),
        _msg("m2", "assistant"),
        _msg("m3", "tool", tool_call_id="tc1"),
        _msg(
            "m4",
            "assistant",
            tool_calls=[ToolCall(id="tc1", function={"name": "f", "arguments": "{}"})],
        ),
    ]
    out = _filter_for_mode(msgs, "chat")
    ids = [m.message_id for m in out]
    assert ids == ["m1", "m2"]


def test_filter_agent_keeps_everything() -> None:
    msgs = [
        _msg("m1", "user"),
        _msg("m2", "tool", tool_call_id="tc1"),
    ]
    out = _filter_for_mode(msgs, "agent")
    assert [m.message_id for m in out] == ["m1", "m2"]


# ── _to_conversation_item dispatch ────────────────────────────────────────


def test_to_item_tool_role_becomes_tool_call_result() -> None:
    m = _msg("m_tool", "tool", text="tool output", tool_call_id="tc1")
    item = _to_conversation_item(m)
    assert isinstance(item, ToolCallResult)
    assert item.tool_call_id == "tc1"
    assert item.content == "tool output"


def test_to_item_assistant_with_tool_calls_becomes_request() -> None:
    m = _msg(
        "m_ac",
        "assistant",
        text="picking a tool",
        tool_calls=[
            ToolCall(id="tc1", function={"name": "search", "arguments": '{"q":"x"}'})
        ],
    )
    item = _to_conversation_item(m)
    assert isinstance(item, ToolCallRequest)
    assert len(item.tool_calls) == 1
    assert item.tool_calls[0].id == "tc1"
    assert item.tool_calls[0].function.name == "search"
    assert item.content == "picking a tool"


def test_to_item_user_becomes_chat_message() -> None:
    m = _msg("m_u", "user", text="hello")
    item = _to_conversation_item(m)
    assert isinstance(item, ChatMessage)
    assert item.role == "user"
    assert item.content == "hello"


def test_to_item_assistant_text_only_becomes_chat_message() -> None:
    m = _msg("m_a", "assistant", text="hi back")
    item = _to_conversation_item(m)
    assert isinstance(item, ChatMessage)
    assert item.role == "assistant"


def test_to_item_tool_without_id_falls_through_and_raises() -> None:
    # role=tool but no tool_call_id → not a ToolCallResult; falls to the
    # final raise (assistant/user branches don't match either).
    m = _msg("m_orphan", "tool", tool_call_id=None)
    with pytest.raises(ValueError, match="cannot map"):
        _to_conversation_item(m)


# ── _merge_dedupe_sort ────────────────────────────────────────────────────


def test_merge_dedupe_sort_dedupes_by_message_id() -> None:
    a = _msg("m1", "user", ts_seconds=100)
    b = _msg("m2", "user", ts_seconds=200)
    a_dup = _msg("m1", "user", ts_seconds=999, text="overwritten?")  # same id
    merged = _merge_dedupe_sort([a, b], [a_dup])
    assert len(merged) == 2
    # The dup is dropped — buffered version wins (setdefault).
    m1 = next(m for m in merged if m.message_id == "m1")
    assert m1.text == "hi"  # original, not "overwritten?"


def test_merge_sorts_by_timestamp_then_id() -> None:
    a = _msg("m_b", "user", ts_seconds=100)
    b = _msg("m_a", "user", ts_seconds=100)
    c = _msg("m_c", "user", ts_seconds=50)
    merged = _merge_dedupe_sort([], [a, b, c])
    assert [m.message_id for m in merged] == ["m_c", "m_a", "m_b"]


# ── _slice_tail ───────────────────────────────────────────────────────────


def test_slice_tail_zero_returns_empty() -> None:
    merged = [_msg(f"m{i}", "user") for i in range(3)]
    assert _slice_tail(merged, []) == []


def test_slice_tail_returns_trailing_n() -> None:
    merged = [_msg(f"m{i}", "user", ts_seconds=100 + i) for i in range(5)]
    tail = [object()] * 2  # only length matters
    out = _slice_tail(merged, tail)  # type: ignore[arg-type]
    assert [m.message_id for m in out] == ["m3", "m4"]


# ── _split_messages_per_cell ──────────────────────────────────────────────


def test_split_messages_per_cell_consumes_left_to_right() -> None:
    from everalgo.types import MemCell

    merged = [_msg(f"m{i}", "user", ts_seconds=100 + i) for i in range(5)]
    cells = [
        MemCell(
            items=[
                ChatMessage(
                    id="x",
                    role="user",
                    sender_id="u",
                    sender_name="u",
                    content="",
                    timestamp=0,
                )
                for _ in range(2)
            ],
            timestamp=0,
        ),
        MemCell(
            items=[
                ChatMessage(
                    id="x",
                    role="user",
                    sender_id="u",
                    sender_name="u",
                    content="",
                    timestamp=0,
                )
                for _ in range(3)
            ],
            timestamp=0,
        ),
    ]
    result = _split_messages_per_cell(merged, cells)
    assert result == [["m0", "m1"], ["m2", "m3", "m4"]]


# ── _unique_all_senders ──────────────────────────────────────────────────


def test_unique_all_senders_preserves_first_occurrence_order() -> None:
    from everalgo.types import MemCell

    cell = MemCell(
        items=[
            ChatMessage(
                id="x",
                role="user",
                sender_id="alice",
                sender_name="Alice",
                content="",
                timestamp=0,
            ),
            ChatMessage(
                id="y",
                role="assistant",
                sender_id="bob",
                sender_name="Bob",
                content="",
                timestamp=0,
            ),
            ChatMessage(
                id="z",
                role="user",
                sender_id="alice",
                sender_name="Alice",
                content="",
                timestamp=0,
            ),
        ],
        timestamp=0,
    )
    assert _unique_all_senders(cell) == ["alice", "bob"]


def test_unique_all_senders_handles_tool_result_without_sender() -> None:
    from everalgo.types import MemCell

    cell = MemCell(
        items=[
            ChatMessage(
                id="x",
                role="user",
                sender_id="alice",
                sender_name="Alice",
                content="",
                timestamp=0,
            ),
            ToolCallResult(tool_call_id="tc1", content="output", timestamp=0),
        ],
        timestamp=0,
    )
    assert _unique_all_senders(cell) == ["alice"]
