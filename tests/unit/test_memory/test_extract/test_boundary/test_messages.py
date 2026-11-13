"""The pure buffer / cell helpers of boundary detection (no storage, no model)."""

from __future__ import annotations

import datetime as _dt

import pytest
from everalgo.boundary import Cuts
from everalgo.types import (
    ChatMessage,
    MemCell,
    ToolCallRequest,
    ToolCallResult,
)

from everos.memory import CanonicalMessage, ToolCall
from everos.memory.extract.boundary.messages import (
    apply_cuts,
    filter_for_mode,
    merge_dedupe_sort,
    to_conversation_item,
    unique_senders,
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


# ── filter_for_mode ──────────────────────────────────────────────────────


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
    out = filter_for_mode(msgs, "chat")
    ids = [m.message_id for m in out]
    assert ids == ["m1", "m2"]


def test_filter_agent_keeps_everything() -> None:
    msgs = [
        _msg("m1", "user"),
        _msg("m2", "tool", tool_call_id="tc1"),
    ]
    out = filter_for_mode(msgs, "agent")
    assert [m.message_id for m in out] == ["m1", "m2"]


# ── merge_dedupe_sort ────────────────────────────────────────────────────


def test_merge_dedupe_sort_dedupes_by_message_id() -> None:
    a = _msg("m1", "user", ts_seconds=100)
    b = _msg("m2", "user", ts_seconds=200)
    a_dup = _msg("m1", "user", ts_seconds=999, text="overwritten?")  # same id
    merged = merge_dedupe_sort([a, b], [a_dup])
    assert len(merged) == 2
    # The dup is dropped — buffered version wins (setdefault).
    m1 = next(m for m in merged if m.message_id == "m1")
    assert m1.text == "hi"  # original, not "overwritten?"


def test_merge_sorts_by_timestamp_then_id() -> None:
    a = _msg("m_b", "user", ts_seconds=100)
    b = _msg("m_a", "user", ts_seconds=100)
    c = _msg("m_c", "user", ts_seconds=50)
    merged = merge_dedupe_sort([], [a, b, c])
    assert [m.message_id for m in merged] == ["m_c", "m_a", "m_b"]


# ── apply_cuts ───────────────────────────────────────────────────────────


def _rows(n: int) -> list[CanonicalMessage]:
    return [_msg(f"m{i}", "user", ts_seconds=100 + i) for i in range(n)]


def _ids(rows: list[CanonicalMessage]) -> list[str]:
    return [m.message_id for m in rows]


def test_apply_cuts_slices_cells_ids_and_tail_by_position() -> None:
    rows = _rows(5)
    items = [to_conversation_item(m) for m in rows]

    out = apply_cuts(rows, items, Cuts(ends=[2, 4], tail_start=4))

    assert out.message_ids == [["m0", "m1"], ["m2", "m3"]]
    assert [cell.items for cell in out.cells] == [items[0:2], items[2:4]]
    assert [cell.timestamp for cell in out.cells] == [101_000, 103_000]
    assert _ids(out.tail) == ["m4"]


def test_apply_cuts_maps_tool_rows_by_position_too() -> None:
    """A tool call request carries no id; its position still names its row."""
    call = ToolCall(id="tc1", function={"name": "f", "arguments": "{}"})
    rows = [
        _msg("u1", "user", ts_seconds=100),
        _msg("call", "assistant", ts_seconds=101, tool_calls=[call]),
        _msg("result", "tool", ts_seconds=102, tool_call_id="tc1"),
        _msg("a1", "assistant", ts_seconds=103),
        _msg("u2", "user", ts_seconds=104),
    ]
    items = [to_conversation_item(m) for m in rows]

    out = apply_cuts(rows, items, Cuts(ends=[4], tail_start=4))

    assert out.message_ids == [["u1", "call", "result", "a1"]]
    assert _ids(out.tail) == ["u2"]


def test_apply_cuts_keeps_the_rows_the_last_cell_and_the_tail_share() -> None:
    rows = _rows(4)
    items = [to_conversation_item(m) for m in rows]

    out = apply_cuts(rows, items, Cuts(ends=[3], tail_start=2))

    assert out.message_ids == [["m0", "m1", "m2"]]
    assert _ids(out.tail) == ["m2", "m3"]


def test_apply_cuts_without_a_cut_keeps_every_row() -> None:
    rows = _rows(3)
    items = [to_conversation_item(m) for m in rows]

    out = apply_cuts(rows, items, Cuts(ends=[], tail_start=0))

    assert (out.cells, out.message_ids, out.tail) == ([], [], rows)


def test_apply_cuts_refuses_cuts_returned_for_another_input() -> None:
    rows = _rows(2)
    items = [to_conversation_item(m) for m in rows]
    with pytest.raises(ValueError, match=r"cuts\.ends"):
        apply_cuts(rows, items, Cuts(ends=[3], tail_start=3))


# ── unique_senders ──────────────────────────────────────────────────


def test_unique_senders_preserves_first_occurrence_order() -> None:

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
    assert unique_senders(cell) == ["alice", "bob"]


def test_unique_senders_handles_tool_result_without_sender() -> None:

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
    assert unique_senders(cell) == ["alice"]


# ── to_conversation_item ──────────────────────────────────────────────────


def test_to_item_tool_role_becomes_tool_call_result() -> None:
    m = _msg("m_tool", "tool", text="tool output", tool_call_id="tc1")
    item = to_conversation_item(m)
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
    item = to_conversation_item(m)
    assert isinstance(item, ToolCallRequest)
    assert len(item.tool_calls) == 1
    assert item.tool_calls[0].id == "tc1"
    assert item.tool_calls[0].function.name == "search"
    assert item.content == "picking a tool"


def test_to_item_user_becomes_chat_message() -> None:
    m = _msg("m_u", "user", text="hello")
    item = to_conversation_item(m)
    assert isinstance(item, ChatMessage)
    assert item.role == "user"
    assert item.content == "hello"


def test_to_item_assistant_text_only_becomes_chat_message() -> None:
    m = _msg("m_a", "assistant", text="hi back")
    item = to_conversation_item(m)
    assert isinstance(item, ChatMessage)
    assert item.role == "assistant"


def test_to_item_tool_without_id_falls_through_and_raises() -> None:
    # role=tool but no tool_call_id → not a ToolCallResult; falls to the
    # final raise (assistant/user branches don't match either).
    m = _msg("m_orphan", "tool", tool_call_id=None)
    with pytest.raises(ValueError, match="cannot map"):
        to_conversation_item(m)


def test_to_item_carries_the_epoch_millisecond_timestamp() -> None:
    item = to_conversation_item(_msg("m_u", "user", ts_seconds=1_700_000_000))
    assert item.timestamp == 1_700_000_000_000
