"""The pure buffer / cell helpers of boundary detection (no storage, no model)."""

from __future__ import annotations

import datetime as _dt

import pytest
from everalgo.boundary import DetectionResult
from everalgo.types import (
    ChatMessage,
    ConversationItem,
    MemCell,
    ToolCallRequest,
    ToolCallResult,
)

from everos.memory import CanonicalMessage, ToolCall
from everos.memory.extract.boundary.messages import (
    close_complete_tail,
    filter_for_mode,
    merge_dedupe_sort,
    slice_tail,
    split_message_ids,
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


# ── slice_tail ───────────────────────────────────────────────────────────


def test_slice_tail_zero_returns_empty() -> None:
    merged = [_msg(f"m{i}", "user") for i in range(3)]
    assert slice_tail(merged, []) == []


def test_slice_tail_returns_trailing_n() -> None:
    merged = [_msg(f"m{i}", "user", ts_seconds=100 + i) for i in range(5)]
    tail = [object()] * 2  # only length matters
    out = slice_tail(merged, tail)  # type: ignore[arg-type]
    assert [m.message_id for m in out] == ["m3", "m4"]


# ── close_complete_tail ──────────────────────────────────────────────


def _items(*ts_seconds: int) -> list[ConversationItem]:
    return [
        to_conversation_item(_msg(f"m{t}", "user", ts_seconds=t)) for t in ts_seconds
    ]


def test_close_complete_tail_true_appends_the_tail_as_last_cell() -> None:
    first = MemCell(items=_items(100), timestamp=100_000)
    tail = _items(200, 300)
    result = DetectionResult(cells=[first], tail=tail, tail_complete=True)

    closed = close_complete_tail(result, is_final=False)

    assert closed.tail == []
    assert len(closed.cells) == 2
    assert closed.cells[0] is first
    assert closed.cells[1].items == tail
    assert closed.cells[1].timestamp == 300_000  # the last item's, not the first's


@pytest.mark.parametrize("verdict", [False, None])
def test_close_complete_tail_keeps_an_unjudged_or_open_tail(
    verdict: bool | None,
) -> None:
    result = DetectionResult(cells=[], tail=_items(100, 200), tail_complete=verdict)

    assert close_complete_tail(result, is_final=False) is result


def test_close_complete_tail_leaves_a_final_call_alone() -> None:
    cell = MemCell(items=_items(100, 200), timestamp=200_000)
    result = DetectionResult(cells=[cell], tail=[], tail_complete=True)

    assert close_complete_tail(result, is_final=True) is result


# ── split_message_ids ──────────────────────────────────────────────


def test_split_message_ids_consumes_left_to_right() -> None:
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
    result = split_message_ids(merged, cells)
    assert result == [["m0", "m1"], ["m2", "m3", "m4"]]


# ── unique_senders ──────────────────────────────────────────────────


def test_unique_senders_preserves_first_occurrence_order() -> None:
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
    assert unique_senders(cell) == ["alice", "bob"]


def test_unique_senders_handles_tool_result_without_sender() -> None:
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
