"""Pure helpers over the session's canonical messages and the cells cut from them.

:func:`to_conversation_item` lives here rather than in ``memory.mappers``: it
reads the clock through ``core.utils.datetime``, which the mappers may
not import.
"""

from __future__ import annotations

__all__ = [
    "Mode",
    "Segmentation",
    "apply_cuts",
    "filter_for_mode",
    "merge_dedupe_sort",
    "to_conversation_item",
    "unique_senders",
]

from collections.abc import Sequence
from itertools import pairwise
from typing import Literal, NamedTuple

from everalgo.boundary import Cuts, to_memcells
from everalgo.types import (
    ChatMessage,
    ConversationItem,
    MemCell,
    ToolCall,
    ToolCallFunction,
    ToolCallRequest,
    ToolCallResult,
)

from everos.core.utils.datetime import to_timestamp_ms
from everos.memory.models import CanonicalMessage

type Mode = Literal["chat", "agent"]


def filter_for_mode(
    messages: Sequence[CanonicalMessage], mode: Mode
) -> list[CanonicalMessage]:
    """Chat mode keeps plain user / assistant text; agent mode keeps everything."""
    if mode == "chat":
        return [
            m for m in messages if m.role in ("user", "assistant") and not m.tool_calls
        ]
    return list(messages)


def merge_dedupe_sort(
    buffered: Sequence[CanonicalMessage], new: Sequence[CanonicalMessage]
) -> list[CanonicalMessage]:
    """Union by ``message_id`` (a buffered row wins), sorted by ``(timestamp, id)``."""
    seen: dict[str, CanonicalMessage] = {m.message_id: m for m in buffered}
    for m in new:
        seen.setdefault(m.message_id, m)
    return sorted(seen.values(), key=lambda m: (m.timestamp, m.message_id))


class Segmentation(NamedTuple):
    """Rows cut into memcells, the parallel lists indexed by cell.

    Attributes:
        cells: The closed memcells, in order.
        message_ids: The ids of the rows each cell holds.
        tail: The rows left for the next call. It may begin inside the last
            cell, and the rows they share are in both.
    """

    cells: list[MemCell]
    message_ids: list[list[str]]
    tail: list[CanonicalMessage]


def apply_cuts(
    rows: Sequence[CanonicalMessage],
    items: Sequence[ConversationItem],
    cuts: Cuts,
) -> Segmentation:
    """Slice ``rows`` by the ``cuts`` everalgo returned for ``items``.

    ``items`` is what the boundary operator was handed, ``items[i]`` being
    ``rows[i]`` mapped, so every cut is a position in both lists: a cell's
    message ids are the ids of the rows at its positions, tool rows included.

    Raises:
        ValueError: ``cuts`` does not fit ``items`` (everalgo's
            ``InvalidInputError``): it was returned for another input.
    """
    return Segmentation(
        cells=to_memcells(items, cuts),
        message_ids=[
            [row.message_id for row in rows[start:end]]
            for start, end in pairwise([0, *cuts.ends])
        ],
        tail=list(rows[cuts.tail_start :]),
    )


def unique_senders(cell: MemCell) -> list[str]:
    """Distinct ``sender_id`` values in the cell, in first-seen order.

    Tool results carry no sender (a tool runner is not a speaker).
    """
    senders: list[str] = []
    for item in cell.items:
        sender = getattr(item, "sender_id", None)
        if sender and sender not in senders:
            senders.append(sender)
    return senders


def to_conversation_item(message: CanonicalMessage) -> ConversationItem:
    """Map one canonical row to one conversation item, in this order of rules.

    1. ``role="tool"`` with a ``tool_call_id`` -> :class:`ToolCallResult`.
    2. ``role="assistant"`` with ``tool_calls`` -> :class:`ToolCallRequest`;
       any text rides along as its ``content``.
    3. ``role`` ``"user"`` or ``"assistant"`` -> :class:`ChatMessage`.

    The mapping is 1:1, so a cut everalgo places in the items is a position
    in the rows as well (:func:`apply_cuts`).

    Raises:
        ValueError: An orphan tool row or an unexpected role; mapping it to
            anything would corrupt the cell-to-message index map.
    """
    if message.role == "tool" and message.tool_call_id:
        return ToolCallResult(
            tool_call_id=message.tool_call_id,
            content=message.text,
            timestamp=to_timestamp_ms(message.timestamp),
        )
    if message.role == "assistant" and message.tool_calls:
        return ToolCallRequest(
            tool_calls=[
                ToolCall(
                    id=call.id,
                    function=ToolCallFunction(
                        name=call.function.get("name", ""),
                        arguments=call.function.get("arguments", ""),
                    ),
                )
                for call in message.tool_calls
            ],
            timestamp=to_timestamp_ms(message.timestamp),
            content=message.text or None,
            sender_id=message.sender_id,
            sender_name=message.sender_name,
        )
    if message.role == "user" or message.role == "assistant":
        return ChatMessage(
            id=message.message_id,
            role=message.role,
            sender_id=message.sender_id,
            sender_name=message.sender_name,
            content=message.text,
            timestamp=to_timestamp_ms(message.timestamp),
        )
    raise ValueError(
        f"cannot map canonical row to a conversation item: role={message.role!r} "
        f"message_id={message.message_id!r} "
        f"has_tool_call_id={message.tool_call_id is not None}"
    )
