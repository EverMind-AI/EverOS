"""Pure helpers over the session's canonical messages and the cells cut from them.

:func:`to_conversation_item` lives here rather than in ``memory.mappers``: it
reads the clock through ``core.utils.datetime``, which the mappers may
not import.
"""

from __future__ import annotations

__all__ = [
    "chat_rows",
    "close_complete_tail",
    "merge_dedupe_sort",
    "picture_names",
    "slice_tail",
    "split_message_ids",
    "to_conversation_item",
    "unique_senders",
]

from collections.abc import Sequence

from everalgo.boundary import DetectionResult
from everalgo.types import ChatMessage, ConversationItem, MemCell

from everos.core.utils.datetime import to_timestamp_ms
from everos.memory.models import CanonicalMessage


def chat_rows(messages: Sequence[CanonicalMessage]) -> list[CanonicalMessage]:
    """The plain user / assistant messages: a chat's memory reads these alone.

    A tool row and an assistant's tool call are an agent's trajectory, not chat;
    they are dropped before segmentation, and so never buffered or cut.
    """
    return [m for m in messages if m.role in ("user", "assistant") and not m.tool_calls]


def merge_dedupe_sort(
    buffered: Sequence[CanonicalMessage], new: Sequence[CanonicalMessage]
) -> list[CanonicalMessage]:
    """Union by ``message_id`` (a buffered row wins), sorted by ``(timestamp, id)``."""
    seen: dict[str, CanonicalMessage] = {m.message_id: m for m in buffered}
    for m in new:
        seen.setdefault(m.message_id, m)
    return sorted(seen.values(), key=lambda m: (m.timestamp, m.message_id))


def close_complete_tail(result: DetectionResult, *, is_final: bool) -> DetectionResult:
    """Close the uncut tail into a last cell when the model judged it complete.

    Only an explicit ``tail_complete is True`` closes it: ``None`` means no
    model judged the tail, and a final call has already closed it upstream.
    The cell takes the tail's last timestamp, as everalgo's own cells do, so
    it orders after its siblings.
    """
    if is_final or result.tail_complete is not True or not result.tail:
        return result
    tail_cell = MemCell(items=list(result.tail), timestamp=result.tail[-1].timestamp)
    return result._replace(cells=[*result.cells, tail_cell], tail=[])


def slice_tail(
    merged: Sequence[CanonicalMessage], tail: Sequence[ConversationItem]
) -> list[CanonicalMessage]:
    """The rows behind ``tail``, which the algorithm returns as a trailing slice."""
    return list(merged[len(merged) - len(tail) :]) if tail else []


def split_message_ids(
    merged: Sequence[CanonicalMessage], cells: Sequence[MemCell]
) -> list[list[str]]:
    """Each cell's message ids, walking ``merged`` in step with the cells' items.

    Rows and items correspond 1:1 and in order, so cell ``i`` holds the next
    ``len(cells[i].items)`` rows.
    """
    result: list[list[str]] = []
    start = 0
    for cell in cells:
        end = start + len(cell.items)
        result.append([m.message_id for m in merged[start:end]])
        start = end
    return result


def unique_senders(cell: MemCell) -> list[str]:
    """Distinct ``sender_id`` values in the cell, in first-seen order."""
    senders: list[str] = []
    for item in cell.items:
        sender = getattr(item, "sender_id", None)
        if sender and sender not in senders:
            senders.append(sender)
    return senders


def picture_names(messages: Sequence[CanonicalMessage]) -> dict[str, str]:
    """``image_id -> client file name`` for the pictures among ``messages``.

    The rendered conversation text keeps only a picture's content hash, so the
    name the client sent would be lost by the time a cell is extracted. This
    reads it off the canonical content items instead, which is the only layer
    that still holds them, and is keyed by hash because that is what the
    extraction route finds in the cell's text.

    First name wins when the same picture is sent twice under two names in one
    cell: the memory is one record about one arrival, and picking the earliest
    spelling keeps it stable across a re-run of the same batch.
    """
    names: dict[str, str] = {}
    for message in messages:
        for item in message.content_items:
            if item.get("type") != "image" or item.get("status") != "stored":
                continue
            image_id = item.get("image_id")
            name = item.get("name")
            if image_id and name and image_id not in names:
                names[str(image_id)] = str(name)
    return names


def to_conversation_item(message: CanonicalMessage) -> ConversationItem:
    """Map one chat row (see :func:`chat_rows`) to one conversation item.

    The mapping is 1:1 so boundary cells can be traced back to message ids.

    Raises:
        ValueError: A row that is not a plain user / assistant message;
            mapping it to anything would corrupt the cell-to-message index map.
    """
    if (message.role == "user" or message.role == "assistant") and not (
        message.tool_calls
    ):
        # A group batch's own message id, when the client sent one, is the
        # id the group's records cite (source quotes, message-level forget);
        # the generated id stays the buffer's and the memcell row's key.
        return ChatMessage(
            id=message.client_message_id or message.message_id,
            role=message.role,
            sender_id=message.sender_id,
            sender_name=message.sender_name,
            content=message.text,
            timestamp=to_timestamp_ms(message.timestamp),
        )
    raise ValueError(
        f"cannot map canonical row to a conversation item: role={message.role!r} "
        f"message_id={message.message_id!r} "
        f"has_tool_calls={bool(message.tool_calls)}"
    )
