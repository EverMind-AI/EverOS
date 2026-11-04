"""Where a conversation is cut, and the cells and tail those cuts produce.

The model only ever sees ``ChatMessage`` items, so every cut is first decided over the chat messages alone and
then carried over to the full item list: a cut before a chat message falls right before it, which keeps the tool
items between two chat messages with the earlier one; tool items ahead of the first chat message join the first
cell; a cut after the last chat message closes the stream and takes the trailing tool items with it.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

from everos.algo.boundary._render import _count_history_tokens
from everos.algo.types import ChatMessage, MemCell

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.algo.types import ConversationItem


# ---------------------------------------------------------------------------
# Batch path
# ---------------------------------------------------------------------------


def _chat_view(
    items: Sequence[ConversationItem],
) -> tuple[list[int], list[ChatMessage]]:
    """The ``ChatMessage`` items in order, and the position of each in ``items``."""
    chat = [
        (position, item)
        for position, item in enumerate(items)
        if isinstance(item, ChatMessage)
    ]
    return [position for position, _ in chat], [message for _, message in chat]


def _force_split_cuts(
    messages: Sequence[ChatMessage], hard_token_limit: int, hard_msg_limit: int
) -> list[int]:
    """Cuts that close leading cells until what is left fits under both hard limits, ascending.

    The last cut, if any, is where the window handed to the model starts. A window of one message is never cut
    further, however long it is.
    """
    cuts: list[int] = []
    start = 0
    while len(messages) - start > 1 and _exceeds_limits(
        messages[start:], hard_token_limit, hard_msg_limit
    ):
        start += _find_force_split_point(
            messages[start:], hard_token_limit, hard_msg_limit
        )
        cuts.append(start)
    return cuts


def _exceeds_limits(
    messages: Sequence[ChatMessage], hard_token_limit: int, hard_msg_limit: int
) -> bool:
    """Whether ``messages`` reach either hard limit."""
    return (
        _count_history_tokens(messages) >= hard_token_limit
        or len(messages) >= hard_msg_limit
    )


def _find_force_split_point(
    messages: Sequence[ChatMessage], hard_token_limit: int, hard_msg_limit: int
) -> int:
    """Binary-halving search for a force-split index; floor 1 so even a single very-long message is accepted."""
    if len(messages) <= 1:
        return len(messages)
    candidate = min(hard_msg_limit - 1, len(messages) - 1)
    while (
        candidate > 1
        and _count_history_tokens(messages[:candidate]) >= hard_token_limit
    ):
        candidate = max(1, candidate // 2)
    return candidate


def _cut_items(
    items: Sequence[ConversationItem],
    chat_positions: Sequence[int],
    chat_cuts: Sequence[int],
) -> tuple[list[MemCell], list[ConversationItem]]:
    """Cut ``items`` at ``chat_cuts`` into closed cells and the tail after the last cut.

    Args:
        items: The full item list.
        chat_positions: The positions ``_chat_view(items)`` returns.
        chat_cuts: Ascending cut points over the chat messages, each in ``1..len(chat_positions)``; a cut at ``k``
            falls between the ``k``-th chat message and the next.

    Returns:
        One cell per cut, stamped with the time of its last chat message, and the items after the last cut.
    """
    item_cuts = [
        0,
        *(_item_cut(chat_cut, chat_positions, len(items)) for chat_cut in chat_cuts),
    ]
    cells = [
        _make_cell(
            items[start:end], timestamp=items[chat_positions[chat_cut - 1]].timestamp
        )
        for (start, end), chat_cut in zip(pairwise(item_cuts), chat_cuts, strict=True)
    ]
    return cells, list(items[item_cuts[-1] :])


def _item_cut(chat_cut: int, chat_positions: Sequence[int], n_items: int) -> int:
    """Where a cut over the chat messages falls among all ``n_items`` items."""
    return chat_positions[chat_cut] if chat_cut < len(chat_positions) else n_items


# ---------------------------------------------------------------------------
# Incremental (step) path
# ---------------------------------------------------------------------------


def _needs_force_split(
    history: Sequence[ChatMessage],
    new: ChatMessage,
    hard_token_limit: int,
    hard_message_limit: int,
) -> bool:
    """Whether ``history + [new]`` reaches a hard limit with enough history to close a cell without the model.

    With fewer than two history messages the model still decides, so a single oversized message is never cut alone.
    """
    reaches_limit = (
        _count_history_tokens([*history, new]) >= hard_token_limit
        or len(history) + 1 >= hard_message_limit
    )
    return reaches_limit and len(history) >= 2


def _close_step(
    history: Sequence[ChatMessage], new: ChatMessage, *, bridge: bool
) -> tuple[MemCell, list[ConversationItem]]:
    """Close ``history`` into one cell; the next history is ``[history[-1], new]`` when bridging, else ``[new]``."""
    next_history: list[ConversationItem] = [history[-1], new] if bridge else [new]
    return _make_cell(history, timestamp=history[-1].timestamp), next_history


def _calculate_time_gap(
    history: Sequence[ChatMessage], new_messages: Sequence[ChatMessage]
) -> str:
    """Bucket the time between the last history message and the first new one into a prompt-ready label.

    Timestamps are milliseconds since the epoch; ``0`` reads as missing.
    """
    if not history or not new_messages:
        return "No time gap information available"
    last_ts_ms = history[-1].timestamp
    first_ts_ms = new_messages[0].timestamp
    if last_ts_ms == 0 or first_ts_ms == 0:
        return "No timestamp information available"
    total_seconds = (first_ts_ms - last_ts_ms) // 1000
    if total_seconds < 0:
        return "Time gap: Messages appear to be out of order"
    if total_seconds < 60:
        return f"Time gap: {_quantity(total_seconds, 'second')} (immediate response)"
    if total_seconds < 3600:
        return f"Time gap: {_quantity(total_seconds // 60, 'minute')} (recent conversation)"
    if total_seconds < 86_400:
        return f"Time gap: {_quantity(total_seconds // 3600, 'hour')} (same day, but significant pause)"
    return f"Time gap: {_quantity(total_seconds // 86_400, 'day')} (long gap, likely new conversation)"


def _quantity(count: int, unit: str) -> str:
    """``1 minute`` / ``2 minutes``: English plural for a whole-number count of ``unit``."""
    return f"{count} {unit}" if count == 1 else f"{count} {unit}s"


def _make_cell(items: Sequence[ConversationItem], *, timestamp: int) -> MemCell:
    """A cell over a copy of ``items``."""
    return MemCell(items=list(items), timestamp=timestamp)
