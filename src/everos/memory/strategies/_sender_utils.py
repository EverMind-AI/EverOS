"""Shared sender-collection helpers for strategy modules."""

from __future__ import annotations

from everalgo.types import MemCell


def collect_user_sender_ids(memcell: MemCell) -> list[str]:
    """Distinct ``role='user'`` sender_ids in stable sorted order.

    User-side strategies may receive agent-trajectory cells that include
    ``ToolCallRequest`` items alongside chat messages. Those items expose
    ``sender_id`` but not ``role``, so sender discovery must probe the
    attribute defensively instead of assuming every item is a ChatMessage.
    """

    sender_ids: set[str] = set()
    for item in memcell.items:
        if getattr(item, "role", None) != "user":
            continue
        sid = getattr(item, "sender_id", None)
        if sid:
            sender_ids.add(sid)
    return sorted(sender_ids)
