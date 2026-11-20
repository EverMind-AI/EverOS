"""Deterministic, human-readable ``message_id`` generation.

Format: ``m_<session_id>_<timestamp_ms>_<digest>``.

``digest`` is 8 hex chars of SHA-256 over who said what (``sender_id``, ``role``,
``content``, ``tool_call_id``, ``tool_calls``). Nothing in the id depends on where
the message sits in its request, so a client that resends a batch -- whole, split
or regrouped -- produces the same ids, and the boundary stage drops what it
already buffered or cut. Two different messages in one millisecond get two ids.

A message repeated verbatim within one request (same sender, words and
millisecond) is told apart by how many identical ones came before it in that
request; only then does the count enter the digest. Order among messages that
share a millisecond is not in the id: the boundary stage keeps arrival order.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

_IDENTITY_FIELDS = ("sender_id", "role", "content", "tool_call_id", "tool_calls")


def _digest(message: Mapping[str, Any], repeat: int) -> str:
    basis: list[Any] = [message.get(field) for field in _IDENTITY_FIELDS]
    if repeat:
        basis.append(repeat)
    raw = json.dumps(basis, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


def message_ids(session_id: str, messages: Sequence[Mapping[str, Any]]) -> list[str]:
    """The id of each message of one request, in order."""
    seen: Counter[str] = Counter()
    ids = []
    for message in messages:
        ts_ms = int(message["timestamp"])
        first = f"m_{session_id}_{ts_ms}_{_digest(message, 0)}"
        ids.append(
            first
            if not seen[first]
            else f"m_{session_id}_{ts_ms}_{_digest(message, seen[first])}"
        )
        seen[first] += 1
    return ids
