"""Deterministic, human-readable ``message_id`` generation.

Format: ``m_<session_id>_<timestamp_ms>_<nth:03d>_<digest>``.

- ``nth`` counts the messages of one request that share the timestamp, in
  request order, so those keep their order when rows sort by
  ``(timestamp, message_id)``.
- ``digest`` is 8 hex chars of SHA-256 over who said what (``sender_id``,
  ``role``, ``content``, ``tool_call_id``, ``tool_calls``), so two different
  messages sent in separate requests with the same millisecond get two ids.

Neither part depends on where the message sits in its request, so a client
that resends a batch -- whole, split or regrouped -- produces the same ids,
and the boundary stage drops what it already buffered or cut.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

_IDENTITY_FIELDS = ("sender_id", "role", "content", "tool_call_id", "tool_calls")


def _digest(message: Mapping[str, Any]) -> str:
    basis = [message.get(field) for field in _IDENTITY_FIELDS]
    raw = json.dumps(basis, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]


def message_ids(session_id: str, messages: Sequence[Mapping[str, Any]]) -> list[str]:
    """The id of each message of one request, in order."""
    nth: Counter[int] = Counter()
    ids = []
    for message in messages:
        ts_ms = int(message["timestamp"])
        ids.append(f"m_{session_id}_{ts_ms}_{nth[ts_ms]:03d}_{_digest(message)}")
        nth[ts_ms] += 1
    return ids
