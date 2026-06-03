"""Deterministic, human-readable ``message_id`` generation.

Format: ``m_<session_id>_<timestamp_ms>_<idx:03d>``.

Human-readable so logs / debugging / md entries stay greppable. Deterministic
so caller retries (same payload) produce the same ids — pipeline merge
naturally dedupes via the message_id PK in ``unprocessed_buffer``.
"""

from __future__ import annotations

_IDX_PAD = 3  # caller batches are capped at 500 messages (DTO limit), 3 digits cover it


def gen_message_id(session_id: str, timestamp_ms: int, idx: int) -> str:
    """Return ``m_<session_id>_<timestamp_ms>_<idx:03d>``."""
    return f"m_{session_id}_{timestamp_ms}_{idx:0{_IDX_PAD}d}"
