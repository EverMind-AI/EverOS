from __future__ import annotations

import sys
from pathlib import Path

USE_CASE_ROOT = (
    Path(__file__).resolve().parents[4] / "use-cases" / "cursor-agent-memory"
)
if str(USE_CASE_ROOT) not in sys.path:
    sys.path.insert(0, str(USE_CASE_ROOT))

from hooklib.everos_client import message_item  # noqa: E402


def test_message_item_shape() -> None:
    msg = message_item(
        sender_id="cursor-user",
        role="user",
        content="hello",
        timestamp_ms=1_700_000_000_000,
    )
    assert msg["sender_id"] == "cursor-user"
    assert msg["role"] == "user"
    assert msg["content"] == "hello"
    assert msg["timestamp"] == 1_700_000_000_000
