"""EverMemBench keeps group scoping inside the benchmark adapter."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

run = importlib.import_module("run")
evermembench = importlib.import_module("adapters.evermembench")


class _RecordingClient:
    """Record benchmark requests while returning successful EverOS responses."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        self.calls.append((path, payload))
        if path.endswith("/search"):
            return 200, {"data": {"episodes": [], "profiles": []}}
        return 200, {"data": {}}


def test_add_and_search_share_one_topic_owner() -> None:
    """Real speaker names must not split one benchmark topic across owners."""
    owner_id = evermembench.owner_of({"index": "01"}, "unused")
    client = _RecordingClient()
    sessions = [
        {
            "session_idx": 1,
            "session_id": "01_Group_1_2026-01-01",
            "messages": [
                {
                    "speaker": "Alice Chen",
                    "text": "Alice joined the project.",
                    "timestamp_ms": 1,
                },
                {
                    "speaker": "Bob Li",
                    "text": "Bob reviewed the plan.",
                    "timestamp_ms": 2,
                },
            ],
        }
    ]

    run.run_add_phase(
        client,
        sessions,
        conv_index=0,
        owner_id=owner_id,
        batch_size=50,
        app_id="default",
        project_id="default",
    )
    run._search_one(
        0,
        {"question": "Who reviewed the plan?", "answer": "Bob Li"},
        client=client,
        method="hybrid",
        top_k=20,
        owner_id=owner_id,
        app_id="default",
        project_id="default",
    )

    add_payload = next(
        payload for path, payload in client.calls if path.endswith("/add")
    )
    assert {message["sender_id"] for message in add_payload["messages"]} == {"01"}
    assert [message["sender_name"] for message in add_payload["messages"]] == [
        "Alice Chen",
        "Bob Li",
    ]

    search_payload = next(
        payload for path, payload in client.calls if path.endswith("/search")
    )
    assert search_payload["user_id"] == "01"
    assert "include_profile" not in search_payload
