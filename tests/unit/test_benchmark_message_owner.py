"""ADD files each message under its sender or, with ``message_owner = "eval_owner"``,
under the queried owner.

LoCoMo is ingested per speaker by default. The EverAlgo reference (d9fe22e) kept one
pool per conversation, so the LoCoMo config files every message under the queried
speaker: a memcell where only the other speaker talks then still reaches the partition
the questions are asked of.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import adapters  # noqa: E402
import run as run_mod  # noqa: E402
from config import BenchmarkConfig  # noqa: E402

_LOCOMO = adapters.get("locomo")


class _RecordingClient:
    """Record benchmark requests while returning successful EverOS responses."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        self.calls.append((path, payload))
        return 200, {"data": {}}


def _senders(message_owner: Literal["sender", "eval_owner"]) -> set[str]:
    config = BenchmarkConfig(message_owner=message_owner)
    client = _RecordingClient()
    run_mod.run_add_phase(
        client,
        [
            {
                "session_idx": 1,
                "messages": [
                    {"speaker": "Ann", "text": "hi", "timestamp_ms": 1},
                    {"speaker": "Bob", "text": "hey", "timestamp_ms": 2},
                ],
            }
        ],
        conv_index=3,
        owner_id="ann_conv3",
        batch_size=50,
        app_id="default",
        project_id="default",
        sender_id_of=run_mod._sender_id_rule(config, _LOCOMO),
    )
    return {
        m["sender_id"]
        for path, payload in client.calls
        if path.endswith("/add")
        for m in payload["messages"]
    }


def test_sender_files_each_turn_under_its_speaker() -> None:
    assert _senders("sender") == {"ann_conv3", "bob_conv3"}


def test_eval_owner_files_every_turn_under_the_queried_owner() -> None:
    assert _senders("eval_owner") == {"ann_conv3"}


def test_an_adapter_without_a_sender_rule_files_under_the_owner() -> None:
    assert run_mod._sender_id_rule(BenchmarkConfig(), object()) is None


def test_sender_is_the_default() -> None:
    assert BenchmarkConfig().message_owner == "sender"


def test_locomo_keeps_one_pool_per_conversation_like_the_reference() -> None:
    assert BenchmarkConfig.from_toml("locomo").message_owner == "eval_owner"
