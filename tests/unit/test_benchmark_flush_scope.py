"""ADD sends a conversation in ``flush_scope`` units, each flushed once at its end.

``"session"`` (the default, and LoCoMo's) keeps every dataset session in its own everos
session. ``"conversation"`` sends the whole conversation as one session with a single
flush at the end, which is how the EverAlgo reference (d9fe22e) ingested LoCoMo; it is
an option, part of the ingest identity, and refused for datasets whose sessions carry
their own ids.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Literal

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import run as run_mod  # noqa: E402
from config import BenchmarkConfig  # noqa: E402


class _RecordingClient:
    """Record benchmark requests while returning successful EverOS responses."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        self.calls.append((path, payload))
        return 200, {"data": {}}


def _sessions(*sizes: int) -> list[dict]:
    ts = iter(range(1, 1 + sum(sizes)))
    return [
        {
            "session_idx": idx,
            "messages": [
                {"speaker": "Ann", "text": f"s{idx} m{i}", "timestamp_ms": next(ts)}
                for i in range(size)
            ],
        }
        for idx, size in enumerate(sizes, start=1)
    ]


def _send(
    sessions: list[dict],
    flush_scope: Literal["session", "conversation"],
    batch_size: int = 2,
) -> list[tuple]:
    client = _RecordingClient()
    run_mod.run_add_phase(
        client,
        sessions,
        conv_index=3,
        owner_id="ann_conv3",
        batch_size=batch_size,
        app_id="default",
        project_id="default",
        flush_scope=flush_scope,
    )
    return [
        (
            path.rsplit("/", 1)[-1],
            payload["session_id"],
            [m["content"][0]["text"] for m in payload.get("messages", [])],
        )
        for path, payload in client.calls
    ]


def test_session_scope_flushes_every_dataset_session() -> None:
    assert _send(_sessions(3, 1), "session") == [
        ("add", "locomo_conv3_s1", ["s1 m0", "s1 m1"]),
        ("add", "locomo_conv3_s1", ["s1 m2"]),
        ("flush", "locomo_conv3_s1", []),
        ("add", "locomo_conv3_s2", ["s2 m0"]),
        ("flush", "locomo_conv3_s2", []),
    ]


def test_conversation_scope_sends_one_session_and_flushes_once_at_the_end() -> None:
    """Batches run across session boundaries, in order, as the reference sent them."""
    assert _send(_sessions(3, 1), "conversation") == [
        ("add", "locomo_conv3", ["s1 m0", "s1 m1"]),
        ("add", "locomo_conv3", ["s1 m2", "s2 m0"]),
        ("flush", "locomo_conv3", []),
    ]


def test_conversation_scope_refuses_sessions_with_their_own_ids() -> None:
    """Gold mapping and readiness are keyed on those ids; one unit would lose them."""
    sessions = _sessions(1)
    sessions[0]["session_id"] = "session_1"
    with pytest.raises(ValueError, match="own session ids"):
        _send(sessions, "conversation")


def test_session_scope_is_the_default() -> None:
    assert BenchmarkConfig().flush_scope == "session"


def test_locomo_keeps_flushing_every_session() -> None:
    """Optional: LoCoMo's shipped config does not change how it is ingested."""
    assert BenchmarkConfig.from_toml("locomo").flush_scope == "session"


def test_the_run_ingests_with_the_configured_scope() -> None:
    src = inspect.getsource(run_mod.run_conversation)
    assert "flush_scope=config.flush_scope" in src
