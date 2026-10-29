"""Agent-mode memorize integration tests.

Covers the agent branches that ``test_memorize_integration.py`` skips:

- :class:`everos.service.MemoryService` agent dispatch (user + agent
  pipelines per cell)
- :class:`everos.memory.extract.boundary.BoundaryDetection` in agent mode
  (tool rows kept in the cells, hidden from the model)
- :mod:`memory.extract.pipeline.agent_memory.AgentMemoryPipeline` end-to-end

Self-contained: the chat-baseline file keeps its fixture local, so we
copy the minimum scaffolding rather than refactor it into a shared
conftest. The model is :func:`tests.fakes.memorize_llm`.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import sqlite3
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from everalgo.testing import FakeLLMClient

from everos.infra.persistence import MemoryRoot
from everos.service import (
    MemorizeResult,
    MemoryService,
    build_application,
    build_providers,
)
from tests.fakes import memorize_llm
from tests.fakes.storage import Storage

_under_test: dict[str, MemoryService] = {}
"""The memory facade of the application the ``memorize_env`` fixture built."""


async def memorize(payload: dict[str, Any], **kwargs: Any) -> MemorizeResult:
    """``MemoryService.memorize`` of the application under test."""
    return await _under_test["memory"].memorize(payload, **kwargs)


def _msg(
    role: str,
    content: str,
    *,
    sender_id: str = "u_alice",
    timestamp: int = 1_700_000_000_000,
    tool_calls: list[dict] | None = None,
    tool_call_id: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sender_id": sender_id,
        "role": role,
        "content": content,
        "timestamp": timestamp,
    }
    if tool_calls is not None:
        out["tool_calls"] = tool_calls
    if tool_call_id is not None:
        out["tool_call_id"] = tool_call_id
    return out


def _user(content: str, ts: int, *, sender: str = "u_alice") -> dict[str, Any]:
    return _msg("user", content, sender_id=sender, timestamp=ts)


def _assistant(content: str, ts: int) -> dict[str, Any]:
    return _msg("assistant", content, sender_id="assistant", timestamp=ts)


def _memcell_rows(tmp_path: Path) -> list[sqlite3.Row]:
    db = tmp_path / ".index" / "sqlite" / "system.db"
    if not db.is_file():
        return []
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM memcell ORDER BY timestamp"))
    finally:
        conn.close()


@pytest_asyncio.fixture
async def memorize_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> AsyncIterator[Callable[..., Any]]:
    """Same shape as the chat-baseline fixture; ``mode`` defaults to ``agent``."""
    monkeypatch.setattr(
        MemoryRoot, "resolve", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ome.toml").write_text("# test\n")

    af_mod = importlib.import_module("everos.memory.strategies.extract_atomic_facts")
    fs_mod = importlib.import_module("everos.memory.strategies.extract_foresight")
    ac_mod = importlib.import_module("everos.memory.strategies.extract_agent_case")

    started: dict[str, Any] = {"engine": None}

    async def _setup(*, mode: str = "agent", fake_llm: FakeLLMClient) -> None:
        monkeypatch.setenv("EVEROS_MEMORIZE__MODE", mode)
        monkeypatch.setenv("EVEROS_LLM__API_KEY", "fake-key")
        monkeypatch.setenv("EVEROS_LLM__BASE_URL", "https://fake.example.com")

        from everos.config import load_settings

        load_settings.cache_clear()

        await storage.create_schema()

        # Silence OME strategies so agent_case / atomic / foresight don't
        # try real extraction logic during these tests.
        noop = AsyncMock(return_value=[])
        for mod in (af_mod, fs_mod, ac_mod):
            extractor_attr = next(
                (n for n in dir(mod) if n.endswith("Extractor")), None
            )
            if extractor_attr:
                monkeypatch.setattr(
                    mod,
                    extractor_attr,
                    lambda *a, **k: type("M", (), {"aextract": noop})(),
                )

        settings = load_settings()
        application = build_application(
            settings,
            providers=dataclasses.replace(build_providers(settings), llm=fake_llm),
        )
        _under_test["memory"] = application.memory
        engine = application._engine
        await engine.start()
        started["engine"] = engine

    yield _setup

    if started.get("engine") is not None:
        await started["engine"].stop()


# ── Tests ────────────────────────────────────────────────────────────


async def test_agent_mode_two_user_assistant_msgs(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Agent mode happy path: one cell, both user + agent pipelines fire."""
    fake = memorize_llm([[]])
    await memorize_env(mode="agent", fake_llm=fake)

    result = await memorize(
        {
            "session_id": "test_agent_basic",
            "messages": [
                _user("hello", 1_700_000_000_000),
                _assistant("hi there", 1_700_000_001_000),
            ],
        },
        is_final=True,
    )
    assert isinstance(result, MemorizeResult)
    assert result.status == "extracted"

    rows = _memcell_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["raw_type"] == "AgentTrajectory"


async def test_agent_mode_preserves_tool_items(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Agent mode keeps ``role=tool`` rows inside the cell (chat mode drops them)."""
    fake = memorize_llm([[]])
    await memorize_env(mode="agent", fake_llm=fake)

    payload = {
        "session_id": "test_agent_tools",
        "messages": [
            _user("debug this", 1_700_000_000_000),
            _msg(
                "assistant",
                "calling tool",
                timestamp=1_700_000_001_000,
                tool_calls=[
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "x", "arguments": "{}"},
                    }
                ],
            ),
            _msg(
                "tool",
                "result",
                sender_id="tool",
                timestamp=1_700_000_002_000,
                tool_call_id="c1",
            ),
            _assistant("here's the answer", 1_700_000_003_000),
        ],
    }
    result = await memorize(payload, is_final=True)
    assert result.status == "extracted"

    rows = _memcell_rows(tmp_path)
    assert len(rows) == 1
    ids = json.loads(rows[0]["message_ids_json"])
    # All four preserved in agent mode (chat mode would have 2).
    assert len(ids) == 4


async def test_agent_mode_dispatch_no_double_insert(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Dual pipeline dispatch must not double-insert the memcell row."""
    fake = memorize_llm([[]])
    await memorize_env(mode="agent", fake_llm=fake)

    await memorize(
        {
            "session_id": "test_agent_dispatch",
            "messages": [
                _user("u1", 1_700_000_000_000),
                _assistant("a1", 1_700_000_001_000),
                _user("u2", 1_700_000_002_000),
                _assistant("a2", 1_700_000_003_000),
            ],
        },
        is_final=True,
    )

    rows = _memcell_rows(tmp_path)
    assert len(rows) == 1  # boundary stage owns the ledger
    payload = json.loads(rows[0]["payload_json"])
    assert len(payload["items"]) == 4
