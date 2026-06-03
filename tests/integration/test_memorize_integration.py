"""End-to-end memorize integration tests.

Drives ``service.memorize.memorize()`` with a ``FakeLLMClient`` so the
full chain (ingest → boundary → user / agent pipeline → md + OME emit)
runs without real LLM calls. Each test isolates state by:

- redirecting ``MemoryRoot.default()`` to a ``tmp_path``
- resetting service-layer lazy singletons
- starting / stopping a per-test ``OfflineEngine``
- patching ``get_llm_client`` (boundary + strategies) onto a fake

OME strategies (atomic / foresight) are silenced via ``mock_aextract`` so
this test focuses on the synchronous boundary + pipeline + md path —
strategy dispatch correctness already has its own coverage in
``test_ome_strategies_integration.py``.
"""

from __future__ import annotations

import importlib
import json
import sqlite3
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from sqlmodel import SQLModel

from everos.core.persistence import MemoryRoot
from everos.service.memorize import MemorizeResult, memorize

# ---------------------------------------------------------------------------
# Canned LLM responses
# ---------------------------------------------------------------------------


def _boundary_response(boundaries: list[int]) -> str:
    """Build a ``detect_boundaries`` JSON response (algo schema)."""
    payload = {
        "reasoning": "test",
        "boundaries": boundaries,
        "should_wait": False,
    }
    return json.dumps(payload)


def _episode_response(title: str = "Test Subject", content: str = "Test body") -> str:
    """Build an ``EpisodeExtractor`` JSON response (algo schema)."""
    return json.dumps({"title": title, "content": content})


def _make_fake_llm(
    boundary_responses: list[list[int]] | None = None,
    *,
    episode_title: str = "Test Subject",
    episode_content: str = "Test body",
) -> FakeLLMClient:
    """Build a ``FakeLLMClient`` that dispatches by prompt fingerprint.

    Pops one ``boundaries=...`` from ``boundary_responses`` per boundary
    prompt seen; every episode prompt returns the same canned
    ``{title, content}``.
    """
    boundary_queue: list[list[int]] = list(boundary_responses or [])

    def handler(messages: list[LLMChatMessage], **_: Any) -> ChatResponse:
        prompt = messages[0].content
        if "boundaries" in prompt.lower() or "memcell" in prompt.lower():
            cuts = boundary_queue.pop(0) if boundary_queue else []
            return ChatResponse(content=_boundary_response(cuts), model="fake")
        # Fall through to episode (also catches atomic/foresight prompts —
        # they'll return success-but-empty in their mocked extractor below).
        return ChatResponse(
            content=_episode_response(episode_title, episode_content),
            model="fake",
        )

    return FakeLLMClient(handler=handler)


# ---------------------------------------------------------------------------
# Shared setup fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def memorize_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Callable[..., AsyncMock]]:
    """Yield a builder that configures a clean memorize environment.

    Usage::

        async def test_x(memorize_env):
            await memorize_env(mode="chat", fake_llm=_make_fake_llm([...]))
            outcome = await memorize({"session_id": "s", "messages": [...]})

    The builder must be called exactly once per test (it primes singletons
    + starts the OME engine). Teardown stops the engine and disposes the
    sqlite engine.
    """
    monkeypatch.setattr(
        MemoryRoot, "default", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)

    svc = importlib.import_module("everos.service.memorize")
    af_mod = importlib.import_module("everos.memory.strategies.extract_atomic_facts")
    fs_mod = importlib.import_module("everos.memory.strategies.extract_foresight")
    client_mod = importlib.import_module("everos.component.llm.client")

    # Reset singletons.
    for attr in (
        "_episode_writer",
        "_prompt_loader",
        "_user_pipeline",
        "_agent_pipeline",
        "_ome_engine",
    ):
        monkeypatch.setattr(svc, attr, None, raising=False)
    monkeypatch.setattr(client_mod, "_llm_client", None, raising=False)
    monkeypatch.setattr(af_mod, "_writer", None, raising=False)
    monkeypatch.setattr(fs_mod, "_writer", None, raising=False)

    started: dict[str, Any] = {"engine": None, "sqlite_engine": None}

    async def _setup(
        *,
        mode: str = "chat",
        fake_llm: FakeLLMClient,
        hard_token_limit: int = 65536,
        hard_msg_limit: int = 500,
    ) -> None:
        # Provide a non-None API key + base_url so get_llm_client doesn't
        # raise; we replace the cached singleton with our fake right after.
        monkeypatch.setenv("EVEROS_MEMORIZE__MODE", mode)
        monkeypatch.setenv("EVEROS_LLM__API_KEY", "fake-key")
        monkeypatch.setenv("EVEROS_LLM__BASE_URL", "https://fake.example.com")
        monkeypatch.setenv(
            "EVEROS_BOUNDARY_DETECTION__HARD_TOKEN_LIMIT", str(hard_token_limit)
        )
        monkeypatch.setenv(
            "EVEROS_BOUNDARY_DETECTION__HARD_MSG_LIMIT", str(hard_msg_limit)
        )
        from everos.config import load_settings

        load_settings.cache_clear()

        # Replace the cached client singleton with our fake so get_llm_client
        # returns the fake on subsequent calls.
        monkeypatch.setattr(client_mod, "_llm_client", fake_llm)

        # Build sqlite schema.
        from everos.infra.persistence.sqlite import dispose_engine, get_engine

        db_engine = get_engine()
        async with db_engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        started["sqlite_engine"] = (get_engine, dispose_engine)

        # Mock the OME extractors so the async strategy chain is a no-op
        # (the strategy itself still runs; it just sees no facts/foresights).
        mock_af = AsyncMock(return_value=[])
        mock_fs = AsyncMock(return_value=[])
        monkeypatch.setattr(
            af_mod,
            "AtomicFactExtractor",
            lambda *a, **k: type("M", (), {"aextract": mock_af})(),
        )
        monkeypatch.setattr(
            fs_mod,
            "ForesightExtractor",
            lambda *a, **k: type("M", (), {"aextract": mock_fs})(),
        )

        engine = svc._get_engine()
        await engine.start()
        started["engine"] = engine

    yield _setup

    if started["engine"] is not None:
        await started["engine"].stop()
    if started["sqlite_engine"] is not None:
        _, dispose = started["sqlite_engine"]
        await dispose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _assistant(content: str, ts: int, *, sender: str = "assistant") -> dict[str, Any]:
    return _msg("assistant", content, sender_id=sender, timestamp=ts)


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


def _buffer_count(tmp_path: Path) -> int:
    db = tmp_path / ".index" / "sqlite" / "system.db"
    if not db.is_file():
        return 0
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM unprocessed_buffer WHERE track='memorize'"
        ).fetchone()[0]
    finally:
        conn.close()


def _episode_paths(tmp_path: Path) -> list[Path]:
    base = tmp_path / "default_app" / "default_project" / "users"
    return sorted(base.rglob("episode-*.md"))


# ---------------------------------------------------------------------------
# Happy path baseline
# ---------------------------------------------------------------------------


async def test_chat_baseline_two_msgs_one_cell(
    tmp_path: Path,
    memorize_env: Callable[..., Any],
) -> None:
    """2 messages → flush forces them into 1 cell + 1 Episode + 1 memcell row."""
    fake = _make_fake_llm(boundary_responses=[[]])  # no internal cuts
    await memorize_env(mode="chat", fake_llm=fake)

    payload = {
        "session_id": "test_chat_1",
        "messages": [
            _user("hello", 1_700_000_000_000),
            _assistant("hi there", 1_700_000_001_000),
        ],
    }
    result = await memorize(payload, is_final=True)

    assert isinstance(result, MemorizeResult)
    assert result.status == "extracted"
    assert result.message_count == 2

    rows = _memcell_rows(tmp_path)
    assert len(rows) == 1
    assert rows[0]["track"] == "memorize"
    assert rows[0]["raw_type"] == "Conversation"
    # MemCell has no single owner — sender_ids carries the participants.
    assert "u_alice" in json.loads(rows[0]["sender_ids_json"])

    assert _buffer_count(tmp_path) == 0

    md_files = _episode_paths(tmp_path)
    assert len(md_files) == 1
    body = md_files[0].read_text()
    assert "Test Subject" in body
    assert "Test body" in body


# ---------------------------------------------------------------------------
# Input-shape boundary cases (6)
# ---------------------------------------------------------------------------


async def test_empty_batch_non_final_is_skipped(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """``messages=[]`` + ``is_final=False`` → skipped, no side effects."""
    await memorize_env(mode="chat", fake_llm=_make_fake_llm())

    result = await memorize(
        {"session_id": "test_empty_nonfinal", "messages": []}, is_final=False
    )
    assert result.status == "accumulated"
    assert result.message_count == 0
    assert _memcell_rows(tmp_path) == []
    assert _episode_paths(tmp_path) == []


async def test_empty_batch_final_drains_empty_buffer(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """``messages=[]`` + ``is_final=True`` on virgin session → no cells, no md."""
    await memorize_env(mode="chat", fake_llm=_make_fake_llm())

    result = await memorize(
        {"session_id": "test_empty_final", "messages": []}, is_final=True
    )
    assert result.status == "accumulated"
    assert _memcell_rows(tmp_path) == []
    assert _episode_paths(tmp_path) == []


async def test_assistant_only_batch_accumulates(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """No role=user message → boundary stage parks everything in buffer."""
    fake = _make_fake_llm(boundary_responses=[])  # no LLM call expected
    await memorize_env(mode="chat", fake_llm=fake)

    result = await memorize(
        {
            "session_id": "test_asst_only",
            "messages": [
                _assistant("hi", 1_700_000_000_000),
                _assistant("anyone here?", 1_700_000_001_000),
            ],
        },
        is_final=False,
    )
    assert result.status == "accumulated"
    assert _memcell_rows(tmp_path) == []
    assert _buffer_count(tmp_path) == 2  # parked in buffer


async def test_single_user_message_accumulates(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Single user msg → boundary returns no cells (need conversation) → buffer it."""
    fake = _make_fake_llm(boundary_responses=[[]])  # boundary called, no cuts
    await memorize_env(mode="chat", fake_llm=fake)

    result = await memorize(
        {
            "session_id": "test_single",
            "messages": [_user("hello?", 1_700_000_000_000)],
        },
        is_final=False,
    )
    assert result.status == "accumulated"
    assert _memcell_rows(tmp_path) == []
    assert _buffer_count(tmp_path) == 1


async def test_chat_mode_filters_tool_messages(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Chat mode drops ``role=tool`` + assistant-with-tool_calls pre-boundary."""
    fake = _make_fake_llm(boundary_responses=[[]])
    await memorize_env(mode="chat", fake_llm=fake)

    result = await memorize(
        {
            "session_id": "test_chat_filter",
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
        },
        is_final=True,
    )
    # After filter: 1 user + 1 assistant text = 2 msgs → 1 cell on flush.
    assert result.status == "extracted"
    rows = _memcell_rows(tmp_path)
    assert len(rows) == 1
    ids = json.loads(rows[0]["message_ids_json"])
    assert len(ids) == 2  # tool + assistant-with-tool_calls dropped


async def test_duplicate_message_id_dedup_across_adds(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Same message replayed across two ``/add`` calls is deduped by message_id."""
    fake = _make_fake_llm(boundary_responses=[[], []])  # 2 boundary calls, both empty
    await memorize_env(mode="chat", fake_llm=fake)

    # message_id is derived from (session_id, ts_ms, idx); same payload twice
    # produces the same id, so the second add should be a no-op insert.
    payload = {
        "session_id": "test_dedup",
        "messages": [
            _user("hi", 1_700_000_000_000),
            _assistant("hi back", 1_700_000_001_000),
        ],
    }
    await memorize(payload, is_final=False)
    await memorize(payload, is_final=False)  # replay
    await memorize({"session_id": "test_dedup", "messages": []}, is_final=True)

    rows = _memcell_rows(tmp_path)
    assert len(rows) == 1
    ids = json.loads(rows[0]["message_ids_json"])
    assert len(ids) == 2  # not 4 — dedup worked
    assert len(set(ids)) == 2  # unique


# ---------------------------------------------------------------------------
# Hard-limit cases (2)
# ---------------------------------------------------------------------------


async def test_hard_msg_limit_force_split(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Exceeding ``hard_msg_limit`` triggers a force-split before the LLM call."""
    fake = _make_fake_llm(boundary_responses=[[]])  # LLM call after force-split
    # hard_msg_limit=3 → batch of 5 msgs forces ~1 split before LLM.
    await memorize_env(
        mode="chat", fake_llm=fake, hard_msg_limit=3, hard_token_limit=10_000
    )

    msgs = [
        _user(f"u{i}", 1_700_000_000_000 + i * 1000, sender="u_alice")
        if i % 2 == 0
        else _assistant(f"a{i}", 1_700_000_000_000 + i * 1000)
        for i in range(5)
    ]
    result = await memorize(
        {"session_id": "test_hardmsg", "messages": msgs}, is_final=True
    )
    assert result.status == "extracted"
    rows = _memcell_rows(tmp_path)
    # Force-split + LLM final → at least 2 cells (force + remaining).
    assert len(rows) >= 2


async def test_hard_token_limit_force_split(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Exceeding ``hard_token_limit`` triggers a force-split (token-based)."""
    fake = _make_fake_llm(boundary_responses=[[]])
    # Very small token budget → even tiny content triggers force-split.
    await memorize_env(
        mode="chat", fake_llm=fake, hard_msg_limit=500, hard_token_limit=20
    )

    msgs = [
        _user("a" * 200, 1_700_000_000_000, sender="u_alice"),
        _assistant("b" * 200, 1_700_000_001_000),
        _user("c" * 200, 1_700_000_002_000, sender="u_alice"),
        _assistant("d" * 200, 1_700_000_003_000),
    ]
    result = await memorize(
        {"session_id": "test_hardtok", "messages": msgs}, is_final=True
    )
    assert result.status == "extracted"
    assert len(_memcell_rows(tmp_path)) >= 2


# ---------------------------------------------------------------------------
# Flush state-machine cases (4)
# ---------------------------------------------------------------------------


async def test_flush_on_virgin_session_is_noop(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Flush a session that never received ``/add`` — should not crash."""
    await memorize_env(mode="chat", fake_llm=_make_fake_llm())

    result = await memorize(
        {"session_id": "test_virgin_flush", "messages": []}, is_final=True
    )
    assert result.status == "accumulated"
    assert _memcell_rows(tmp_path) == []


async def test_add_then_flush_then_add(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """After flush drains the buffer, a follow-up ``/add`` still works."""
    fake = _make_fake_llm(boundary_responses=[[], []])
    await memorize_env(mode="chat", fake_llm=fake)

    sid = "test_add_flush_add"
    await memorize(
        {
            "session_id": sid,
            "messages": [
                _user("first", 1_700_000_000_000),
                _assistant("ack", 1_700_000_001_000),
            ],
        },
        is_final=False,
    )
    await memorize({"session_id": sid, "messages": []}, is_final=True)

    rows_after_flush_1 = len(_memcell_rows(tmp_path))
    assert rows_after_flush_1 == 1

    # Second turn after the flush.
    await memorize(
        {
            "session_id": sid,
            "messages": [
                _user("second turn", 1_700_000_010_000),
                _assistant("ok", 1_700_000_011_000),
            ],
        },
        is_final=True,
    )
    assert len(_memcell_rows(tmp_path)) == 2  # cumulative


async def test_consecutive_flushes_second_is_noop(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Flush twice in a row — second call finds empty buffer, no-ops."""
    fake = _make_fake_llm(boundary_responses=[[]])
    await memorize_env(mode="chat", fake_llm=fake)

    sid = "test_double_flush"
    await memorize(
        {
            "session_id": sid,
            "messages": [
                _user("hi", 1_700_000_000_000),
                _assistant("ok", 1_700_000_001_000),
            ],
        },
        is_final=False,
    )
    res1 = await memorize({"session_id": sid, "messages": []}, is_final=True)
    res2 = await memorize({"session_id": sid, "messages": []}, is_final=True)

    assert res1.status == "extracted"
    assert res2.status == "accumulated"  # nothing left
    assert len(_memcell_rows(tmp_path)) == 1


async def test_flush_drains_assistant_only_buffer(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Buffer with only assistant messages: flush still forces them into a cell."""
    fake = _make_fake_llm(boundary_responses=[[]])
    await memorize_env(mode="chat", fake_llm=fake)

    sid = "test_asst_then_flush"
    # Two assistant-only adds → both park in buffer.
    await memorize(
        {
            "session_id": sid,
            "messages": [_assistant("a1", 1_700_000_000_000)],
        },
        is_final=False,
    )
    await memorize(
        {
            "session_id": sid,
            "messages": [_assistant("a2", 1_700_000_001_000)],
        },
        is_final=False,
    )
    assert _buffer_count(tmp_path) == 2

    # Add a user message + flush — boundary should now run.
    result = await memorize(
        {
            "session_id": sid,
            "messages": [_user("anyone there?", 1_700_000_002_000)],
        },
        is_final=True,
    )
    assert result.status == "extracted"
    assert _buffer_count(tmp_path) == 0


# ---------------------------------------------------------------------------
# Multi-session cases (2)
# ---------------------------------------------------------------------------


async def test_two_sessions_are_isolated(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Two session_ids share the engine but their buffers / cells stay separate."""
    fake = _make_fake_llm(boundary_responses=[[], []])  # 1 per session
    await memorize_env(mode="chat", fake_llm=fake)

    await memorize(
        {
            "session_id": "sess_A",
            "messages": [
                _user("hi from A", 1_700_000_000_000, sender="u_alice"),
                _assistant("ack A", 1_700_000_001_000),
            ],
        },
        is_final=True,
    )
    await memorize(
        {
            "session_id": "sess_B",
            "messages": [
                _user("hi from B", 1_700_000_010_000, sender="u_bob"),
                _assistant("ack B", 1_700_000_011_000),
            ],
        },
        is_final=True,
    )

    rows = _memcell_rows(tmp_path)
    assert len(rows) == 2
    sessions = sorted(r["session_id"] for r in rows)
    assert sessions == ["sess_A", "sess_B"]
    # MemCell has no single owner — sender_ids carries who participated.
    senders = {r["session_id"]: json.loads(r["sender_ids_json"]) for r in rows}
    assert "u_alice" in senders["sess_A"]
    assert "u_bob" in senders["sess_B"]


async def test_same_session_multi_add_concatenates(
    tmp_path: Path, memorize_env: Callable[..., Any]
) -> None:
    """Multiple adds on the same session accumulate in one buffer until flushed."""
    fake = _make_fake_llm(boundary_responses=[[], [], []])
    await memorize_env(mode="chat", fake_llm=fake)

    sid = "test_multi_add"
    for i in range(3):
        await memorize(
            {
                "session_id": sid,
                "messages": [
                    _user(f"u{i}", 1_700_000_000_000 + i * 2000),
                    _assistant(f"a{i}", 1_700_000_001_000 + i * 2000),
                ],
            },
            is_final=False,
        )
    # Buffer should have 6 messages now (no boundary cuts).
    assert _buffer_count(tmp_path) == 6

    result = await memorize({"session_id": sid, "messages": []}, is_final=True)
    assert result.status == "extracted"
    rows = _memcell_rows(tmp_path)
    assert len(rows) == 1  # one cell from the flush
    ids = json.loads(rows[0]["message_ids_json"])
    assert len(ids) == 6  # all 6 messages folded in
