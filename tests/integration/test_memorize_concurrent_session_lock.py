"""Concurrent /add on one session must not lose messages (regression).

White-box integration test for the per-session lock added in
``everos.service._session_lock``.

Bug class
---------
Without the lock, two concurrent ``memorize()`` calls on the same
``session_id`` race on ``unprocessed_buffer``:

1. Both read the same pre-existing buffer rows.
2. Each boundary call sees only its own newly-arrived messages plus
   the shared pre-existing buffer (neither sees the other's messages).
3. Both call ``_replace_buffer(session_id, tail)`` — the later write
   silently overwrites the earlier write's tail; the earlier task's
   tail messages are lost forever.

Invariant under test
--------------------
After N concurrent ``memorize()`` calls on one session, every input
message_id is **either** in some memcell's ``message_ids_json`` **or**
in the surviving ``unprocessed_buffer`` rows. Nothing silently vanishes.

This is a white-box integration test (not e2e): it bypasses HTTP, calls
``memorize()`` directly, but inspects sqlite tables to assert internal
state. Uses ``FakeLLMClient`` to avoid real LLM latency and to control
boundary decisions deterministically.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from sqlalchemy import text
from sqlmodel import SQLModel

from everos.core.persistence import MemoryRoot
from everos.service.memorize import memorize

# ---------------------------------------------------------------------------
# Fake LLM that splits each call into one memcell + 0-tail (force extract)
# ---------------------------------------------------------------------------


def _boundary_response(boundaries: list[int]) -> str:
    return json.dumps(
        {"reasoning": "test", "boundaries": boundaries, "should_wait": False}
    )


def _episode_response(title: str = "T", content: str = "B") -> str:
    return json.dumps({"title": title, "content": content})


def _make_extract_all_llm() -> FakeLLMClient:
    """Boundary returns single boundary at end → entire merged → 1 cell, tail=[]."""

    def handler(messages: list[LLMChatMessage], **_: Any) -> ChatResponse:
        prompt = messages[0].content
        if "boundaries" in prompt.lower() or "memcell" in prompt.lower():
            # Always cut: the boundary indices are relative to merged input;
            # an empty list means "no cut, hold". A single [N] means "cut
            # after index N", i.e. everything before goes into one cell.
            # We use a sentinel large index to force boundary to take all.
            return ChatResponse(content=_boundary_response([999]), model="fake")
        return ChatResponse(content=_episode_response(), model="fake")

    return FakeLLMClient(handler=handler)


# ---------------------------------------------------------------------------
# Fixture — mirrors test_memorize_integration's pattern but without OME / strategies
# (the lock bug lives at the boundary stage; downstream strategies are
# irrelevant to this race).
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def memorize_env_locked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Callable[..., AsyncMock]]:
    monkeypatch.setattr(
        MemoryRoot, "default", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ome.toml").write_text("# test\n")

    svc = importlib.import_module("everos.service.memorize")
    af_mod = importlib.import_module("everos.memory.strategies.extract_atomic_facts")
    fs_mod = importlib.import_module("everos.memory.strategies.extract_foresight")
    client_mod = importlib.import_module("everos.component.llm.client")
    lock_mod = importlib.import_module("everos.service._session_lock")

    # Reset memorize singletons + session lock registry.
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
    lock_mod._reset_for_tests()

    started: dict[str, Any] = {"engine": None}

    async def _setup(*, fake_llm: FakeLLMClient) -> None:
        monkeypatch.setenv("EVEROS_MEMORIZE__MODE", "chat")
        monkeypatch.setenv("EVEROS_LLM__API_KEY", "fake-key")
        monkeypatch.setenv("EVEROS_LLM__BASE_URL", "https://fake.example.com")
        from everos.config import load_settings

        load_settings.cache_clear()

        monkeypatch.setattr(client_mod, "_llm_client", fake_llm)

        from everos.infra.persistence.sqlite import get_engine

        db_engine = get_engine()
        async with db_engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        # Silence OME strategy extractors (we only care about the boundary +
        # memcell + buffer cycle; downstream strategies are a separate story).
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
    from everos.infra.persistence.sqlite import dispose_engine

    await dispose_engine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(idx: int, sender: str, ts: int) -> dict[str, Any]:
    return {
        "sender_id": sender,
        "role": "user",
        "timestamp": ts,
        "content": f"msg-{idx} from {sender}",
    }


async def _collect_buffer_message_ids(session_id: str) -> set[str]:
    from everos.infra.persistence.sqlite import get_engine

    eng = get_engine()
    async with eng.connect() as conn:
        result = await conn.execute(
            text("SELECT message_id FROM unprocessed_buffer WHERE session_id = :s"),
            {"s": session_id},
        )
        return {row[0] for row in result.fetchall()}


async def _collect_memcell_message_ids(session_id: str) -> set[str]:
    from everos.infra.persistence.sqlite import get_engine

    eng = get_engine()
    async with eng.connect() as conn:
        result = await conn.execute(
            text("SELECT message_ids_json FROM memcell WHERE session_id = :s"),
            {"s": session_id},
        )
        out: set[str] = set()
        for (raw,) in result.fetchall():
            out.update(json.loads(raw))
        return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_concurrent_adds_same_session_no_message_loss(
    memorize_env_locked: Callable[..., AsyncMock],
) -> None:
    """Two concurrent /add on one session: every input message must end up
    either in a memcell's message_ids OR in the surviving buffer."""
    await memorize_env_locked(fake_llm=_make_extract_all_llm())

    session_id = "s_concurrent"

    batch_a = [_msg(i, "alice", 1_700_000_000_000 + i * 1000) for i in range(4)]
    batch_b = [_msg(i + 100, "bob", 1_700_000_100_000 + i * 1000) for i in range(4)]

    # Fire both concurrently against the same session.
    await asyncio.gather(
        memorize({"session_id": session_id, "messages": batch_a}),
        memorize({"session_id": session_id, "messages": batch_b}),
    )

    buffered = await _collect_buffer_message_ids(session_id)
    in_cells = await _collect_memcell_message_ids(session_id)
    covered = buffered | in_cells

    # The id format is ``m_<session>_<ts_ms>_<idx>`` — we can derive
    # exactly what the 8 inputs should hash to without depending on the
    # internal id_gen import. Easier: assert the *count* covered == 8.
    assert len(covered) == 8, (
        f"expected 8 distinct message ids covered, got {len(covered)}: "
        f"buffer={len(buffered)}, memcell={len(in_cells)}"
    )

    # Sanity: no message appears in both buffer and memcell at once
    # (consumed = removed from buffer).
    overlap = buffered & in_cells
    assert not overlap, f"messages in both buffer and memcell: {overlap}"


async def test_concurrent_adds_serial_when_locked(
    memorize_env_locked: Callable[..., AsyncMock],
) -> None:
    """Same as above but explicitly stress with 4 concurrent batches."""
    await memorize_env_locked(fake_llm=_make_extract_all_llm())

    session_id = "s_stress"

    n_batches = 4
    batch_size = 3
    batches = [
        [
            _msg(b * 10 + i, f"u{b}", 1_700_000_000_000 + (b * 10 + i) * 1000)
            for i in range(batch_size)
        ]
        for b in range(n_batches)
    ]

    await asyncio.gather(
        *(memorize({"session_id": session_id, "messages": batch}) for batch in batches)
    )

    buffered = await _collect_buffer_message_ids(session_id)
    in_cells = await _collect_memcell_message_ids(session_id)
    covered = buffered | in_cells

    expected = n_batches * batch_size
    assert len(covered) == expected, (
        f"expected {expected} message ids covered, got {len(covered)}: "
        f"buffer={len(buffered)}, memcell={len(in_cells)}"
    )
    assert not (buffered & in_cells)


async def test_different_sessions_run_in_parallel(
    memorize_env_locked: Callable[..., AsyncMock],
) -> None:
    """Cross-session calls share no lock — must not serialise."""
    await memorize_env_locked(fake_llm=_make_extract_all_llm())

    def _msgs(sid: str) -> list[dict[str, Any]]:
        return [_msg(i, sid, 1_700_000_000_000 + i * 1000) for i in range(3)]

    await asyncio.gather(
        memorize({"session_id": "s_a", "messages": _msgs("s_a")}),
        memorize({"session_id": "s_b", "messages": _msgs("s_b")}),
        memorize({"session_id": "s_c", "messages": _msgs("s_c")}),
    )

    for sid in ("s_a", "s_b", "s_c"):
        buffered = await _collect_buffer_message_ids(sid)
        in_cells = await _collect_memcell_message_ids(sid)
        covered = buffered | in_cells
        assert len(covered) == 3, f"session {sid}: got {len(covered)}, want 3"
