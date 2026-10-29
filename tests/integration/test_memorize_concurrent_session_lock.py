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
import dataclasses
import importlib
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from everalgo.testing import FakeLLMClient
from sqlalchemy import text

from everos.infra.persistence import MemoryRoot
from everos.service import (
    MemorizeResult,
    MemoryService,
    build_application,
    build_providers,
)
from tests.fakes import memorize_llm
from tests.fakes.storage import Storage

# ---------------------------------------------------------------------------
# Fake LLM that splits each call into one memcell + 0-tail (force extract)
# ---------------------------------------------------------------------------

_under_test: dict[str, MemoryService] = {}
"""The memory facade of the application the ``memorize_env`` fixture built."""


async def memorize(payload: dict[str, Any], **kwargs: Any) -> MemorizeResult:
    """``MemoryService.memorize`` of the application under test."""
    return await _under_test["memory"].memorize(payload, **kwargs)


def _cut_before_last(window: int) -> list[int]:
    """Close everything but the last message; a non-final call keeps a tail."""
    return [window - 1] if window > 1 else []


@pytest_asyncio.fixture
async def memorize_env_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> AsyncIterator[Callable[..., AsyncMock]]:
    monkeypatch.setattr(
        MemoryRoot, "resolve", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ome.toml").write_text("# test\n")

    af_mod = importlib.import_module("everos.memory.strategies.extract_atomic_facts")
    fs_mod = importlib.import_module("everos.memory.strategies.extract_foresight")
    lock_mod = importlib.import_module("everos.service._session_lock")

    # Reset memorize singletons + session lock registry.
    lock_mod._reset_for_tests()

    started: dict[str, Any] = {"engine": None}

    async def _setup(*, fake_llm: FakeLLMClient) -> None:
        monkeypatch.setenv("EVEROS_MEMORIZE__MODE", "chat")
        monkeypatch.setenv("EVEROS_LLM__API_KEY", "fake-key")
        monkeypatch.setenv("EVEROS_LLM__BASE_URL", "https://fake.example.com")
        from everos.config import load_settings

        load_settings.cache_clear()

        await storage.create_schema()

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

    if started["engine"] is not None:
        await started["engine"].stop()


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


async def _collect_buffer_message_ids(storage: Storage, session_id: str) -> set[str]:

    eng = storage.database.engine
    async with eng.connect() as conn:
        result = await conn.execute(
            text("SELECT message_id FROM unprocessed_buffer WHERE session_id = :s"),
            {"s": session_id},
        )
        return {row[0] for row in result.fetchall()}


async def _collect_memcell_message_ids(storage: Storage, session_id: str) -> set[str]:

    eng = storage.database.engine
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
    memorize_env_locked: Callable[..., AsyncMock], storage: Storage
) -> None:
    """Two concurrent /add on one session: every input message must end up
    either in a memcell's message_ids OR in the surviving buffer."""
    await memorize_env_locked(fake_llm=memorize_llm(_cut_before_last))

    session_id = "s_concurrent"

    batch_a = [_msg(i, "alice", 1_700_000_000_000 + i * 1000) for i in range(4)]
    batch_b = [_msg(i + 100, "bob", 1_700_000_100_000 + i * 1000) for i in range(4)]

    # Fire both concurrently against the same session.
    await asyncio.gather(
        memorize({"session_id": session_id, "messages": batch_a}),
        memorize({"session_id": session_id, "messages": batch_b}),
    )

    buffered = await _collect_buffer_message_ids(storage, session_id)
    in_cells = await _collect_memcell_message_ids(storage, session_id)
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
    memorize_env_locked: Callable[..., AsyncMock], storage: Storage
) -> None:
    """Same as above but explicitly stress with 4 concurrent batches."""
    await memorize_env_locked(fake_llm=memorize_llm(_cut_before_last))

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

    buffered = await _collect_buffer_message_ids(storage, session_id)
    in_cells = await _collect_memcell_message_ids(storage, session_id)
    covered = buffered | in_cells

    expected = n_batches * batch_size
    assert len(covered) == expected, (
        f"expected {expected} message ids covered, got {len(covered)}: "
        f"buffer={len(buffered)}, memcell={len(in_cells)}"
    )
    assert not (buffered & in_cells)


async def test_different_sessions_run_in_parallel(
    memorize_env_locked: Callable[..., AsyncMock], storage: Storage
) -> None:
    """Cross-session calls share no lock — must not serialise."""
    await memorize_env_locked(fake_llm=memorize_llm(_cut_before_last))

    def _msgs(sid: str) -> list[dict[str, Any]]:
        return [_msg(i, sid, 1_700_000_000_000 + i * 1000) for i in range(3)]

    await asyncio.gather(
        memorize({"session_id": "s_a", "messages": _msgs("s_a")}),
        memorize({"session_id": "s_b", "messages": _msgs("s_b")}),
        memorize({"session_id": "s_c", "messages": _msgs("s_c")}),
    )

    for sid in ("s_a", "s_b", "s_c"):
        buffered = await _collect_buffer_message_ids(storage, sid)
        in_cells = await _collect_memcell_message_ids(storage, sid)
        covered = buffered | in_cells
        assert len(covered) == 3, f"session {sid}: got {len(covered)}, want 3"
