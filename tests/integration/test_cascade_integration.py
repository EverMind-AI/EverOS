"""End-to-end cascade flow.

Drives the full pipeline once with real components except the embedder
(stubbed so the test never hits an external API):

    EpisodeWriter.append_entry   ─▶  md file on disk
    watchdog FSEvents thread     ─▶  CascadeWatcher._enqueue_async
    md_change_state.upsert        ─▶  pending row
    CascadeWorker.drain_once     ─▶  EpisodeHandler.handle_added_or_modified
    episode_repo.upsert          ─▶  LanceDB row

Asserts the row landed with the right shape (md_path, content_sha256,
episode tokens, vector dim). Validates that the three loops actually
talk to each other — no unit test covers the cross-loop wiring.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlmodel import SQLModel

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import build_tokenizer
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb import (
    dispose_connection,
    ensure_business_indexes,
    episode_repo,
)
from everos.infra.persistence.markdown import EpisodeWriter
from everos.infra.persistence.sqlite import (
    dispose_engine,
    get_engine,
    md_change_state_repo,
)
from everos.memory.cascade import CascadeConfig, CascadeOrchestrator


class _StubEmbedder(EmbeddingProvider):
    """1024-dim deterministic vector; counts calls for the assertion."""

    dim = 1024

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [0.0] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [await self.embed(t) for t in texts]


@pytest.fixture
async def cascade_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[MemoryRoot]:
    """Boot sqlite + lancedb against a tmp memory_root; dispose at teardown.

    Cascade uses module-level singletons; we reset them up-front to
    guarantee no state leaks in from neighbouring tests, then dispose
    on the way out so the next test sees a clean slate.
    """
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    # Embedding settings are required for the lifespan factory; the
    # stub bypasses real network, but the orchestrator still expects
    # the env to be valid-looking.
    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "stub-model")
    monkeypatch.setenv("EVEROS_EMBEDDING__BASE_URL", "http://stub.invalid/v1")
    monkeypatch.setenv("EVEROS_EMBEDDING__API_KEY", "stub-key")

    await dispose_connection()
    await dispose_engine()

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    await ensure_business_indexes()

    yield MemoryRoot.default()

    await dispose_connection()
    await dispose_engine()


async def _poll(condition, *, deadline_seconds: float = 10.0, interval: float = 0.05):  # type: ignore[no-untyped-def]
    """Poll ``condition()`` (async) until truthy, or :class:`TimeoutError`.

    Wraps the loop in :func:`asyncio.timeout` so the test surfaces a
    clean ``TimeoutError`` instead of silently spinning. The polling
    interval is a low-cost sleep; the deadline is the hard cap.
    """
    async with asyncio.timeout(deadline_seconds):
        while True:
            result = await condition()
            if result:
                return result
            await asyncio.sleep(interval)


async def test_append_to_md_propagates_to_lancedb(
    cascade_runtime: MemoryRoot,
) -> None:
    """Happy path: writer append → watcher → state row → worker → LanceDB."""
    memory_root = cascade_runtime
    embedder = _StubEmbedder()
    orchestrator = CascadeOrchestrator(
        memory_root=memory_root,
        embedder=embedder,
        tokenizer=build_tokenizer(),
        # Tight worker poll so the test wraps in seconds, not minutes.
        # Scanner interval kept long so the watcher path is the one
        # actually exercised (the scanner would mask a watcher bug).
        config=CascadeConfig(
            scan_interval_seconds=60.0,
            worker_batch_size=10,
            worker_max_retry=1,
            worker_poll_interval_seconds=0.05,
            worker_retry_backoff_seconds=0.0,
        ),
    )
    await orchestrator.start()
    # Give the watchdog Observer thread a beat to actually subscribe;
    # this is the watchdog API gap (start() returns before the kqueue
    # / FSEvents subscription is live on macOS).
    await asyncio.sleep(0.3)

    try:
        writer = EpisodeWriter(memory_root)
        today = _dt.date(2026, 5, 14)
        eid = await writer.append_entry(
            "u_integration",
            inline={
                "owner_id": "u_integration",
                "session_id": "s_int",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_id": "mc_integration_parent",
                "sender_ids": ["u_integration"],
            },
            sections={
                "Subject": "Test",
                "Summary": "Stub",
                "Content": "the user mentioned dark mode preference",
            },
            date=today,
        )
        md_path = (
            "default_app/default_project/users/u_integration/episodes/"
            "episode-2026-05-14.md"
        )

        # 1. Watcher enqueues the path.
        async def _state_appeared():  # type: ignore[no-untyped-def]
            return await md_change_state_repo.get_by_id(md_path)

        row = await _poll(_state_appeared, deadline_seconds=5.0)
        assert row.kind == "episode"

        # 2. Worker drives it to done.
        async def _state_done():  # type: ignore[no-untyped-def]
            r = await md_change_state_repo.get_by_id(md_path)
            return r if (r is not None and r.status == "done") else None

        done_row = await _poll(_state_done, deadline_seconds=10.0)
        assert done_row.error is None

        # 3. LanceDB carries the typed episode row.
        episode_id = f"u_integration_{eid.format()}"
        ep_row = await episode_repo.get_by_id(episode_id)
        assert ep_row is not None
        assert ep_row.episode == "the user mentioned dark mode preference"
        assert ep_row.episode_tokens  # tokenizer ran
        assert ep_row.md_path == md_path
        assert ep_row.parent_id == "mc_integration_parent"
        assert ep_row.content_sha256
        assert len(ep_row.vector) == 1024
        assert embedder.calls >= 1
    finally:
        await orchestrator.stop()


async def test_delete_md_wipes_lancedb_row(
    cascade_runtime: MemoryRoot,
) -> None:
    """Append + drain, then ``unlink`` the md and watch the row evaporate."""
    memory_root = cascade_runtime
    orchestrator = CascadeOrchestrator(
        memory_root=memory_root,
        embedder=_StubEmbedder(),
        tokenizer=build_tokenizer(),
        config=CascadeConfig(
            scan_interval_seconds=60.0,
            worker_batch_size=10,
            worker_max_retry=1,
            worker_poll_interval_seconds=0.05,
            worker_retry_backoff_seconds=0.0,
        ),
    )
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        writer = EpisodeWriter(memory_root)
        today = _dt.date(2026, 5, 14)
        eid = await writer.append_entry(
            "u_del",
            inline={
                "owner_id": "u_del",
                "session_id": "s",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_id": "mc_del_parent",
                "sender_ids": ["u_del"],
            },
            sections={"Content": "to be removed"},
            date=today,
        )
        md_path = (
            "default_app/default_project/users/u_del/episodes/episode-2026-05-14.md"
        )
        absolute = memory_root.root / md_path

        async def _ep_present():  # type: ignore[no-untyped-def]
            return await episode_repo.get_by_id(f"u_del_{eid.format()}")

        await _poll(_ep_present, deadline_seconds=10.0)

        # Now remove the file; the watcher's on_deleted should fire.
        absolute.unlink()

        async def _ep_gone():  # type: ignore[no-untyped-def]
            row = await episode_repo.get_by_id(f"u_del_{eid.format()}")
            return row is None

        assert await _poll(_ep_gone, deadline_seconds=10.0)
    finally:
        await orchestrator.stop()
