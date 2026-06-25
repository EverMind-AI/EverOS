"""``CascadeOrchestrator`` — idempotent start/stop, queue_summary forwards."""

from __future__ import annotations

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
)
from everos.infra.persistence.sqlite import dispose_engine, get_engine
from everos.memory.cascade import CascadeConfig, CascadeOrchestrator


class _StubEmbedder(EmbeddingProvider):
    dim = 1024

    async def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [[0.0] * self.dim for _ in texts]


@pytest.fixture
async def runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[MemoryRoot]:
    """Boot sqlite + lancedb against a tmp memory_root."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
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


def _make_orchestrator(memory_root: MemoryRoot) -> CascadeOrchestrator:
    return CascadeOrchestrator(
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


async def test_double_start_is_idempotent(runtime: MemoryRoot) -> None:
    """Calling start twice does not relaunch tasks."""
    orch = _make_orchestrator(runtime)
    await orch.start()
    # Capture watcher identity to verify the second start doesn't replace it.
    first_watcher = orch._watcher
    await orch.start()
    assert orch._watcher is first_watcher
    await orch.stop()


async def test_stop_before_start_is_noop(runtime: MemoryRoot) -> None:
    orch = _make_orchestrator(runtime)
    await orch.stop()  # must not raise; nothing to do


async def test_double_stop_is_idempotent(runtime: MemoryRoot) -> None:
    orch = _make_orchestrator(runtime)
    await orch.start()
    await orch.stop()
    await orch.stop()  # second stop is a no-op


async def test_queue_summary_returns_empty_on_fresh_runtime(
    runtime: MemoryRoot,
) -> None:
    orch = _make_orchestrator(runtime)
    summary = await orch.queue_summary()
    assert summary.pending == 0
    assert summary.done == 0
    assert summary.failed_retryable == 0
    assert summary.failed_permanent == 0


async def test_drain_once_returns_zero_on_empty_queue(
    runtime: MemoryRoot,
) -> None:
    orch = _make_orchestrator(runtime)
    assert await orch.drain_once() == 0
