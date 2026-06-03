"""Repro: high-frequency atomic-replace bursts vs. cascade drain.

Drives N successive ``AtomicFactWriter.append_entries`` calls against the
same daily-log md, simulating multiple OME memcells landing in the same
owner+day bucket within a few ms of each other.

Before the watcher.on_deleted stat-guard, macOS FSEvents emits a paired
(moved, deleted) per ``os.replace`` and the synthetic deletion can
become the final ``change_type`` of the row — driving the worker into
``handle_deleted`` and wiping LanceDB while md is intact. Repeat the
test ~20x to surface the race if it ever resurfaces.

Scanner interval is held at 60s so the watcher path is the only thing
exercised (a scanner sweep would mask a watcher bug).
"""

from __future__ import annotations

import asyncio
import datetime as _dt
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest
from sqlmodel import SQLModel

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import build_tokenizer
from everos.core.persistence import MarkdownReader, MemoryRoot
from everos.infra.persistence.lancedb import (
    dispose_connection,
    ensure_business_indexes,
)
from everos.infra.persistence.lancedb.lancedb_manager import get_table
from everos.infra.persistence.lancedb.tables.atomic_fact import AtomicFact
from everos.infra.persistence.markdown import AtomicFactWriter
from everos.infra.persistence.sqlite import (
    dispose_engine,
    get_engine,
    md_change_state_repo,
)
from everos.memory.cascade import CascadeConfig, CascadeOrchestrator


@pytest.fixture(autouse=True)
def _reset_lancedb_write_locks() -> None:
    """Drop the per-table write-lock pool between tests; mirrors the
    unit-test fixture in test_repository.py. Without this, the second
    test in this module hits "Lock bound to a different event loop"
    because LanceRepoBase stashes locks in a ClassVar dict."""
    from everos.core.persistence.lancedb.repository import LanceRepoBase

    LanceRepoBase._reset_locks_for_tests()


class _StubEmbedder(EmbeddingProvider):
    dim = 1024

    async def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [[0.0] * self.dim for _ in texts]


@pytest.fixture
async def cascade_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[MemoryRoot]:
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
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


async def _wait_drain(deadline: float = 15.0) -> None:
    async with asyncio.timeout(deadline):
        while True:
            summary = await md_change_state_repo.queue_summary()
            if summary.pending == 0:
                return
            await asyncio.sleep(0.05)


async def _count_lance_rows(md_path: str) -> int:
    table = await get_table(AtomicFact.TABLE_NAME, AtomicFact)
    return await table.count_rows(filter=f"md_path = '{md_path}'")


async def _count_md_entries(absolute: Path) -> int:
    if not await anyio.Path(absolute).is_file():
        return 0
    parsed = await MarkdownReader.read(absolute)
    return len(parsed.entries)


@pytest.mark.parametrize(
    "n_calls,items_per_call,inter_call_sleep_ms",
    [
        (20, 1, 0.0),
        (20, 1, 1.0),
        (20, 3, 0.0),
        (10, 3, 5.0),
    ],
)
async def test_high_freq_atomic_fact_append_no_loss(
    cascade_runtime: MemoryRoot,
    n_calls: int,
    items_per_call: int,
    inter_call_sleep_ms: float,
) -> None:
    memory_root = cascade_runtime
    orchestrator = CascadeOrchestrator(
        memory_root=memory_root,
        embedder=_StubEmbedder(),
        tokenizer=build_tokenizer(),
        config=CascadeConfig(
            scan_interval_seconds=60.0,
            worker_batch_size=20,
            worker_max_retry=1,
            worker_poll_interval_seconds=0.05,
            worker_retry_backoff_seconds=0.0,
        ),
    )
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        writer = AtomicFactWriter(root=memory_root)
        bucket = _dt.date(2026, 5, 19)
        owner_id = "bob"
        total = 0
        for i in range(n_calls):
            items = [
                (
                    {
                        "owner_id": owner_id,
                        "session_id": f"s_{i}_{j}",
                        "timestamp": "2026-05-19T07:04:26+00:00",
                        "parent_id": f"mc_{i}",
                        "sender_ids": [owner_id],
                    },
                    {"Fact": f"fact body call={i} item={j}"},
                )
                for j in range(items_per_call)
            ]
            await writer.append_entries(owner_id, items, date=bucket)
            total += items_per_call
            if inter_call_sleep_ms > 0:
                await asyncio.sleep(inter_call_sleep_ms / 1000.0)

        await _wait_drain(deadline=15.0)
        # FSEvents has ~30-100ms kernel-to-userspace delivery latency,
        # so the watcher's `on_*` callbacks for the LAST few
        # os.replace() bursts may arrive AFTER sqlite first reads
        # `pending == 0`. Absorb that tail: settle 500ms, then drain
        # again until truly quiescent.
        await asyncio.sleep(0.5)
        await _wait_drain(deadline=15.0)

        md_path = (
            f"default_app/default_project/users/{owner_id}/.atomic_facts/"
            f"atomic_fact-{bucket.isoformat()}.md"
        )
        absolute = memory_root.root / md_path
        md_entries = await _count_md_entries(absolute)
        lance_rows = await _count_lance_rows(md_path)
        state_row = await md_change_state_repo.get_by_id(md_path)

        assert md_entries == total, (
            f"writer self-check failed: total={total} md={md_entries}"
        )
        assert lance_rows == md_entries, (
            f"CASCADE LOSS: md={md_entries} lance={lance_rows} "
            f"state={state_row.status if state_row else 'NONE'} "
            f"lsn={state_row.lsn if state_row else None}"
        )
    finally:
        await orchestrator.stop()
