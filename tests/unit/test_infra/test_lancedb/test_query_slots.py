"""One process runs at most ``[lancedb] max_concurrent_queries`` search queries.

Uncapped, a burst of searches (one agentic search fans out to about 28
queries) opened every data file of a table once per query -- 981 open
descriptors against a 1024 soft limit in a 2026-11 soak -- and turned all
their results into Python objects on the event loop at the same time.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path

from everos.config import LanceDBSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.lancedb._repos.atomic_fact import _AtomicFactRepo
from everos.infra.persistence.lancedb.lancedb_manager import LanceDbConnection
from everos.infra.persistence.records import AtomicFact

_DIM = 8


def _connection(tmp_path: Path, cap: int) -> LanceDbConnection:
    root = MemoryRoot(tmp_path)
    root.ensure()
    return LanceDbConnection(root, LanceDBSettings(max_concurrent_queries=cap))


async def test_no_more_queries_run_at_once_than_the_cap(tmp_path: Path) -> None:
    conn = _connection(tmp_path, cap=2)
    in_flight = peak = 0

    async def query() -> None:
        nonlocal in_flight, peak
        async with conn.query_slot():
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1

    await asyncio.gather(*(query() for _ in range(6)))

    assert peak == 2


async def test_a_search_waits_for_a_free_slot(tmp_path: Path) -> None:
    conn = _connection(tmp_path, cap=1)
    repo = _AtomicFactRepo(source=conn, dim=_DIM)
    await repo.upsert(
        [
            AtomicFact(
                id="u1_af_0",
                entry_id="af_0",
                owner_id="u1",
                owner_type="user",
                timestamp=dt.datetime(2026, 11, 5, tzinfo=dt.UTC),
                parent_id="ep_0",
                sender_ids=["u1"],
                fact="likes hiking",
                fact_tokens="likes hiking",
                md_path="users/u1/.atomic_facts/af.md",
                content_sha256="sha0",
                vector=[0.1] * _DIM,
            )
        ]
    )

    async with conn.query_slot():
        search = asyncio.create_task(repo.dense_search([0.1] * _DIM, None, limit=1))
        await asyncio.sleep(0.2)
        assert not search.done()

    assert [row["id"] for row in await search] == ["u1_af_0"]


# ── open-file headroom ────────────────────────────────────────────────────
#
# A slot alone did not bound descriptors: one search opens every data file of
# its table, so 32 of them on a fragmented table spiked to 6,000 descriptors
# and once hit the 8192 soft limit (2026-11 8-hour soak). A search holding a
# slot now waits while the process is at FD_HIGH_WATER of its soft limit.


def _fake_usage(monkeypatch, readings: list[tuple[int, int]]) -> list[int]:
    """Serve ``readings`` in turn (the last one repeats); count the reads."""
    from everos.infra.persistence.lancedb import lancedb_manager

    reads: list[int] = []

    def usage() -> tuple[int, int]:
        reads.append(1)
        return readings[min(len(reads), len(readings)) - 1]

    monkeypatch.setattr(lancedb_manager, "open_file_usage", usage)
    monkeypatch.setattr(lancedb_manager, "_fd_reading", (float("-inf"), None))
    return reads


async def test_a_query_runs_at_once_under_the_high_water(
    tmp_path: Path, monkeypatch
) -> None:
    _fake_usage(monkeypatch, [(100, 1000)])
    conn = _connection(tmp_path, cap=2)

    async with asyncio.timeout(0.1), conn.query_slot():
        pass


async def test_a_query_waits_for_open_file_headroom(
    tmp_path: Path, monkeypatch
) -> None:
    from everos.infra.persistence.lancedb import lancedb_manager

    limit = 1000
    high = int(limit * lancedb_manager.FD_HIGH_WATER)
    reads = _fake_usage(monkeypatch, [(high, limit)] * 5 + [(high - 1, limit)])
    conn = _connection(tmp_path, cap=2)
    ran = asyncio.Event()

    async def query() -> None:
        async with conn.query_slot():
            ran.set()

    task = asyncio.create_task(query())
    await asyncio.sleep(0.05)
    assert not ran.is_set()  # at the high water: waiting

    await asyncio.wait_for(task, timeout=1)
    assert ran.is_set() and len(reads) == 6


async def test_a_query_stops_waiting_after_the_cap_and_logs(
    tmp_path: Path, monkeypatch
) -> None:
    from everos.infra.persistence.lancedb import lancedb_manager

    _fake_usage(monkeypatch, [(999, 1000)])
    monkeypatch.setattr(lancedb_manager, "_FD_WAIT_MAX_S", 0.1)
    warnings: list[str] = []
    monkeypatch.setattr(
        lancedb_manager.logger, "warning", lambda event, **_: warnings.append(event)
    )
    conn = _connection(tmp_path, cap=1)

    async with asyncio.timeout(1), conn.query_slot():
        pass

    assert warnings == ["lancedb_fd_headroom_wait_expired"]
