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
