"""Strict md <-> lancedb consistency across all 4 daily-log kinds.

For each registered daily-log kind, seed N entries via the kind's
writer, wait for the cascade to drain, then assert exact equality
between md state and LanceDB state:

* ``frontmatter.entry_count == N``
* number of ``<!-- entry:... -->`` blocks == N
* ``lance_repo.count_rows(md_path=...) == N``
* lance ``entry_id`` set == md ``entry_id`` set

This is the strict counterpart to the loose ``>=`` assertions in
:mod:`test_add_flush_user_pipeline_e2e` (which can't be exact because
LLM output is non-deterministic).

Skill / profile are single-file (not daily-log) kinds and are covered
by the e2e pipeline tests where the OME drives real LLM emissions.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as _dt
from collections.abc import AsyncIterator, Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from sqlmodel import SQLModel

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import build_tokenizer
from everos.core.persistence import MarkdownReader, MemoryRoot
from everos.infra.persistence.lancedb import (
    agent_case_repo,
    atomic_fact_repo,
    dispose_connection,
    ensure_business_indexes,
    episode_repo,
    foresight_repo,
)
from everos.infra.persistence.lancedb.lancedb_manager import get_table
from everos.infra.persistence.lancedb.tables.agent_case import AgentCase
from everos.infra.persistence.lancedb.tables.atomic_fact import AtomicFact
from everos.infra.persistence.lancedb.tables.episode import Episode
from everos.infra.persistence.lancedb.tables.foresight import Foresight
from everos.infra.persistence.markdown import (
    AgentCaseWriter,
    AtomicFactWriter,
    EpisodeWriter,
    ForesightWriter,
)
from everos.infra.persistence.sqlite import (
    dispose_engine,
    get_engine,
    md_change_state_repo,
)
from everos.memory.cascade import CascadeConfig, CascadeOrchestrator
from everos.memory.cascade.registry import KIND_REGISTRY
from tests._consistency_assertions import _daily_log_sha_for_entry


@pytest.fixture(autouse=True)
def _reset_lancedb_write_locks() -> None:
    """ClassVar lock pool reset; see test_repository.py for rationale."""
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


@dataclasses.dataclass(frozen=True)
class _DailyLogKindCase:
    """A single registered daily-log kind, packaged for parametrization."""

    name: str
    scope: str  # "users" | "agents"
    dir_name: str
    file_prefix: str
    writer_factory: Callable[[MemoryRoot], Any]
    repo: Any
    table_cls: type
    build_item: Callable[[str, int], tuple[Mapping[str, object], Mapping[str, str]]]


def _af_item(scope_id: str, j: int):
    return (
        {
            "owner_id": scope_id,
            "session_id": f"s_{j}",
            "timestamp": "2026-05-19T07:04:26+00:00",
            "parent_id": f"mc_{j}",
            "sender_ids": [scope_id],
        },
        {"Fact": f"af fact body {j}"},
    )


def _ep_item(scope_id: str, j: int):
    return (
        {
            "owner_id": scope_id,
            "session_id": f"s_{j}",
            "timestamp": "2026-05-19T07:04:26+00:00",
            "parent_id": f"mc_{j}",
            "sender_ids": [scope_id],
        },
        {"Subject": f"subj {j}", "Summary": f"sum {j}", "Content": f"content {j}"},
    )


def _fs_item(scope_id: str, j: int):
    return (
        {
            "owner_id": scope_id,
            "session_id": f"s_{j}",
            "timestamp": "2026-05-19T07:04:26+00:00",
            "parent_id": f"mc_{j}",
            "sender_ids": [scope_id],
        },
        {"Foresight": f"foresight body {j}"},
    )


def _ac_item(scope_id: str, j: int):
    return (
        {
            "owner_id": scope_id,
            "session_id": f"s_{j}",
            "timestamp": "2026-05-19T07:04:26+00:00",
            "parent_id": f"mc_{j}",
            "quality_score": 0.9,
        },
        {
            "TaskIntent": f"task intent {j}",
            "Approach": f"approach {j}",
            "KeyInsight": f"insight {j}",
        },
    )


_KIND_CASES: list[_DailyLogKindCase] = [
    _DailyLogKindCase(
        name="atomic_fact",
        scope="users",
        dir_name=".atomic_facts",
        file_prefix="atomic_fact",
        writer_factory=AtomicFactWriter,
        repo=atomic_fact_repo,
        table_cls=AtomicFact,
        build_item=_af_item,
    ),
    _DailyLogKindCase(
        name="episode",
        scope="users",
        dir_name="episodes",
        file_prefix="episode",
        writer_factory=EpisodeWriter,
        repo=episode_repo,
        table_cls=Episode,
        build_item=_ep_item,
    ),
    _DailyLogKindCase(
        name="foresight",
        scope="users",
        dir_name=".foresights",
        file_prefix="foresight",
        writer_factory=ForesightWriter,
        repo=foresight_repo,
        table_cls=Foresight,
        build_item=_fs_item,
    ),
    _DailyLogKindCase(
        name="agent_case",
        scope="agents",
        dir_name=".cases",
        file_prefix="agent_case",
        writer_factory=AgentCaseWriter,
        repo=agent_case_repo,
        table_cls=AgentCase,
        build_item=_ac_item,
    ),
]


async def _wait_path_done(md_path: str, *, deadline: float = 15.0) -> None:
    async with asyncio.timeout(deadline):
        while True:  # noqa: ASYNC110 - polling cascade state
            row = await md_change_state_repo.get_by_id(md_path)
            if row is not None:
                break
            await asyncio.sleep(0.05)
        while True:  # noqa: ASYNC110 - polling cascade state
            row = await md_change_state_repo.get_by_id(md_path)
            if row is not None and row.status in ("done", "failed"):
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.1)


@pytest.mark.parametrize("case", _KIND_CASES, ids=lambda c: c.name)
async def test_md_lance_strict_consistency_per_kind(
    cascade_runtime: MemoryRoot,
    case: _DailyLogKindCase,
) -> None:
    """Per-kind strict equality: md entries / frontmatter / lance rows all == N."""
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
        writer = case.writer_factory(root=memory_root)
        scope_id = f"sid_{case.name}"
        bucket = _dt.date(2026, 5, 19)
        n = 5
        items = [case.build_item(scope_id, j) for j in range(n)]
        eids = await writer.append_entries(scope_id, items, date=bucket)
        assert len(eids) == n, f"writer returned {len(eids)} eids, expected {n}"

        md_path = (
            f"default_app/default_project/{case.scope}/{scope_id}/{case.dir_name}/"
            f"{case.file_prefix}-{bucket.isoformat()}.md"
        )
        absolute = memory_root.root / md_path
        await _wait_path_done(md_path)

        # 1) frontmatter.entry_count == N
        parsed = await MarkdownReader.read(absolute)
        assert parsed.frontmatter.get("entry_count") == n, (
            f"{case.name}: frontmatter.entry_count="
            f"{parsed.frontmatter.get('entry_count')}, expected {n}"
        )

        # 2) md entry blocks == N
        assert len(parsed.entries) == n, (
            f"{case.name}: md has {len(parsed.entries)} entry blocks, expected {n}"
        )

        # 3) lance count_rows(md_path) == N (strict equality)
        table = await get_table(case.table_cls.TABLE_NAME, case.table_cls)
        lance_count = await table.count_rows(filter=f"md_path = '{md_path}'")
        assert lance_count == n, (
            f"{case.name}: md={n} lance={lance_count} for {md_path}"
        )

        # 4) lance entry_id set == md entry_id set
        lance_rows = await case.repo.find_where(f"md_path = '{md_path}'", limit=100)
        lance_eids = {r.entry_id for r in lance_rows}
        md_eids = {e.id for e in parsed.entries}
        assert lance_eids == md_eids, (
            f"{case.name}: lance eids {lance_eids} != md eids {md_eids}"
        )

        # 4b) lance content_sha256 per entry == md-recomputed content_sha256
        # Catches "id present but content mismatched" — orthogonal to (4).
        handler_cls = next(
            spec.handler_factory for spec in KIND_REGISTRY if spec.name == case.name
        )
        md_sha_by_id = {
            e.id: _daily_log_sha_for_entry(handler_cls, e.as_structured())
            for e in parsed.entries
        }
        lance_sha_by_id = {r.entry_id: r.content_sha256 for r in lance_rows}
        assert md_sha_by_id == lance_sha_by_id, (
            f"{case.name}: per-entry content_sha256 mismatch "
            f"@ {md_path}: md={md_sha_by_id} lance={lance_sha_by_id}"
        )

        # 5) row state row is terminally done (not failed)
        state_row = await md_change_state_repo.get_by_id(md_path)
        assert state_row is not None and state_row.status == "done", (
            f"{case.name}: state row status={state_row.status if state_row else 'NONE'}"
        )
    finally:
        await orchestrator.stop()
