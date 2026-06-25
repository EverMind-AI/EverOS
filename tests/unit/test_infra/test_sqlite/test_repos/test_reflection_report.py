"""Tests for :class:`_ReflectionReportRepo` — reflection audit persistence.

Verifies create, latest-for-cluster lookup, and reflected-cluster-id
listing including status filtering (rolled_back rows excluded).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite.repos.reflection_report import (
    _ReflectionReportRepo,
)
from everos.infra.persistence.sqlite.tables import ReflectionReport


@pytest.fixture
async def repo(tmp_path: Path) -> _ReflectionReportRepo:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    engine = create_system_engine(mr.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return _ReflectionReportRepo(session_factory=factory)


def _make_report(
    *,
    report_id: str = "rr_001",
    cluster_id: str = "cl_aaa000000001",
    owner_id: str = "u_alice",
    mode: str = "consolidation",
    source_members: str = "mc_one,mc_two",
    source_count: int = 2,
    merged_entry_id: str = "ep_merged_001",
    deprecated_fact_count: int = 1,
    status: str = "completed",
) -> ReflectionReport:
    return ReflectionReport(
        id=report_id,
        cluster_id=cluster_id,
        owner_id=owner_id,
        mode=mode,
        source_members=source_members,
        source_count=source_count,
        merged_entry_id=merged_entry_id,
        deprecated_fact_count=deprecated_fact_count,
        status=status,
    )


async def test_create_and_get_latest(repo: _ReflectionReportRepo) -> None:
    """Create a report then retrieve it as the latest for that cluster."""
    report = _make_report()
    await repo.create(report)

    latest = await repo.get_latest_for_cluster("cl_aaa000000001")
    assert latest is not None
    assert latest.id == "rr_001"
    assert latest.cluster_id == "cl_aaa000000001"
    assert latest.owner_id == "u_alice"
    assert latest.mode == "consolidation"
    assert latest.source_count == 2
    assert latest.merged_entry_id == "ep_merged_001"
    assert latest.deprecated_fact_count == 1
    assert latest.status == "completed"


async def test_get_latest_returns_none_when_empty(
    repo: _ReflectionReportRepo,
) -> None:
    """No reports exist for a cluster -> None."""
    result = await repo.get_latest_for_cluster("cl_nonexistent")
    assert result is None


async def test_list_reflected_cluster_ids(
    repo: _ReflectionReportRepo,
) -> None:
    """Reports for two distinct clusters -> both cluster ids returned."""
    await repo.create(_make_report(report_id="rr_001", cluster_id="cl_aaa000000001"))
    await repo.create(_make_report(report_id="rr_002", cluster_id="cl_bbb000000002"))

    ids = await repo.list_reflected_cluster_ids("u_alice")
    assert ids == {"cl_aaa000000001", "cl_bbb000000002"}


async def test_list_reflected_excludes_rolled_back(
    repo: _ReflectionReportRepo,
) -> None:
    """A rolled_back report should not appear in the reflected set."""
    await repo.create(
        _make_report(
            report_id="rr_001",
            cluster_id="cl_aaa000000001",
            status="completed",
        )
    )
    await repo.create(
        _make_report(
            report_id="rr_002",
            cluster_id="cl_bbb000000002",
            status="rolled_back",
        )
    )

    ids = await repo.list_reflected_cluster_ids("u_alice")
    assert ids == {"cl_aaa000000001"}
    assert "cl_bbb000000002" not in ids
