"""Tests for :class:`_EpisodeRepo.list_by_owner_after_ts`.

Exercises the new scalar-only query method that selects episodes by owner,
timestamp, and parent_type (for direct-path profile extraction without
vector/cluster dependencies). Confirms that rows with ``vector=None`` are
included (Tier 1-safe).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.config import LanceDBSettings
from everos.core.persistence import (
    MemoryRoot,
    open_lancedb_connection,
)
from everos.infra.persistence.lancedb.repos.episode import _EpisodeRepo
from everos.infra.persistence.lancedb.tables.episode import Episode


@pytest.fixture(autouse=True)
def _reset_write_locks() -> None:
    """Drop the per-table write-lock pool between tests."""
    from everos.core.persistence.lancedb import LanceRepoBase

    LanceRepoBase._reset_locks_for_tests()


@pytest.fixture
async def episode_repo(tmp_path: Path) -> _EpisodeRepo:
    """Open a tmp connection, create the ``episode`` table, return a repo."""
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())
    table = await conn.create_table("episode", schema=Episode)
    return _EpisodeRepo(table=table)


def _make_episode(
    *,
    owner_id: str,
    entry_id: str,
    timestamp_ms: int,
    parent_type: str = "memcell",
    app_id: str = "default",
    project_id: str = "default",
    vector: list[float] | None = None,
) -> Episode:
    """Helper to construct an Episode row for testing."""
    import datetime as dt

    # Convert milliseconds to datetime
    timestamp_us = timestamp_ms * 1000
    timestamp_dt = dt.datetime.fromtimestamp(timestamp_us / 1e6, tz=dt.UTC)

    return Episode(
        id=f"{owner_id}_{entry_id}",
        entry_id=entry_id,
        owner_id=owner_id,
        owner_type="user",
        app_id=app_id,
        project_id=project_id,
        session_id="sess_1",
        timestamp=timestamp_dt,
        parent_type=parent_type,
        parent_id="memcell_1",
        sender_ids=["user", "assistant"],
        episode="test episode",
        episode_tokens="test episode",
        md_path=f"users/{owner_id}/episodes/{entry_id}.md",
        content_sha256="abc123",
        vector=vector,
    )


async def test_returns_only_after_ts(episode_repo: _EpisodeRepo) -> None:
    """Returned rows all have timestamp > after_ts."""
    rows = [
        _make_episode(owner_id="u1", entry_id="ep_1", timestamp_ms=500),
        _make_episode(owner_id="u1", entry_id="ep_2", timestamp_ms=1500),
        _make_episode(owner_id="u1", entry_id="ep_3", timestamp_ms=2500),
    ]
    await episode_repo.add(rows)

    result = await episode_repo.list_by_owner_after_ts(
        owner_id="u1",
        after_ts=1000,
        parent_type="memcell",
        app_id="default",
        project_id="default",
    )

    assert len(result) == 2
    entry_ids = {ep.entry_id for ep in result}
    assert entry_ids == {"ep_2", "ep_3"}


async def test_scoped_by_owner_id(episode_repo: _EpisodeRepo) -> None:
    """Only returns rows matching the specified owner_id."""
    rows = [
        _make_episode(owner_id="u1", entry_id="ep_1", timestamp_ms=500),
        _make_episode(owner_id="u1", entry_id="ep_2", timestamp_ms=1500),
        _make_episode(owner_id="u2", entry_id="ep_3", timestamp_ms=1500),
    ]
    await episode_repo.add(rows)

    result = await episode_repo.list_by_owner_after_ts(
        owner_id="u1",
        after_ts=0,
        parent_type="memcell",
        app_id="default",
        project_id="default",
    )

    assert len(result) == 2
    assert all(ep.owner_id == "u1" for ep in result)


async def test_filtered_by_parent_type(episode_repo: _EpisodeRepo) -> None:
    """Only returns rows with the specified parent_type."""
    rows = [
        _make_episode(
            owner_id="u1", entry_id="ep_1", timestamp_ms=500, parent_type="memcell"
        ),
        _make_episode(
            owner_id="u1", entry_id="ep_2", timestamp_ms=1500, parent_type="memcell"
        ),
        _make_episode(
            owner_id="u1", entry_id="ep_3", timestamp_ms=2500, parent_type="cluster"
        ),
    ]
    await episode_repo.add(rows)

    result = await episode_repo.list_by_owner_after_ts(
        owner_id="u1",
        after_ts=0,
        parent_type="memcell",
        app_id="default",
        project_id="default",
    )

    assert len(result) == 2
    assert all(ep.parent_type == "memcell" for ep in result)


async def test_includes_rows_with_null_vector(episode_repo: _EpisodeRepo) -> None:
    """Rows with vector=None are included (Tier 1-safe)."""
    rows = [
        _make_episode(owner_id="u1", entry_id="ep_1", timestamp_ms=500, vector=None),
        _make_episode(
            owner_id="u1", entry_id="ep_2", timestamp_ms=1500, vector=[0.1] * 1024
        ),
        _make_episode(owner_id="u1", entry_id="ep_3", timestamp_ms=2500, vector=None),
    ]
    await episode_repo.add(rows)

    result = await episode_repo.list_by_owner_after_ts(
        owner_id="u1",
        after_ts=400,
        parent_type="memcell",
        app_id="default",
        project_id="default",
    )

    # All three rows pass the timestamp filter; both vector-null and vector-present
    # should be included
    assert len(result) == 3
    null_vectors = [ep for ep in result if ep.vector is None]
    assert len(null_vectors) == 2


async def test_respects_scope_app_id_project_id(episode_repo: _EpisodeRepo) -> None:
    """Rows are scoped by app_id and project_id."""
    rows = [
        _make_episode(
            owner_id="u1",
            entry_id="ep_1",
            timestamp_ms=500,
            app_id="app_a",
            project_id="proj_1",
        ),
        _make_episode(
            owner_id="u1",
            entry_id="ep_2",
            timestamp_ms=1500,
            app_id="app_a",
            project_id="proj_1",
        ),
        _make_episode(
            owner_id="u1",
            entry_id="ep_3",
            timestamp_ms=1500,
            app_id="app_b",
            project_id="proj_2",
        ),
    ]
    await episode_repo.add(rows)

    result = await episode_repo.list_by_owner_after_ts(
        owner_id="u1",
        after_ts=0,
        parent_type="memcell",
        app_id="app_a",
        project_id="proj_1",
    )

    assert len(result) == 2
    assert all(ep.app_id == "app_a" and ep.project_id == "proj_1" for ep in result)


async def test_returns_empty_when_no_match(episode_repo: _EpisodeRepo) -> None:
    """Returns empty list when no rows match all filters."""
    rows = [
        _make_episode(owner_id="u1", entry_id="ep_1", timestamp_ms=500),
    ]
    await episode_repo.add(rows)

    result = await episode_repo.list_by_owner_after_ts(
        owner_id="u1",
        after_ts=9999,
        parent_type="memcell",
        app_id="default",
        project_id="default",
    )

    assert result == []


async def test_ordering_is_ascending_by_timestamp(episode_repo: _EpisodeRepo) -> None:
    """Results are ordered by timestamp ascending (oldest first)."""
    rows = [
        _make_episode(owner_id="u1", entry_id="ep_3", timestamp_ms=2500),
        _make_episode(owner_id="u1", entry_id="ep_1", timestamp_ms=500),
        _make_episode(owner_id="u1", entry_id="ep_2", timestamp_ms=1500),
    ]
    await episode_repo.add(rows)

    result = await episode_repo.list_by_owner_after_ts(
        owner_id="u1",
        after_ts=0,
        parent_type="memcell",
        app_id="default",
        project_id="default",
    )

    entry_ids = [ep.entry_id for ep in result]
    assert entry_ids == ["ep_1", "ep_2", "ep_3"]
