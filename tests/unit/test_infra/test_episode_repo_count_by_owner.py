"""Tests for :meth:`_EpisodeRepo.count_by_owner`.

Pins two invariants added when Tier-1 profile throttling started depending
on this counter (round-2 finding #4):

- The default (``parent_type=None``) still counts every row for the owner,
  preserving pre-existing zero-arg behavior for any caller that predates the
  ``parent_type`` filter.
- Passing ``parent_type='memcell'`` narrows the count to memcell-parented
  rows, so Reflection-merged rows (``parent_type='cluster'``) don't inflate
  the throttle counter that ``_select_via_timestamp`` never selects.
"""

from __future__ import annotations

import datetime as dt
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
    from everos.core.persistence.lancedb import LanceRepoBase

    LanceRepoBase._reset_locks_for_tests()


@pytest.fixture
async def episode_repo(tmp_path: Path) -> _EpisodeRepo:
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
) -> Episode:
    timestamp_dt = dt.datetime.fromtimestamp(timestamp_ms / 1000.0, tz=dt.UTC)
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
        vector=None,
    )


async def _seed_mixed(repo: _EpisodeRepo) -> None:
    """Seed 5 memcell episodes + 1 Reflection-merged (cluster) episode."""
    rows = [
        _make_episode(
            owner_id="u1",
            entry_id=f"ep_{i}",
            timestamp_ms=1000 + i,
            parent_type="memcell",
        )
        for i in range(5)
    ]
    rows.append(
        _make_episode(
            owner_id="u1",
            entry_id="ep_merged",
            timestamp_ms=9999,
            parent_type="cluster",
        )
    )
    await repo.add(rows)


async def test_count_by_owner_default_no_parent_type_still_counts_all(
    episode_repo: _EpisodeRepo,
) -> None:
    """Default (no ``parent_type`` kwarg) preserves old behavior: every row counts."""
    await _seed_mixed(episode_repo)

    count = await episode_repo.count_by_owner(
        "u1", app_id="default", project_id="default"
    )

    assert count == 6


async def test_count_by_owner_with_parent_type_memcell_excludes_cluster(
    episode_repo: _EpisodeRepo,
) -> None:
    """``parent_type='memcell'`` narrows the count to the direct-path selector."""
    await _seed_mixed(episode_repo)

    count = await episode_repo.count_by_owner(
        "u1",
        app_id="default",
        project_id="default",
        parent_type="memcell",
    )

    assert count == 5


async def test_count_by_owner_scoped_by_app_and_project(
    episode_repo: _EpisodeRepo,
) -> None:
    """Scope predicates still apply when ``parent_type`` is set."""
    rows = [
        _make_episode(
            owner_id="u1",
            entry_id="ep_a",
            timestamp_ms=1000,
            app_id="app_a",
            project_id="proj_1",
        ),
        _make_episode(
            owner_id="u1",
            entry_id="ep_b",
            timestamp_ms=1001,
            app_id="app_b",
            project_id="proj_2",
        ),
    ]
    await episode_repo.add(rows)

    count = await episode_repo.count_by_owner(
        "u1",
        app_id="app_a",
        project_id="proj_1",
        parent_type="memcell",
    )

    assert count == 1
