"""Round-3 hygiene tests for :class:`_EpisodeRepo`.

Pins three contracts introduced by round-3 review finding #9:

- ``list_by_owner_after_ts`` and ``count_by_owner`` both exclude
  ``deprecated_by IS NOT NULL`` rows — matches the filter idiom used
  by the search filter compiler and the reflection read paths.
- ``list_by_owner_after_ts(columns=[...])`` returns raw dicts instead
  of full ``Episode`` objects, letting callers project only the
  fields they need (the vector columns are 1024-D and would otherwise
  ride along every call).
- ``list_by_owner_after_ts(limit=N)`` bounds the row count, letting
  callers avoid pulling the entire history for a hot owner.
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


def _episode(
    *,
    entry_id: str,
    timestamp_ms: int,
    deprecated_by: str | None = None,
    parent_id: str = "mc_default",
) -> Episode:
    ts = dt.datetime.fromtimestamp(timestamp_ms / 1000.0, tz=dt.UTC)
    return Episode(
        id=f"u1_{entry_id}",
        entry_id=entry_id,
        owner_id="u1",
        owner_type="user",
        app_id="default",
        project_id="default",
        session_id="sess",
        timestamp=ts,
        parent_type="memcell",
        parent_id=parent_id,
        sender_ids=["user"],
        episode="hello",
        episode_tokens="hello",
        md_path=f"users/u1/episodes/{entry_id}.md",
        content_sha256="a" * 64,
        deprecated_by=deprecated_by,
    )


# ── deprecated_by filter ──────────────────────────────────────────────────


async def test_list_after_ts_excludes_deprecated_rows(
    episode_repo: _EpisodeRepo,
) -> None:
    """Reflection-superseded episodes must not leak into the direct-path
    selector — otherwise profile extraction would feed stale memcell
    references to the LLM."""
    await episode_repo.add(
        [
            _episode(entry_id="live_1", timestamp_ms=1000),
            _episode(entry_id="live_2", timestamp_ms=2000),
            _episode(
                entry_id="dep",
                timestamp_ms=3000,
                deprecated_by="cluster_x",
            ),
        ]
    )

    result = await episode_repo.list_by_owner_after_ts(
        owner_id="u1",
        after_ts=0,
        parent_type="memcell",
    )

    assert len(result) == 2
    assert {ep.entry_id for ep in result} == {"live_1", "live_2"}  # type: ignore[union-attr]


async def test_count_by_owner_excludes_deprecated_rows(
    episode_repo: _EpisodeRepo,
) -> None:
    """The Tier-1 profile-throttle count must ignore Reflection-superseded
    rows — matches ``list_by_owner_after_ts`` so the throttle rate keeps
    pace with the surface the strategy actually operates on."""
    await episode_repo.add(
        [
            _episode(entry_id="live_1", timestamp_ms=1000),
            _episode(entry_id="live_2", timestamp_ms=2000),
            _episode(
                entry_id="dep",
                timestamp_ms=3000,
                deprecated_by="cluster_x",
            ),
        ]
    )

    count = await episode_repo.count_by_owner("u1", parent_type="memcell")

    assert count == 2


# ── column projection ─────────────────────────────────────────────────────


async def test_list_after_ts_columns_returns_raw_dicts(
    episode_repo: _EpisodeRepo,
) -> None:
    """`columns=[...]` opts into a raw-dict return so callers can skip
    the 1024-D vector columns. Full ``Episode`` reconstruction only
    happens when no projection is passed."""
    await episode_repo.add(
        [
            _episode(entry_id="a", timestamp_ms=1000, parent_id="mc_a"),
            _episode(entry_id="b", timestamp_ms=2000, parent_id="mc_b"),
        ]
    )

    result = await episode_repo.list_by_owner_after_ts(
        owner_id="u1",
        after_ts=0,
        parent_type="memcell",
        columns=["parent_id"],
    )

    assert all(isinstance(row, dict) for row in result)
    parent_ids = {row["parent_id"] for row in result}  # type: ignore[index]
    assert parent_ids == {"mc_a", "mc_b"}


async def test_list_after_ts_columns_none_returns_typed_episodes(
    episode_repo: _EpisodeRepo,
) -> None:
    """Backwards-compat: default (`columns=None`) still returns typed
    ``Episode`` objects — no change for callers that didn't opt in."""
    await episode_repo.add(
        [_episode(entry_id="a", timestamp_ms=1000, parent_id="mc_a")]
    )

    result = await episode_repo.list_by_owner_after_ts(
        owner_id="u1",
        after_ts=0,
        parent_type="memcell",
    )

    assert len(result) == 1
    assert isinstance(result[0], Episode)


# ── limit ────────────────────────────────────────────────────────────────


async def test_list_after_ts_respects_limit(
    episode_repo: _EpisodeRepo,
) -> None:
    """Callers with large historical windows should bound the pull so
    they don't drag every memcell for the owner across the wire."""
    rows = [_episode(entry_id=f"e{i}", timestamp_ms=1000 + i) for i in range(20)]
    await episode_repo.add(rows)

    result = await episode_repo.list_by_owner_after_ts(
        owner_id="u1",
        after_ts=0,
        parent_type="memcell",
        limit=5,
    )

    assert len(result) == 5
