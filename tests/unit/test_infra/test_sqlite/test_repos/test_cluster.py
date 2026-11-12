"""Tests for :class:`ClusterRepo` — cluster + cluster_member persistence.

Verifies the boundary translations between the algo value object
(:class:`everalgo.clustering.Cluster`) and the two-table storage shape:

- centroid ``np.ndarray`` ↔ raw ``bytes``,
- ``last_ts`` int ms-epoch stored verbatim (no datetime conversion),
- ``preview`` ``list[str]`` ↔ JSON,
- ``members`` ``list[str]`` ↔ ``cluster_member`` rows (forward + reverse).

The repo is the only path that touches the storage; downstream cluster
strategies must always see a fully-hydrated :class:`AlgoCluster` on read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from everalgo.clustering import Cluster as AlgoCluster
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import create_session_factory, create_system_engine
from everos.infra.persistence.sqlite._repos.cluster import ClusterRepo


@pytest.fixture
async def repo(tmp_path: Path) -> ClusterRepo:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    engine = create_system_engine(mr.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return ClusterRepo(session_factory=factory)


def _make_cluster(
    *,
    cluster_id: str,
    centroid_vals: list[float],
    members: list[str],
    last_ts_ms: int = 1_700_000_000_000,
    count: int = 1,
    preview: list[str] | None = None,
) -> AlgoCluster:
    return AlgoCluster(
        id=cluster_id,
        centroid=np.array(centroid_vals, dtype=np.float32),
        count=count,
        last_ts=last_ts_ms,
        preview=preview or [],
        members=members,
    )


# ── round-trip ─────────────────────────────────────────────────────────


async def test_upsert_then_list_round_trips_full_algo_cluster(
    repo: ClusterRepo,
) -> None:
    """Insert → list — every algo field survives storage."""
    cluster = _make_cluster(
        cluster_id="cl_aaa000000001",
        centroid_vals=[0.25, -0.5, 0.75],
        members=["ep_one", "ep_two"],
        last_ts_ms=1_700_000_001_500,
        count=2,
        preview=["alice likes hiking", "alice plans tokyo"],
    )
    await repo.upsert_with_members(
        cluster,
        owner_id="u_alice",
        owner_type="user",
        kind="user_scene",
        member_type="episode",
    )

    rows = await repo.list_for_owner("u_alice", "user_scene")
    assert len(rows) == 1
    got = rows[0]
    assert got.id == "cl_aaa000000001"
    assert got.count == 2
    assert got.last_ts == 1_700_000_001_500
    assert got.preview == ["alice likes hiking", "alice plans tokyo"]
    assert got.members == ["ep_one", "ep_two"]
    np.testing.assert_allclose(
        np.asarray(got.centroid),
        np.array([0.25, -0.5, 0.75], dtype=np.float32),
    )


async def test_list_for_owner_isolates_by_owner_and_kind(
    repo: ClusterRepo,
) -> None:
    """Different owner_id or different kind = separate buckets."""
    alice = _make_cluster(
        cluster_id="cl_alice00000001",
        centroid_vals=[1.0, 0.0],
        members=["ep_a"],
    )
    bob = _make_cluster(
        cluster_id="cl_bob0000000001",
        centroid_vals=[0.0, 1.0],
        members=["ep_b"],
    )
    group = _make_cluster(
        cluster_id="cl_group000001",
        centroid_vals=[0.5, 0.5],
        members=["gep_20260517_0001"],
    )
    await repo.upsert_with_members(
        alice,
        owner_id="u_alice",
        owner_type="user",
        kind="user_scene",
        member_type="episode",
    )
    await repo.upsert_with_members(
        bob,
        owner_id="u_bob",
        owner_type="user",
        kind="user_scene",
        member_type="episode",
    )
    await repo.upsert_with_members(
        group,
        owner_id="trip",
        owner_type="group",
        kind="group_scene",
        member_type="episode",
    )

    alice_rows = await repo.list_for_owner("u_alice", "user_scene")
    bob_rows = await repo.list_for_owner("u_bob", "user_scene")
    group_rows = await repo.list_for_owner("trip", "group_scene")
    assert [r.id for r in alice_rows] == ["cl_alice00000001"]
    assert [r.id for r in bob_rows] == ["cl_bob0000000001"]
    assert [r.id for r in group_rows] == ["cl_group000001"]
    assert await repo.list_for_owner("trip", "user_scene") == []


# ── upsert (idempotency + members merge) ────────────────────────────────


async def test_upsert_appends_new_members_and_overwrites_scalar_fields(
    repo: ClusterRepo,
) -> None:
    """A second upsert with new members appends; centroid / count / preview replace."""
    initial = _make_cluster(
        cluster_id="cl_xxxxxxxxxxx1",
        centroid_vals=[1.0, 0.0],
        members=["ep_one"],
        count=1,
        preview=["first sample"],
    )
    await repo.upsert_with_members(
        initial,
        owner_id="u_alice",
        owner_type="user",
        kind="user_scene",
        member_type="episode",
    )

    # Merge: same cluster_id, count up, member list grew, centroid shifted.
    updated = _make_cluster(
        cluster_id="cl_xxxxxxxxxxx1",
        centroid_vals=[0.5, 0.5],
        members=["ep_one", "ep_two"],
        count=2,
        preview=["first sample", "second sample"],
        last_ts_ms=1_700_000_002_000,
    )
    await repo.upsert_with_members(
        updated,
        owner_id="u_alice",
        owner_type="user",
        kind="user_scene",
        member_type="episode",
    )

    rows = await repo.list_for_owner("u_alice", "user_scene")
    assert len(rows) == 1
    got = rows[0]
    assert got.count == 2
    assert got.members == ["ep_one", "ep_two"]
    assert got.preview == ["first sample", "second sample"]
    np.testing.assert_allclose(
        np.asarray(got.centroid),
        np.array([0.5, 0.5], dtype=np.float32),
    )


async def test_upsert_is_idempotent_under_retry(repo: ClusterRepo) -> None:
    """OME at-least-once retry: same upsert twice → state unchanged, no duplicates."""
    cluster = _make_cluster(
        cluster_id="cl_idempot00001",
        centroid_vals=[0.1, 0.9],
        members=["ep_one", "ep_two"],
        count=2,
    )
    await repo.upsert_with_members(
        cluster,
        owner_id="u_alice",
        owner_type="user",
        kind="user_scene",
        member_type="episode",
    )
    await repo.upsert_with_members(
        cluster,
        owner_id="u_alice",
        owner_type="user",
        kind="user_scene",
        member_type="episode",
    )
    rows = await repo.list_for_owner("u_alice", "user_scene")
    assert len(rows) == 1
    assert rows[0].members == ["ep_one", "ep_two"]


async def test_upsert_rejects_unset_cluster_id(repo: ClusterRepo) -> None:
    """Algo's ``Cluster.id`` is caller-supplied — None is a programming error."""
    cluster = AlgoCluster(
        id=None,
        centroid=np.array([1.0], dtype=np.float32),
        count=1,
        last_ts=1_700_000_000_000,
        preview=[],
        members=["ep_one"],
    )
    with pytest.raises(ValueError, match="cluster_id"):
        await repo.upsert_with_members(
            cluster,
            owner_id="u_alice",
            owner_type="user",
            kind="user_scene",
            member_type="episode",
        )


# ── reverse lookup ──────────────────────────────────────────────────────


async def test_find_cluster_id_for_member_reverse_lookup(
    repo: ClusterRepo,
) -> None:
    """``(member_type, member_id) → cluster_id`` index works both ways across kinds."""
    user_cluster = _make_cluster(
        cluster_id="cl_user0000001",
        centroid_vals=[1.0, 0.0],
        members=["ep_one"],
    )
    group_cluster = _make_cluster(
        cluster_id="cl_group000001",
        centroid_vals=[0.0, 1.0],
        members=["gep_20260517_0001"],
    )
    await repo.upsert_with_members(
        user_cluster,
        owner_id="u_alice",
        owner_type="user",
        kind="user_scene",
        member_type="episode",
    )
    await repo.upsert_with_members(
        group_cluster,
        owner_id="trip",
        owner_type="group",
        kind="group_scene",
        member_type="episode",
    )

    assert (
        await repo.find_cluster_id_for_member(
            "episode",
            "ep_one",
            app_id="default",
            project_id="default",
            owner_id="u_alice",
        )
        == "cl_user0000001"
    )
    assert (
        await repo.find_cluster_id_for_member(
            "episode",
            "gep_20260517_0001",
            app_id="default",
            project_id="default",
            owner_id="trip",
        )
        == "cl_group000001"
    )
    # Type-discriminated: the same id under another member type misses.
    assert (
        await repo.find_cluster_id_for_member(
            "memcell",
            "ep_one",
            app_id="default",
            project_id="default",
            owner_id="u_alice",
        )
        is None
    )
    # Owner-scoped: one owner's member is no member of the other owner's.
    assert (
        await repo.find_cluster_id_for_member(
            "episode",
            "gep_20260517_0001",
            app_id="default",
            project_id="default",
            owner_id="u_alice",
        )
        is None
    )
    assert (
        await repo.find_cluster_id_for_member(
            "episode",
            "ep_missing",
            app_id="default",
            project_id="default",
            owner_id="u_alice",
        )
        is None
    )


async def test_find_cluster_id_for_member_scoped_to_owner(
    repo: ClusterRepo,
) -> None:
    """Same ``member_id`` under two owners resolves to each owner's own cluster.

    Regression: prior signature took only ``(member_type, member_id)`` and
    would falsely resolve owner B's episode ``ep_20260517_00000001`` to
    owner A's cluster (or vice versa) whenever ``entry_id`` collided —
    which is by design for entry_id (see
    ``infra/persistence/markdown/_base/entries.py``: entry_id is per-owner
    unique, cross-owner uniqueness comes from the composite
    ``<owner>_<entry_id>`` at the storage layer).
    """
    entry_id = "ep_20260517_00000001"
    alice_cluster = _make_cluster(
        cluster_id="cl_alice000001",
        centroid_vals=[1.0, 0.0],
        members=[entry_id],
    )
    bob_cluster = _make_cluster(
        cluster_id="cl_bob00000001",
        centroid_vals=[0.0, 1.0],
        members=[entry_id],  # SAME entry_id, different owner
    )
    await repo.upsert_with_members(
        alice_cluster,
        owner_id="u_alice",
        owner_type="user",
        kind="user_scene",
        member_type="episode",
    )
    await repo.upsert_with_members(
        bob_cluster,
        owner_id="u_bob",
        owner_type="user",
        kind="user_scene",
        member_type="episode",
    )

    # Each owner resolves to their own cluster — no cross-owner bleed.
    assert (
        await repo.find_cluster_id_for_member(
            "episode",
            entry_id,
            app_id="default",
            project_id="default",
            owner_id="u_alice",
        )
        == "cl_alice000001"
    )
    assert (
        await repo.find_cluster_id_for_member(
            "episode",
            entry_id,
            app_id="default",
            project_id="default",
            owner_id="u_bob",
        )
        == "cl_bob00000001"
    )
    # A third owner with no cluster attached returns None even though the
    # entry_id exists under two other owners.
    assert (
        await repo.find_cluster_id_for_member(
            "episode",
            entry_id,
            app_id="default",
            project_id="default",
            owner_id="u_carol",
        )
        is None
    )
