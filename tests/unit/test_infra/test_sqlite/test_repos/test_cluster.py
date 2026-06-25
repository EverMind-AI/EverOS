"""Tests for :class:`_ClusterRepo` — cluster + cluster_member persistence.

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
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite.repos.cluster import (
    _ClusterRepo,
    mint_cluster_id,
)


@pytest.fixture
async def repo(tmp_path: Path) -> _ClusterRepo:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    engine = create_system_engine(mr.system_db, SqliteSettings())
    factory = create_session_factory(engine)
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return _ClusterRepo(session_factory=factory)


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


def test_mint_cluster_id_shape() -> None:
    cid = mint_cluster_id()
    assert cid.startswith("cl_")
    assert len(cid) == 3 + 12  # ``cl_`` + 12 hex chars


# ── round-trip ─────────────────────────────────────────────────────────


async def test_upsert_then_list_round_trips_full_algo_cluster(
    repo: _ClusterRepo,
) -> None:
    """Insert → list — every algo field survives storage."""
    cluster = _make_cluster(
        cluster_id="cl_aaa000000001",
        centroid_vals=[0.25, -0.5, 0.75],
        members=["mc_one", "mc_two"],
        last_ts_ms=1_700_000_001_500,
        count=2,
        preview=["alice likes hiking", "alice plans tokyo"],
    )
    await repo.upsert_with_members(
        cluster,
        owner_id="u_alice",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )

    rows = await repo.list_for_owner("u_alice", "user_memory")
    assert len(rows) == 1
    got = rows[0]
    assert got.id == "cl_aaa000000001"
    assert got.count == 2
    assert got.last_ts == 1_700_000_001_500
    assert got.preview == ["alice likes hiking", "alice plans tokyo"]
    assert got.members == ["mc_one", "mc_two"]
    np.testing.assert_allclose(
        np.asarray(got.centroid),
        np.array([0.25, -0.5, 0.75], dtype=np.float32),
    )


async def test_list_for_owner_isolates_by_owner_and_kind(
    repo: _ClusterRepo,
) -> None:
    """Different owner_id or different kind = separate buckets."""
    alice = _make_cluster(
        cluster_id="cl_alice00000001",
        centroid_vals=[1.0, 0.0],
        members=["mc_a"],
    )
    bob = _make_cluster(
        cluster_id="cl_bob0000000001",
        centroid_vals=[0.0, 1.0],
        members=["mc_b"],
    )
    agent_case = _make_cluster(
        cluster_id="cl_case0000001",
        centroid_vals=[0.5, 0.5],
        members=["ac_20260517_0001"],
    )
    await repo.upsert_with_members(
        alice,
        owner_id="u_alice",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )
    await repo.upsert_with_members(
        bob,
        owner_id="u_bob",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )
    await repo.upsert_with_members(
        agent_case,
        owner_id="agent_42",
        owner_type="agent",
        kind="agent_case",
        member_type="case",
    )

    alice_rows = await repo.list_for_owner("u_alice", "user_memory")
    bob_rows = await repo.list_for_owner("u_bob", "user_memory")
    agent_rows = await repo.list_for_owner("agent_42", "agent_case")
    assert [r.id for r in alice_rows] == ["cl_alice00000001"]
    assert [r.id for r in bob_rows] == ["cl_bob0000000001"]
    assert [r.id for r in agent_rows] == ["cl_case0000001"]


# ── upsert (idempotency + members merge) ────────────────────────────────


async def test_upsert_appends_new_members_and_overwrites_scalar_fields(
    repo: _ClusterRepo,
) -> None:
    """A second upsert with new members appends; centroid / count / preview replace."""
    initial = _make_cluster(
        cluster_id="cl_xxxxxxxxxxx1",
        centroid_vals=[1.0, 0.0],
        members=["mc_one"],
        count=1,
        preview=["first sample"],
    )
    await repo.upsert_with_members(
        initial,
        owner_id="u_alice",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )

    # Merge: same cluster_id, count up, member list grew, centroid shifted.
    updated = _make_cluster(
        cluster_id="cl_xxxxxxxxxxx1",
        centroid_vals=[0.5, 0.5],
        members=["mc_one", "mc_two"],
        count=2,
        preview=["first sample", "second sample"],
        last_ts_ms=1_700_000_002_000,
    )
    await repo.upsert_with_members(
        updated,
        owner_id="u_alice",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )

    rows = await repo.list_for_owner("u_alice", "user_memory")
    assert len(rows) == 1
    got = rows[0]
    assert got.count == 2
    assert got.members == ["mc_one", "mc_two"]
    assert got.preview == ["first sample", "second sample"]
    np.testing.assert_allclose(
        np.asarray(got.centroid),
        np.array([0.5, 0.5], dtype=np.float32),
    )


async def test_upsert_is_idempotent_under_retry(repo: _ClusterRepo) -> None:
    """OME at-least-once retry: same upsert twice → state unchanged, no duplicates."""
    cluster = _make_cluster(
        cluster_id="cl_idempot00001",
        centroid_vals=[0.1, 0.9],
        members=["mc_one", "mc_two"],
        count=2,
    )
    await repo.upsert_with_members(
        cluster,
        owner_id="u_alice",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )
    await repo.upsert_with_members(
        cluster,
        owner_id="u_alice",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )
    rows = await repo.list_for_owner("u_alice", "user_memory")
    assert len(rows) == 1
    assert rows[0].members == ["mc_one", "mc_two"]


async def test_upsert_rejects_unset_cluster_id(repo: _ClusterRepo) -> None:
    """Algo's ``Cluster.id`` is caller-supplied — None is a programming error."""
    cluster = AlgoCluster(
        id=None,
        centroid=np.array([1.0], dtype=np.float32),
        count=1,
        last_ts=1_700_000_000_000,
        preview=[],
        members=["mc_one"],
    )
    with pytest.raises(ValueError, match="cluster_id"):
        await repo.upsert_with_members(
            cluster,
            owner_id="u_alice",
            owner_type="user",
            kind="user_memory",
            member_type="memcell",
        )


# ── reverse lookup ──────────────────────────────────────────────────────


async def test_find_cluster_id_for_member_reverse_lookup(
    repo: _ClusterRepo,
) -> None:
    """``(member_type, member_id) → cluster_id`` index works both ways across kinds."""
    user_cluster = _make_cluster(
        cluster_id="cl_user0000001",
        centroid_vals=[1.0, 0.0],
        members=["mc_one"],
    )
    case_cluster = _make_cluster(
        cluster_id="cl_case0000001",
        centroid_vals=[0.0, 1.0],
        members=["ac_20260517_0001"],
    )
    await repo.upsert_with_members(
        user_cluster,
        owner_id="u_alice",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )
    await repo.upsert_with_members(
        case_cluster,
        owner_id="agent_42",
        owner_type="agent",
        kind="agent_case",
        member_type="case",
    )

    assert (
        await repo.find_cluster_id_for_member("memcell", "mc_one") == "cl_user0000001"
    )
    assert (
        await repo.find_cluster_id_for_member("case", "ac_20260517_0001")
        == "cl_case0000001"
    )
    # Type-discriminated: same id under wrong type misses.
    assert await repo.find_cluster_id_for_member("case", "mc_one") is None
    assert await repo.find_cluster_id_for_member("memcell", "ac_20260517_0001") is None
    assert await repo.find_cluster_id_for_member("memcell", "mc_missing") is None


# ── remove_members ─────────────────────────────────────────────────────


async def test_remove_members_deletes_specified(repo: _ClusterRepo) -> None:
    """Removing a subset of members leaves the rest intact."""
    cluster = _make_cluster(
        cluster_id="cl_rm_000000001",
        centroid_vals=[1.0, 0.0],
        members=["mc_one", "mc_two", "mc_three"],
        count=3,
    )
    await repo.upsert_with_members(
        cluster,
        owner_id="u_alice",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )

    await repo.remove_members("cl_rm_000000001", {"mc_one", "mc_three"})

    members = await repo.get_members_with_type("cl_rm_000000001")
    assert [mid for mid, _ in members] == ["mc_two"]


async def test_remove_members_empty_set_is_noop(
    repo: _ClusterRepo,
) -> None:
    """An empty member_ids set should not touch the database."""
    cluster = _make_cluster(
        cluster_id="cl_noop0000001",
        centroid_vals=[1.0, 0.0],
        members=["mc_one"],
    )
    await repo.upsert_with_members(
        cluster,
        owner_id="u_alice",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )

    await repo.remove_members("cl_noop0000001", set())

    members = await repo.get_members_with_type("cl_noop0000001")
    assert len(members) == 1


# ── add_member ─────────────────────────────────────────────────────────


async def test_add_member_with_episode_type(repo: _ClusterRepo) -> None:
    """Add a single member and verify it appears in the membership list."""
    cluster = _make_cluster(
        cluster_id="cl_add_00000001",
        centroid_vals=[1.0, 0.0],
        members=["mc_one"],
    )
    await repo.upsert_with_members(
        cluster,
        owner_id="u_alice",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )

    await repo.add_member("cl_add_00000001", "ep_new_001", "episode")

    members = await repo.get_members_with_type("cl_add_00000001")
    member_ids = [mid for mid, _ in members]
    assert "ep_new_001" in member_ids
    # Verify type stored correctly
    ep_row = [(mid, mt) for mid, mt in members if mid == "ep_new_001"]
    assert ep_row[0][1] == "episode"


# ── update_metadata ────────────────────────────────────────────────────


async def test_update_metadata_changes_cluster_row(
    repo: _ClusterRepo,
) -> None:
    """update_metadata overwrites centroid, count, last_ts_ms, preview."""
    cluster = _make_cluster(
        cluster_id="cl_meta0000001",
        centroid_vals=[1.0, 0.0],
        members=["mc_one"],
        count=1,
        last_ts_ms=1_700_000_000_000,
        preview=["old preview"],
    )
    await repo.upsert_with_members(
        cluster,
        owner_id="u_alice",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )

    new_centroid = np.array([0.5, 0.5], dtype=np.float32).tobytes()
    await repo.update_metadata(
        "cl_meta0000001",
        centroid_blob=new_centroid,
        count=3,
        last_ts_ms=1_700_000_099_000,
        preview_json='["new preview"]',
    )

    rows = await repo.list_for_owner("u_alice", "user_memory")
    assert len(rows) == 1
    got = rows[0]
    assert got.count == 3
    assert got.last_ts == 1_700_000_099_000
    assert got.preview == ["new preview"]
    np.testing.assert_allclose(
        np.asarray(got.centroid),
        np.array([0.5, 0.5], dtype=np.float32),
    )


# ── list_ids_and_member_counts ─────────────────────────────────────────


async def test_list_ids_and_member_counts_returns_actual_member_count(
    repo: _ClusterRepo,
) -> None:
    """Count comes from cluster_member rows, not the Cluster.count field."""
    c1 = _make_cluster(
        cluster_id="cl_cnt_00000001",
        centroid_vals=[1.0, 0.0],
        members=["mc_one", "mc_two"],
        count=99,  # deliberately wrong — repo counts actual rows
    )
    c2 = _make_cluster(
        cluster_id="cl_cnt_00000002",
        centroid_vals=[0.0, 1.0],
        members=["mc_three"],
        count=99,
    )
    await repo.upsert_with_members(
        c1,
        owner_id="u_alice",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )
    await repo.upsert_with_members(
        c2,
        owner_id="u_alice",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )

    result = await repo.list_ids_and_member_counts("u_alice", "user_memory")
    result_dict = dict(result)
    assert result_dict["cl_cnt_00000001"] == 2
    assert result_dict["cl_cnt_00000002"] == 1


# ── get_members_with_type ──────────────────────────────────────────────


async def test_get_members_with_type_returns_tuples(
    repo: _ClusterRepo,
) -> None:
    """Returns (member_id, member_type) tuples in insertion order."""
    cluster = _make_cluster(
        cluster_id="cl_mtype000001",
        centroid_vals=[1.0, 0.0],
        members=["mc_one", "mc_two"],
    )
    await repo.upsert_with_members(
        cluster,
        owner_id="u_alice",
        owner_type="user",
        kind="user_memory",
        member_type="memcell",
    )

    members = await repo.get_members_with_type("cl_mtype000001")
    assert len(members) == 2
    assert members[0] == ("mc_one", "memcell")
    assert members[1] == ("mc_two", "memcell")
