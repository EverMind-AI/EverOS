"""Tests for :func:`trigger_profile_clustering`.

Mirrors the skill-side test layout: in-memory cluster store + fake
embedder, ``cluster_by_geometry`` patched, call the strategy directly and
check what the cluster store holds; the strategy emits no event.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import structlog.testing
from everalgo.clustering import Cluster as AlgoCluster

from everos.core.locks import LOCKS
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted
from everos.memory.models import Cluster, ClusterKind, ClusterMemberType
from everos.memory.ports import Scope
from everos.memory.strategies import (
    TRIGGER_PROFILE_CLUSTERING_SPEC,
    trigger_profile_clustering,
)
from tests.fakes import InMemoryClusterStore, make_deps, make_stores, seed_sources

_MODULE = "everos.memory.strategies.trigger_profile_clustering"
_ALICE = Scope(
    app_id="default", project_id="default", owner_id="u_alice", owner_type="user"
)


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


def _event(
    *,
    owner_id: str = "u_alice",
    memcell_id: str = "mc_aaaaaaaaaaa1",
    episode_entry_id: str = "ep_20260517_0001",
    episode_text: str = "alice likes hiking",
    episode_timestamp_ms: int = 1_700_000_001_000,
    source: str = "pipeline",
    language: str | None = None,
) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id=episode_entry_id,
        episode_text=episode_text,
        episode_timestamp_ms=episode_timestamp_ms,
        owner_id=owner_id,
        session_id="s_test",
        source=source,
        language=language,
    )


def _with_episode(
    deps: MemoryDeps,
    *,
    owner_id: str = "u_alice",
    episode_entry_id: str = "ep_20260517_0001",
) -> MemoryDeps:
    """The episode the event announces is in md and its memcell exists."""
    return seed_sources(
        deps,
        memcell_id="mc_aaaaaaaaaaa1",
        episode_entry_id=episode_entry_id,
        owner_id=owner_id,
    )


def test_spec() -> None:
    spec = TRIGGER_PROFILE_CLUSTERING_SPEC
    assert spec.name == "trigger_profile_clustering"
    assert EpisodeExtracted in spec.trigger.on
    assert spec.emits == frozenset()
    assert spec.max_retries == 2
    assert spec.applies_to is not None


async def test_creates_new_cluster_when_no_existing() -> None:
    """Empty existing → cluster_by_geometry returns None → new cluster persisted."""
    deps = _with_episode(make_deps())
    with (
        patch(
            f"{_MODULE}.cluster_by_geometry", new=MagicMock(return_value=None)
        ) as mock_cluster,
        patch(f"{_MODULE}.mint_cluster_id", return_value="cl_newuser00001"),
        structlog.testing.capture_logs() as captured,
    ):
        events = await trigger_profile_clustering(_event(), deps)

    new_cluster, existing = mock_cluster.call_args.args
    assert isinstance(new_cluster, AlgoCluster)
    assert new_cluster.id == "cl_newuser00001"
    assert new_cluster.count == 1
    assert new_cluster.last_ts == 1_700_000_001_000
    assert new_cluster.members == ["ep_20260517_0001"]
    assert new_cluster.preview == ["alice likes hiking"]
    assert existing == []
    assert deps.embedding.calls == ["alice likes hiking"]

    clusters = deps.stores.cluster
    assert list(clusters.clusters) == ["cl_newuser00001"]
    assert clusters.owners["cl_newuser00001"] == (_ALICE, "user_memory")
    assert clusters.member_types["cl_newuser00001"] == {"ep_20260517_0001": "episode"}

    assert events == []
    assert any(r.get("event") == "profile_cluster_updated" for r in captured)


async def test_merges_into_existing_cluster_when_algo_matches() -> None:
    """algo returns merged Cluster → persisted under the existing id."""
    deps = _with_episode(make_deps())
    deps.stores.cluster.seed(
        _ALICE,
        "user_memory",
        Cluster(
            id="cl_existing0001",
            centroid=(0.15,) * 8,
            count=1,
            last_ts=1_700_000_000_000,
            preview=["earlier episode"],
            members=["ep_20260517_0000"],
        ),
    )
    merged_cluster = AlgoCluster(
        id="cl_existing0001",
        centroid=np.array([0.17] * 8, dtype=np.float32),
        count=2,
        last_ts=1_700_000_001_000,
        preview=["earlier episode", "alice likes hiking"],
        members=["ep_20260517_0000", "ep_20260517_0001"],
    )
    with patch(
        f"{_MODULE}.cluster_by_geometry", new=MagicMock(return_value=merged_cluster)
    ) as mock_cluster:
        events = await trigger_profile_clustering(_event(), deps)

    (_new, existing) = mock_cluster.call_args.args
    assert [c.id for c in existing] == ["cl_existing0001"]
    persisted = deps.stores.cluster.clusters["cl_existing0001"]
    assert persisted.count == 2
    assert persisted.members == ["ep_20260517_0000", "ep_20260517_0001"]
    assert events == []


# ── partition lock (owner_id-level serialisation) ────────────────────────


class _SlowClusterStore(InMemoryClusterStore):
    """Records entry / exit around a delayed upsert."""

    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self._log = log

    async def upsert(
        self,
        scope: Scope,
        kind: ClusterKind,
        cluster: Cluster,
        *,
        member_type: ClusterMemberType,
    ) -> None:
        # Delay inside the partition-lock critical section so two concurrent
        # runs on the same owner are observably serialised.
        member = cluster.members[0]
        self._log.append(f"enter:{member}")
        await asyncio.sleep(0.01)
        self._log.append(f"leave:{member}")


async def _run_serialisation_probe(owner_a: str, owner_b: str) -> list[str]:
    """Drive two trigger_profile_clustering runs and record entry/exit order."""
    log: list[str] = []
    deps = make_deps(stores=make_stores(cluster=_SlowClusterStore(log)))
    for owner, entry_id in ((owner_a, "ep_run_a"), (owner_b, "ep_run_b")):
        _with_episode(deps, owner_id=owner, episode_entry_id=entry_id)
    with patch(f"{_MODULE}.cluster_by_geometry", new=MagicMock(return_value=None)):
        await asyncio.gather(
            trigger_profile_clustering(
                _event(owner_id=owner_a, episode_entry_id="ep_run_a"), deps
            ),
            trigger_profile_clustering(
                _event(owner_id=owner_b, episode_entry_id="ep_run_b"), deps
            ),
        )
    return log


async def test_partition_lock_serialises_runs_on_same_owner() -> None:
    """Two runs sharing ``owner_id`` must not overlap critical sections."""
    log = await _run_serialisation_probe("u_alice", "u_alice")
    assert log in (
        ["enter:ep_run_a", "leave:ep_run_a", "enter:ep_run_b", "leave:ep_run_b"],
        ["enter:ep_run_b", "leave:ep_run_b", "enter:ep_run_a", "leave:ep_run_a"],
    )


async def test_partition_lock_lets_different_owners_run_in_parallel() -> None:
    """Runs on distinct ``owner_id`` must overlap (no false serialisation)."""
    log = await _run_serialisation_probe("u_alice", "u_bob")
    assert log.index("enter:ep_run_a") < log.index("leave:ep_run_b")
    assert log.index("enter:ep_run_b") < log.index("leave:ep_run_a")


# ── body-guard (embedding provider) ──────────────────────────────────────


async def test_returns_without_side_effects_when_embedding_unavailable() -> None:
    """No embedding provider → early return; no store access, no event."""
    deps = make_deps(embedding=None)
    with structlog.testing.capture_logs() as captured:
        events = await trigger_profile_clustering(_event(), deps)

    assert events == []
    assert deps.stores.cluster.upserts == []
    gated = [
        e
        for e in captured
        if e.get("event") == "strategy_gated_off_embedding_unavailable"
    ]
    assert len(gated) == 1
    assert gated[0]["strategy_name"] == "trigger_profile_clustering"
    assert gated[0]["owner_id"] == "u_alice"


def test_applies_to_rejects_non_pipeline_source() -> None:
    """Events with source != 'pipeline' must not pass the applies_to gate."""
    applies_to = TRIGGER_PROFILE_CLUSTERING_SPEC.applies_to
    assert callable(applies_to)
    deps = make_deps()
    assert applies_to(_event(), deps) is True
    assert applies_to(_event(source="reflection"), deps) is False


# ── Embedding outside the clustering lock (memory delete, spec 1 §5.5) ──


class _LockCheckingEmbedder:
    """Embeds, and records whether the caller held the clustering lock."""

    dim = 4

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.held_cluster_lock: list[bool] = []

    async def embed(self, text: str) -> list[float]:
        self.held_cluster_lock.append(
            any(
                kind == "partition:trigger_profile_clustering"
                for kind, _ in LOCKS.held()
            )
        )
        return self.vectors.get(text, [1.0, 0.0, 0.0, 0.0])


async def test_clustering_embeds_without_holding_cluster_lock() -> None:
    """A11: a slow embedding must not hold the owner's clustering lock (a
    single-id deletion waits on it)."""
    embedder = _LockCheckingEmbedder()
    deps = _with_episode(make_deps(embedding=embedder))

    await trigger_profile_clustering(_event(), deps)

    assert embedder.held_cluster_lock == [False]


_P5_VECTORS = {
    0: [1.0, 0.0, 0.0, 0.0],
    1: [0.0, 1.0, 0.0, 0.0],
    2: [0.7, 0.7, 0.0, 0.0],
}
_P5_PLAN = [(0, 0), (1, 1), (2, 0), (3, 2), (4, 1), (5, 0), (6, 2), (7, 1)]


async def test_clustering_result_unchanged_for_fixture_sequence() -> None:
    """P5: moving the embedding out of the lock changes no clustering result.

    The expected clusters were recorded on the code before the move (commit
    after "feat(memory): check sources before every derived write"), by
    running this same sequence and printing the clusters' members and count.
    """
    embedder = _LockCheckingEmbedder()
    deps = make_deps(embedding=embedder)
    for n, group in _P5_PLAN:
        text = f"entry {n}"
        embedder.vectors[text] = _P5_VECTORS[group]
        entry = f"ep_20260101_{n + 1:04d}"
        seed_sources(deps, memcell_id=f"mc_{n}", episode_entry_id=entry, owner_id="u")
        await trigger_profile_clustering(
            EpisodeExtracted(
                memcell_id=f"mc_{n}",
                episode_entry_id=entry,
                episode_text=text,
                episode_timestamp_ms=1_700_000_000_000 + n * 60_000,
                owner_id="u",
                session_id="s",
            ),
            deps,
        )

    clusters = deps.stores.cluster.clusters.values()  # type: ignore[attr-defined]
    assert sorted((sorted(c.members), c.count) for c in clusters) == [
        (
            [
                "ep_20260101_0001",
                "ep_20260101_0003",
                "ep_20260101_0004",
                "ep_20260101_0006",
                "ep_20260101_0007",
            ],
            5,
        ),
        (["ep_20260101_0002", "ep_20260101_0005", "ep_20260101_0008"], 3),
    ]
