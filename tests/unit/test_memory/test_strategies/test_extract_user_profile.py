"""Tests for :func:`extract_user_profile` (cluster path).

The strategy reads the profile and clusters from their stores, resolves
memcells through the episode index and the memcell store, and runs
``ProfileExtractor`` (patched). In-memory doubles from :mod:`tests.fakes`
stand in for every port, so the assertions are on what was stored.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from everalgo.types import ChatMessage, MemCell
from everalgo.types import Profile as AlgoProfile

from everos.memory._partition_locks import _reset_for_tests
from everos.memory.deps import MemoryDeps
from everos.memory.events import ProfileClusterUpdated
from everos.memory.models import Cluster, Episode, Profile
from everos.memory.ports import MemoryRef, Scope, Stored
from everos.memory.strategies import EXTRACT_USER_PROFILE_SPEC, extract_user_profile
from tests.fakes import make_deps

_EXTRACTOR = "everos.memory.strategies.extract_user_profile.ProfileExtractor"


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


def _scope(owner_id: str = "u_alice") -> Scope:
    return Scope(
        app_id="default", project_id="default", owner_id=owner_id, owner_type="user"
    )


def _event(
    *,
    owner_id: str = "u_alice",
    memcell_id: str = "mc_aaaaaaaaaaa1",
    cluster_id: str = "cl_user00000001",
) -> ProfileClusterUpdated:
    return ProfileClusterUpdated(
        memcell_id=memcell_id,
        cluster_id=cluster_id,
        owner_id=owner_id,
    )


def _cluster(*, cluster_id: str, members: list[str], last_ts: int) -> Cluster:
    return Cluster(
        id=cluster_id,
        centroid=(0.0,) * 8,
        count=len(members),
        last_ts=last_ts,
        members=members,
    )


def _indexed_episode(scope: Scope, entry_id: str, memcell_id: str) -> Stored[Episode]:
    return Stored(
        ref=MemoryRef(
            id=f"{scope.owner_id}_{entry_id}", entry_id=entry_id, md_path="ep.md"
        ),
        scope=scope,
        item=Episode(
            owner_id=scope.owner_id,
            episode="...",
            timestamp=1_700_000_000_000,
            parent_id=memcell_id,
            parent_type="memcell",
        ),
    )


def _memcell(memcell_id: str, *, sender_id: str, ts_ms: int) -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id=f"{memcell_id}_m1",
                role="user",
                content=f"hi from {sender_id}",
                timestamp=ts_ms,
                sender_id=sender_id,
            ),
        ],
        timestamp=ts_ms,
    )


def _seed(
    deps: MemoryDeps,
    scope: Scope,
    *,
    cluster_id: str,
    entry_id: str,
    memcell_id: str,
    last_ts: int,
) -> None:
    """One cluster → one indexed episode → one stored memcell."""
    deps.stores.cluster.seed(
        scope,
        "user_memory",
        _cluster(cluster_id=cluster_id, members=[entry_id], last_ts=last_ts),
    )
    deps.indexes.episode.seed(_indexed_episode(scope, entry_id, memcell_id))
    deps.stores.memcell.seed(
        memcell_id, _memcell(memcell_id, sender_id=scope.owner_id, ts_ms=last_ts)
    )


def test_spec() -> None:
    spec = EXTRACT_USER_PROFILE_SPEC
    assert spec.name == "extract_user_profile"
    assert ProfileClusterUpdated in spec.trigger.on
    assert spec.emits == frozenset()
    assert spec.max_retries == 2


async def test_init_mode_writes_profile_when_no_existing() -> None:
    """No prior profile → ProfileExtractor invoked without ``old_profile``."""
    deps = make_deps()
    _seed(
        deps,
        _scope(),
        cluster_id="cl_user00000001",
        entry_id="ep_20260101_0001",
        memcell_id="mc_aaaaaaaaaaa1",
        last_ts=1_700_000_001_000,
    )
    new_profile = AlgoProfile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "Alice is a hiker.",
            "timestamp": 1_700_000_001_000,
            "explicit_info": ["lives in tokyo"],
            "implicit_traits": ["adventurous"],
        }
    )
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract = AsyncMock(return_value=new_profile)
        events = await extract_user_profile(_event(), deps)

    assert events == []
    # INIT mode — old_profile is None.
    extractor_call = extractor_cls.return_value.aextract.call_args
    assert extractor_call.kwargs["old_profile"] is None
    assert extractor_call.kwargs["sender_id"] == "u_alice"
    assert [mc.timestamp for mc in extractor_call.args[0]] == [1_700_000_001_000]

    # The freshly extracted profile is stored under the owner's scope.
    stored = deps.stores.profile.profiles[_scope()]
    assert stored.summary == "Alice is a hiker."
    assert stored.timestamp == 1_700_000_001_000
    assert stored.explicit_info == ["lives in tokyo"]
    assert stored.implicit_traits == ["adventurous"]


async def test_update_mode_rehydrates_old_profile() -> None:
    """Existing profile → algo Profile rehydrated and passed as old_profile."""
    deps = make_deps()
    _seed(
        deps,
        _scope(),
        cluster_id="cl_user00000001",
        entry_id="ep_20260101_0001",
        memcell_id="mc_aaaaaaaaaaa1",
        last_ts=1_700_000_002_000,
    )
    deps.stores.profile.profiles[_scope()] = Profile(
        owner_id="u_alice",
        summary="prior summary",
        timestamp=1_700_000_000_000,
        explicit_info=["prior fact"],
        implicit_traits=["prior trait"],
    )
    new_profile = AlgoProfile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "updated summary",
            "timestamp": 1_700_000_002_000,
            "explicit_info": ["prior fact", "new fact"],
            "implicit_traits": ["prior trait"],
        }
    )
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract = AsyncMock(return_value=new_profile)
        await extract_user_profile(_event(), deps)

    # UPDATE mode — old_profile is the rehydrated algo type carrying prior fields.
    old = extractor_cls.return_value.aextract.call_args.kwargs["old_profile"]
    assert isinstance(old, AlgoProfile)
    assert old.summary == "prior summary"
    assert old.timestamp == 1_700_000_000_000
    assert old.model_dump()["explicit_info"] == ["prior fact"]
    assert deps.stores.profile.profiles[_scope()].summary == "updated summary"


async def test_skips_when_no_members() -> None:
    """An empty target cluster set (no fresh clusters) → no extractor call."""
    # Existing profile timestamp newer than every cluster's last_ts → no
    # target_cluster matches `last_ts > last_profile_ts`, and the current
    # cluster_id names a cluster that does not exist, so nothing is selected.
    deps = make_deps()
    deps.stores.cluster.seed(
        _scope(),
        "user_memory",
        _cluster(
            cluster_id="cl_other000001",
            members=["ep_other00000"],
            last_ts=1_600_000_000_000,
        ),
    )
    deps.stores.profile.profiles[_scope()] = Profile(
        owner_id="u_alice", summary="prior", timestamp=1_900_000_000_000
    )
    with patch(_EXTRACTOR) as extractor_cls:
        await extract_user_profile(_event(cluster_id="cl_unknown00000"), deps)

    extractor_cls.assert_not_called()
    assert deps.stores.profile.writes == []


# ── partition lock (owner_id-level serialisation) ────────────────────────


async def _run_serialisation_probe(owner_a: str, owner_b: str) -> list[str]:
    """Drive two extract_user_profile runs and record entry/exit order."""
    log: list[str] = []

    async def slow_aextract(_memcells, *, sender_id, **_kwargs):
        log.append(f"enter:{sender_id}")
        await asyncio.sleep(0.01)
        log.append(f"leave:{sender_id}")
        return AlgoProfile(
            owner_id=sender_id,
            summary="summary",
            timestamp=1_700_000_000_000,
            explicit_info=[],
            implicit_traits=[],
        )

    deps = make_deps()
    for owner, tag in ((owner_a, "a"), (owner_b, "b")):
        _seed(
            deps,
            _scope(owner),
            cluster_id=f"cl_{tag}",
            entry_id=f"ep_{tag}",
            memcell_id=f"mc_{tag}",
            last_ts=1_700_000_000_000,
        )
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract = slow_aextract
        await asyncio.gather(
            extract_user_profile(_event(owner_id=owner_a, cluster_id="cl_a"), deps),
            extract_user_profile(_event(owner_id=owner_b, cluster_id="cl_b"), deps),
        )
    return log


async def test_partition_lock_serialises_runs_on_same_owner() -> None:
    """Two runs sharing ``owner_id`` must not overlap critical sections."""
    log = await _run_serialisation_probe("u_alice", "u_alice")
    assert log == ["enter:u_alice", "leave:u_alice", "enter:u_alice", "leave:u_alice"]


async def test_partition_lock_lets_different_owners_run_in_parallel() -> None:
    """Runs on distinct ``owner_id`` must overlap (no false serialisation)."""
    log = await _run_serialisation_probe("u_alice", "u_bob")
    assert log.index("enter:u_alice") < log.index("leave:u_bob")
    assert log.index("enter:u_bob") < log.index("leave:u_alice")
