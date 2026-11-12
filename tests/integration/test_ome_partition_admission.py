"""Real profile/cluster strategies through OME, with local model and store ports."""

from __future__ import annotations

import asyncio
from pathlib import Path

from everalgo.testing import FakeLLMClient

from everos.core.locks import LOCKS
from everos.infra.ome import OfflineEngine, OMEConfig, RunStatus, to_ome_strategy
from everos.memory.models import Cluster, ClusterKind, ClusterMemberType
from everos.memory.ports import Scope
from everos.memory.strategies import (
    EXTRACT_USER_PROFILE_SPEC,
    TRIGGER_PROFILE_CLUSTERING_SPEC,
)
from tests.fakes import InMemoryClusterStore, make_deps, make_stores
from tests.fakes.llm import user_track_script
from tests.unit.test_memory.test_strategies.test_extract_user_profile import (
    _event,
    _scope,
    _seed,
)


class LocalEmbedding:
    async def embed(self, text: str) -> list[float]:
        return [1.0] + [0.0] * 7


class _SignallingClusterStore(InMemoryClusterStore):
    """Sets ``stored`` once a cluster of ``owner_id`` is upserted."""

    def __init__(self, owner_id: str) -> None:
        super().__init__()
        self.stored = asyncio.Event()
        self._owner_id = owner_id

    async def upsert(
        self,
        scope: Scope,
        kind: ClusterKind,
        cluster: Cluster,
        *,
        member_type: ClusterMemberType,
    ) -> None:
        await super().upsert(scope, kind, cluster, member_type=member_type)
        if scope.owner_id == self._owner_id:
            self.stored.set()


async def test_other_owner_finishes_while_twenty_profile_attempts_are_blocked(
    tmp_path: Path,
) -> None:
    """Twenty of Alice's profile runs queue on her partition with one in the
    model; Bob's cluster is stored and his profile reaches the model meanwhile."""
    release_model = asyncio.Event()
    all_arrived = asyncio.Event()
    second_call = asyncio.Event()
    calls = 0
    active = 0
    peak = 0
    arrivals = 0

    def observe(task, wanted, held) -> None:
        nonlocal arrivals
        if wanted == ("partition:extract_user_profile", "default:default:u_alice"):
            arrivals += 1
            if arrivals == 20:
                all_arrived.set()

    async def answer(messages, *, response_format, **kwargs):
        nonlocal calls, active, peak
        calls += 1
        if calls == 2:
            second_call.set()
        active += 1
        peak = max(peak, active)
        try:
            await release_model.wait()
            return user_track_script(
                {
                    "_ProfileItems": "profile_init_empty",
                    "_UpdateOutput": "profile_update_empty",
                }[response_format.__name__]
            )
        finally:
            active -= 1

    clusters = _SignallingClusterStore("u_bob")
    deps = make_deps(
        llm=FakeLLMClient(handler=answer),
        embedding=LocalEmbedding(),
        stores=make_stores(cluster=clusters),
    )
    alice = [
        _event(episode_entry_id=f"ep_20260101_{i + 1:04d}", memcell_id=f"mc_{i:012x}")
        for i in range(20)
    ]
    for event in alice:
        _seed(
            deps,
            _scope(),
            entry_id=event.episode_entry_id,
            memcell_id=event.memcell_id,
        )
    bob = _event(
        owner_id="u_bob",
        episode_entry_id="ep_20260101_0099",
        memcell_id="mc_bob",
        owner_name="Bob",
    )
    _seed(
        deps,
        _scope("u_bob"),
        entry_id=bob.episode_entry_id,
        memcell_id=bob.memcell_id,
        narrative="Bob plans a trip to Kyoto.",
    )
    config = OMEConfig(jobstore_path=tmp_path / "ome.db", config_watch=False)
    assert config.max_concurrent_runs == 20
    engine = OfflineEngine(config=config)
    engine.register(to_ome_strategy(EXTRACT_USER_PROFILE_SPEC, deps))
    engine.register(to_ome_strategy(TRIGGER_PROFILE_CLUSTERING_SPEC, deps))
    LOCKS.set_acquire_hook(observe)
    await engine.start()
    try:
        async with asyncio.timeout(15):
            for event in alice:
                await engine.emit(event)
            await all_arrived.wait()
            assert calls == 1
            running = await engine.list_runs(
                "extract_user_profile", status=RunStatus.RUNNING
            )
            assert len(running) == 20

            await engine.emit(bob)
            # Alice's model call still holds her partition; the second call
            # can only be Bob's profile.
            await clusters.stored.wait()
            await second_call.wait()
            (bob_cluster,) = await clusters.list_for_owner(
                _scope("u_bob"), "user_memory"
            )
            assert bob.episode_entry_id in bob_cluster.members
            assert not release_model.is_set()

            release_model.set()
            assert await engine.wait_idle(timeout=5)
            profile_runs = await engine.list_runs("extract_user_profile")
            cluster_runs = await engine.list_runs("trigger_profile_clustering")
            assert (len(profile_runs), len(cluster_runs)) == (21, 21)
            assert all(
                r.status == RunStatus.SUCCESS for r in profile_runs + cluster_runs
            )
            assert calls == 21 and peak == 2
            assert len(deps.stores.profile.writes) == 21
            assert await deps.stores.profile.read(_scope()) is not None
    finally:
        release_model.set()
        LOCKS.set_acquire_hook(None)
        await engine.stop()
