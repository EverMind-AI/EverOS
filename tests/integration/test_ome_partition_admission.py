"""Real profile/cluster strategies through OME, with local model and store ports."""

from __future__ import annotations

import asyncio
from pathlib import Path

from everalgo.testing import FakeLLMClient

from everos.core.locks import LOCKS
from everos.infra.ome import OfflineEngine, OMEConfig, RunStatus, to_ome_strategy
from everos.memory.models import Cluster, Profile
from everos.memory.ports import MemoryRef, Scope
from everos.memory.strategies import (
    EXTRACT_USER_PROFILE_SPEC,
    TRIGGER_PROFILE_CLUSTERING_SPEC,
)
from tests.fakes import InMemoryProfileStore, make_deps, make_stores, seed_sources
from tests.fakes.llm import user_track_script
from tests.unit.test_memory.test_strategies.test_extract_user_profile import (
    _event,
    _memcell,
    _scope,
    _seed,
)
from tests.unit.test_memory.test_strategies.test_trigger_profile_clustering import (
    _event as cluster_event,
)


class LocalEmbedding:
    async def embed(self, text: str) -> list[float]:
        return [1.0] + [0.0] * 7


async def test_other_owner_finishes_while_twenty_profile_attempts_are_blocked(
    tmp_path: Path,
) -> None:
    """The model gate stays closed until Bob's cluster is stored and SUCCESS."""
    release_model = asyncio.Event()
    bob_finished = asyncio.Event()
    all_arrived = asyncio.Event()
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
        active += 1
        peak = max(peak, active)
        try:
            if "u_bob" not in messages[-1].content:
                await release_model.wait()
            return user_track_script(
                {
                    "_ProfileItems": "profile_init_empty",
                    "_MemCellUpdateOutput": "profile_update_empty",
                }[response_format.__name__]
            )
        finally:
            active -= 1

    class Profiles(InMemoryProfileStore):
        async def write(self, scope: Scope, profile: Profile) -> MemoryRef:
            ref = await super().write(scope, profile)
            if scope.owner_id == "u_bob":
                bob_finished.set()
            return ref

    deps = make_deps(
        llm=FakeLLMClient(handler=answer),
        embedding=LocalEmbedding(),
        stores=make_stores(profile=Profiles()),
    )
    entries = [f"ep_20260101_{i + 1:04d}" for i in range(20)]
    cells = [f"mc_{i + 1:012x}" for i in range(20)]
    for i, (entry, cell) in enumerate(zip(entries, cells, strict=True)):
        _seed(
            deps,
            _scope(),
            cluster_id="cl_user00000001",
            entry_id=entry,
            memcell_id=cell,
            last_ts=1_700_000_001_000 + i,
        )
    deps.stores.cluster.seed(
        _scope(),
        "user_memory",
        Cluster(
            id="cl_user00000001",
            centroid=(0.0,) * 8,
            count=20,
            last_ts=1_700_000_001_019,
            members=entries,
        ),
    )
    seed_sources(
        deps,
        memcell_id="mc_bbbbbbbbbbb1",
        episode_entry_id="ep_20260101_0002",
        owner_id="u_bob",
    )
    deps.stores.memcell.seed(
        "mc_bbbbbbbbbbb1",
        _memcell("mc_bbbbbbbbbbb1", sender_id="u_bob", ts_ms=1_700_000_001_000),
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
            for cell in cells:
                await engine.emit(_event(memcell_id=cell))
            await all_arrived.wait()
            assert calls == 1
            assert (
                len(
                    await engine.list_runs(
                        "extract_user_profile",
                        status=RunStatus.RUNNING,
                    )
                )
                == 20
            )
            bob = cluster_event(
                owner_id="u_bob",
                memcell_id="mc_bbbbbbbbbbb1",
                episode_entry_id="ep_20260101_0002",
            )
            await engine.emit(bob)
            # wait_for_event observes terminal run records, not just invocation.
            await engine.wait_for_event(bob.event_id, timeout=2)
            records = await engine.list_runs_by_event_id(bob.event_id)
            assert len(records) == 1 and records[0].status == RunStatus.SUCCESS
            clusters = await deps.stores.cluster.list_for_owner(
                _scope("u_bob"), "user_memory"
            )
            assert len(clusters) == 1
            assert "ep_20260101_0002" in clusters[0].members
            await asyncio.wait_for(bob_finished.wait(), timeout=2)
            assert await deps.stores.profile.read(_scope("u_bob")) is not None
            assert not release_model.is_set()
            assert calls == 2
            release_model.set()
            assert await engine.wait_idle(timeout=5)
            records = await engine.list_runs(
                "extract_user_profile"
            ) + await engine.list_runs("trigger_profile_clustering")
            assert len(records) == 22
            assert all(r.status == RunStatus.SUCCESS for r in records)
            assert calls == 21 and peak == 2
            assert len(deps.stores.profile.writes) == 21
            assert await deps.stores.profile.read(_scope()) is not None
    finally:
        release_model.set()
        LOCKS.set_acquire_hook(None)
        await engine.stop()
