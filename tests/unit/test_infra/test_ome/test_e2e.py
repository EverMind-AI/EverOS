"""End-to-end pipeline test exercising the chain emit semantics.

MemCellSaved -> atomic (leaf strategy)
EpisodeSaved -> cluster -> ClusteringCompleted -> profile (Counter threshold=3)
"""

from __future__ import annotations

import asyncio

import pytest

from everos.infra.ome import (
    BaseEvent,
    Counter,
    Cron,
    CronTick,
    Immediate,
    StrategyContext,
    offline_strategy,
)
from everos.infra.ome.engine import _cron_entry
from everos.infra.ome.testing import StrategyTestHarness


class MemCellSaved(BaseEvent):
    user_id: str
    cell_id: str


class EpisodeSaved(BaseEvent):
    user_id: str
    episode_text: str


class ClusteringCompleted(BaseEvent):
    user_id: str


@pytest.mark.asyncio
async def test_chain_emit_without_counter_gate() -> None:
    """Variant of the full-chain test without a Counter gate.

    Profile fires once per ClusteringCompleted instead of once per N.
    """
    log: list[tuple[str, str]] = []

    @offline_strategy(
        name="cluster_e2e",
        trigger=Immediate(on=[EpisodeSaved]),
        emits=[ClusteringCompleted],
    )
    async def cluster(event: EpisodeSaved, ctx: StrategyContext) -> None:
        log.append(("cluster", event.user_id))
        await ctx.emit(ClusteringCompleted(user_id=event.user_id))

    @offline_strategy(
        name="profile_e2e",
        trigger=Immediate(on=[ClusteringCompleted]),
        emits=[],
    )
    async def profile(event: ClusteringCompleted, ctx: StrategyContext) -> None:
        log.append(("profile", event.user_id))

    async with StrategyTestHarness() as h:
        h.register(cluster)
        h.register(profile)
        await h.start()
        # Emit 3 episodes -> cluster runs 3x -> emits ClusteringCompleted 3x ->
        # profile runs 3x (no counter gate).
        await h.emit(EpisodeSaved(user_id="u1", episode_text="t1"))
        await asyncio.sleep(0.15)
        await h.emit(EpisodeSaved(user_id="u1", episode_text="t2"))
        await asyncio.sleep(0.15)
        await h.emit(EpisodeSaved(user_id="u1", episode_text="t3"))
        await asyncio.sleep(0.2)
        await h.drain(timeout=15)

        cluster_runs = await h.list_runs("cluster_e2e")
        profile_runs = await h.list_runs("profile_e2e")

    cluster_calls = [c for c in log if c[0] == "cluster"]
    profile_calls = [c for c in log if c[0] == "profile"]
    assert len(cluster_calls) == 3, (
        f"Expected 3 cluster, got {len(cluster_calls)}: {log}"
    )
    assert len(profile_calls) == 3, (
        f"Expected 3 profile, got {len(profile_calls)}: {log}"
    )
    assert len(cluster_runs) == 3
    assert len(profile_runs) == 3


@pytest.mark.asyncio
async def test_chain_pipeline_runs_full_path() -> None:
    """Full chain with atomic, cluster, and profile (Counter gated)."""
    log: list[tuple[str, str]] = []

    @offline_strategy(name="atomic_e2e", trigger=Immediate(on=[MemCellSaved]), emits=[])
    async def atomic(event: MemCellSaved, ctx: StrategyContext) -> None:
        log.append(("atomic", event.cell_id))

    @offline_strategy(
        name="cluster_e2e",
        trigger=Immediate(on=[EpisodeSaved]),
        emits=[ClusteringCompleted],
    )
    async def cluster(event: EpisodeSaved, ctx: StrategyContext) -> None:
        log.append(("cluster", event.user_id))
        await ctx.emit(ClusteringCompleted(user_id=event.user_id))

    @offline_strategy(
        name="profile_e2e",
        trigger=Immediate(on=[ClusteringCompleted]),
        emits=[],
        gate=Counter(threshold=3, event_field="user_id"),
    )
    async def profile(event: ClusteringCompleted, ctx: StrategyContext) -> None:
        log.append(("profile", event.user_id))

    async with StrategyTestHarness() as h:
        h.register(atomic)
        h.register(cluster)
        h.register(profile)
        await h.start()
        # Two memcells (each fires atomic).
        await h.emit(MemCellSaved(user_id="u1", cell_id="c1"))
        await asyncio.sleep(0.15)
        await h.emit(MemCellSaved(user_id="u1", cell_id="c2"))
        await asyncio.sleep(0.15)
        # Three episodes -> cluster runs 3x -> ClusteringCompleted 3x ->
        # profile Counter at threshold=3 fires once.
        await h.emit(EpisodeSaved(user_id="u1", episode_text="t1"))
        await asyncio.sleep(0.15)
        await h.emit(EpisodeSaved(user_id="u1", episode_text="t2"))
        await asyncio.sleep(0.15)
        await h.emit(EpisodeSaved(user_id="u1", episode_text="t3"))
        await asyncio.sleep(0.2)
        await h.drain(timeout=15)

        # Validate using run records
        atomic_runs = await h.list_runs("atomic_e2e")
        cluster_runs = await h.list_runs("cluster_e2e")
        profile_runs = await h.list_runs("profile_e2e")

    atomic_calls = [c for c in log if c[0] == "atomic"]
    cluster_calls = [c for c in log if c[0] == "cluster"]
    profile_calls = [c for c in log if c[0] == "profile"]
    assert len(atomic_calls) == 2, (
        f"Expected 2 atomic calls, got {len(atomic_calls)}: {log}"
    )
    assert len(cluster_calls) == 3, (
        f"Expected 3 cluster calls, got {len(cluster_calls)}: {log}"
    )
    assert len(profile_calls) == 1, (
        f"Expected 1 profile call, got {len(profile_calls)}: {log}"
    )
    assert len(atomic_runs) == 2
    assert len(cluster_runs) == 3
    assert len(profile_runs) == 1


@pytest.mark.asyncio
async def test_cron_strategy_executes_when_cron_entry_fires() -> None:
    """Verify that the cron-trigger code path actually reaches the strategy.

    APScheduler timing is mocked away — we directly call the module-level
    _cron_entry function that APS would invoke on schedule. This proves
    the registry/dispatcher/runner chain wires cron strategies correctly.
    """
    seen: list[str] = []

    @offline_strategy(name="cron_e2e", trigger=Cron(expr="0 * * * *"), emits=[])
    async def cron_job(event: CronTick, ctx: StrategyContext) -> None:
        seen.append(event.strategy_name)

    async with StrategyTestHarness() as h:
        h.register(cron_job)
        await h.start()
        # Directly invoke what APS would call; bypass scheduler timing.
        await _cron_entry(h._engine._engine_id, "cron_e2e")  # noqa: SLF001
        await h.drain(timeout=5)
        runs = await h.list_runs("cron_e2e")

    assert seen == ["cron_e2e"]
    assert len(runs) == 1
