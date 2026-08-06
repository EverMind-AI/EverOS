"""``CascadeOrchestrator`` — idempotent start/stop, queue_summary forwards."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlmodel import SQLModel

from everos.component.tokenizer import build_tokenizer
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb import (
    dispose_connection,
    ensure_business_indexes,
)
from everos.infra.persistence.sqlite import dispose_engine, get_engine
from everos.memory.cascade import CascadeConfig, CascadeOrchestrator


@pytest.fixture
async def runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[MemoryRoot]:
    """Boot sqlite + lancedb against a tmp memory_root."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "stub-model")
    monkeypatch.setenv("EVEROS_EMBEDDING__BASE_URL", "http://stub.invalid/v1")
    monkeypatch.setenv("EVEROS_EMBEDDING__API_KEY", "stub-key")

    await dispose_connection()
    await dispose_engine()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    await ensure_business_indexes()
    yield MemoryRoot.resolve()
    await dispose_connection()
    await dispose_engine()


def _make_orchestrator(memory_root: MemoryRoot) -> CascadeOrchestrator:
    return CascadeOrchestrator(
        memory_root=memory_root,
        tokenizer=build_tokenizer(),
        config=CascadeConfig(
            scan_interval_seconds=60.0,
            worker_batch_size=10,
            worker_max_retry=1,
            worker_poll_interval_seconds=0.05,
            worker_retry_backoff_seconds=0.0,
        ),
    )


async def test_double_start_is_idempotent(runtime: MemoryRoot) -> None:
    """Calling start twice does not relaunch tasks."""
    orch = _make_orchestrator(runtime)
    await orch.start()
    # Capture watcher identity to verify the second start doesn't replace it.
    first_watcher = orch._watcher
    await orch.start()
    assert orch._watcher is first_watcher
    await orch.stop()


async def test_stop_before_start_is_noop(runtime: MemoryRoot) -> None:
    orch = _make_orchestrator(runtime)
    await orch.stop()  # must not raise; nothing to do


async def test_double_stop_is_idempotent(runtime: MemoryRoot) -> None:
    orch = _make_orchestrator(runtime)
    await orch.start()
    await orch.stop()
    await orch.stop()  # second stop is a no-op


async def test_queue_summary_returns_empty_on_fresh_runtime(
    runtime: MemoryRoot,
) -> None:
    orch = _make_orchestrator(runtime)
    summary = await orch.queue_summary()
    assert summary.pending == 0
    assert summary.done == 0
    assert summary.failed_retryable == 0
    assert summary.failed_permanent == 0


async def test_drain_once_returns_zero_on_empty_queue(
    runtime: MemoryRoot,
) -> None:
    orch = _make_orchestrator(runtime)
    assert await orch.drain_once() == 0


async def test_health_is_healthy_on_fresh_runtime(runtime: MemoryRoot) -> None:
    """A quiet, freshly-booted cascade reports healthy with no reasons."""
    orch = _make_orchestrator(runtime)
    health = await orch.health()
    assert health.healthy is True
    assert health.reasons == []
    assert health.failed_permanent == 0
    assert health.prune_stale_seconds == 0.0


async def test_permanent_failures_are_informational_not_unhealthy(
    runtime: MemoryRoot,
) -> None:
    """A permanently-failed md row is reported but must NOT flip ``healthy``.

    A per-file triage backlog is normal steady state; folding it into the
    verdict would pin the signal red forever. ``failed_permanent`` is
    surfaced as an informational count while ``healthy`` stays true so
    long as the pipeline itself (drain / optimize / prune) is fine.
    """
    from everos.component.utils.datetime import get_utc_now
    from everos.infra.persistence.sqlite import md_change_state_repo

    orch = _make_orchestrator(runtime)
    await md_change_state_repo.upsert(
        "users/u1/episodes/ep_1.md",
        kind="episode",
        change_type="added",
        mtime=get_utc_now().timestamp(),
    )
    await md_change_state_repo.claim_pending_batch(10)
    await md_change_state_repo.mark_failed(
        "users/u1/episodes/ep_1.md",
        retryable=False,
        error="boom",
        new_retry_count=0,
    )

    health = await orch.health()
    assert health.failed_permanent == 1  # reported…
    assert health.healthy is True  # …but pipeline is operationally fine
    assert health.reasons == []


async def test_operational_signal_flips_healthy(
    runtime: MemoryRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operational reason (prune stalled) — not a data-quality backlog —
    is what flips ``healthy`` false."""
    import everos.memory.cascade.worker as wmod

    orch = _make_orchestrator(runtime)
    # Freeze the monotonic clock so staleness is deterministic regardless of
    # the runner's boot uptime (see test_worker prune-staleness test).
    now = 10_000.0
    monkeypatch.setattr(wmod.time, "monotonic", lambda: now)
    # simulate a running worker whose version cleanup has gone stale
    orch._worker._started_at = now - (wmod._PRUNE_STALE_SECONDS_ALERT + 100)
    orch._worker._optimizer_states["episode"] = wmod._KindOptimizerState()

    health = await orch.health()
    assert health.healthy is False
    assert any("cleanup stalled" in r for r in health.reasons)


async def test_maintenance_cadences_reach_the_worker_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``[cascade]`` settings must actually land on the worker.

    These four were constructor-only for long enough that the 12h rebuild sweep
    could not be exercised by any soak run shorter than half a day — the gap was
    not a missing parameter but a config layer that dropped it. Asserting on the
    worker's own attributes rather than on ``CascadeConfig`` is the point: a
    field that stops being forwarded still passes a config-level check.
    """
    from everos.config import load_settings
    from everos.memory.cascade.orchestrator import CascadeConfig, CascadeOrchestrator

    monkeypatch.setenv("EVEROS_CASCADE__OPTIMIZE_HEARTBEAT_SECONDS", "11")
    monkeypatch.setenv("EVEROS_CASCADE__OPTIMIZE_PRUNE_INTERVAL_SECONDS", "22")
    monkeypatch.setenv("EVEROS_CASCADE__OPTIMIZE_PRUNE_RETENTION_SECONDS", "33")
    monkeypatch.setenv("EVEROS_CASCADE__OPTIMIZE_REBUILD_INTERVAL_SECONDS", "44")
    load_settings.cache_clear()  # type: ignore[attr-defined]
    try:
        cfg = CascadeConfig.from_settings()
        assert (
            cfg.optimize_heartbeat_seconds,
            cfg.optimize_prune_interval_seconds,
            cfg.optimize_prune_retention_seconds,
            cfg.optimize_rebuild_interval_seconds,
        ) == (11.0, 22.0, 33.0, 44.0)

        orch = CascadeOrchestrator(
            memory_root=MemoryRoot.resolve(), tokenizer=build_tokenizer(), config=cfg
        )
        worker = orch._worker
        assert worker._optimize_heartbeat == 11.0
        assert worker._optimize_prune_interval == 22.0
        assert worker._optimize_prune_retention == 33.0
        assert worker._optimize_rebuild_interval == 44.0
    finally:
        load_settings.cache_clear()  # type: ignore[attr-defined]


def test_deadlines_are_deliberately_not_configurable() -> None:
    """The hang-catchers must stay constants, not settings.

    A read/write/prune deadline is sized from a measured duration: too low
    manufactures failures on a healthy table, too high leaves a wedged one
    invisible for longer. Neither end is a tuning preference, so exposing them
    invites a change that can only make things worse. Cadences are the opposite
    — they depend on write volume — which is why only those moved.
    """
    from everos.config import CascadeSettings

    exposed = set(CascadeSettings.model_fields)
    assert exposed == {
        "optimize_heartbeat_seconds",
        "optimize_prune_interval_seconds",
        "optimize_prune_retention_seconds",
        "optimize_rebuild_interval_seconds",
    }
    assert not any("timeout" in f or "deadline" in f for f in exposed)
