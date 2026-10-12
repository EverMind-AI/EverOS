"""``CascadeOrchestrator`` — idempotent start/stop, queue_summary forwards."""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import patch

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
    """Calling start twice does not relaunch tasks.

    The watcher is stubbed because a real one takes an inotify watch, a per-user
    kernel resource this host runs out of (the IDE's indexer holds 350k of the
    524k ceiling). Idempotency is the subject here; the observer thread is not.
    """
    with patch("everos.memory.cascade.orchestrator.CascadeWatcher") as watcher_cls:
        orch = _make_orchestrator(runtime)
        await orch.start()
        # Capture watcher identity to verify the second start doesn't replace it.
        first_watcher = orch._watcher
        await orch.start()
        assert orch._watcher is first_watcher
        assert watcher_cls.call_count == 1
        await orch.stop()


async def test_start_settles_the_reconcile_marker(runtime: MemoryRoot) -> None:
    """The server path: the first ``start()`` on a root runs the one-time
    reconcile and records it, so the next start does not queue the ledger."""
    from everos.memory.cascade.orchestrator import reconcile_index_once

    with patch("everos.memory.cascade.orchestrator.CascadeWatcher"):
        orch = _make_orchestrator(runtime)
        await orch.start()
        try:
            marker = runtime.index_dir / ".index_reconcile_version"
            assert marker.read_text(encoding="utf-8") == "1"
            assert await reconcile_index_once(runtime) == []
        finally:
            await orch.stop()


async def test_stop_before_start_is_noop(runtime: MemoryRoot) -> None:
    orch = _make_orchestrator(runtime)
    await orch.stop()  # must not raise; nothing to do


async def test_double_stop_is_idempotent(runtime: MemoryRoot) -> None:
    with patch("everos.memory.cascade.orchestrator.CascadeWatcher"):
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


async def test_forced_reconcile_settles_the_marker(runtime: MemoryRoot) -> None:
    """``cascade sync --force-all`` runs the reconcile pass on demand and
    records it, so the unscoped ``sync_once`` right after it does not queue
    the whole ledger a second time."""
    from everos.infra.persistence.sqlite import md_change_state_repo
    from everos.memory.cascade.orchestrator import reconcile_index_once

    path = "users/u1/episodes/ep_1.md"
    orch = _make_orchestrator(runtime)
    await md_change_state_repo.upsert(
        path, kind="episode", change_type="added", mtime=1.0
    )
    await md_change_state_repo.claim_pending_batch(10)
    await md_change_state_repo.mark_done(path)

    forced = await orch.force_enqueue_all()

    assert [r.md_path for r in forced] == [path]
    row = await md_change_state_repo.get_by_id(path)
    assert row is not None and row.status == "pending" and row.mtime == 1.0
    assert (runtime.index_dir / ".index_reconcile_version").read_text() == "1"
    assert await reconcile_index_once(runtime) == []


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
    monkeypatch.setenv("EVEROS_CASCADE__SCAN_INTERVAL_SECONDS", "55")
    load_settings.cache_clear()  # type: ignore[attr-defined]
    try:
        cfg = CascadeConfig.from_settings()
        assert (
            cfg.optimize_heartbeat_seconds,
            cfg.optimize_prune_interval_seconds,
            cfg.optimize_prune_retention_seconds,
            cfg.optimize_rebuild_interval_seconds,
            cfg.scan_interval_seconds,
        ) == (11.0, 22.0, 33.0, 44.0, 55.0)

        orch = CascadeOrchestrator(
            memory_root=MemoryRoot.resolve(), tokenizer=build_tokenizer(), config=cfg
        )
        worker = orch._worker
        assert worker._optimize_heartbeat == 11.0
        assert worker._optimize_prune_interval == 22.0
        assert worker._optimize_prune_retention == 33.0
        assert worker._optimize_rebuild_interval == 44.0
        # The scanner is the other consumer; on an event-less mount it is the
        # only one, so it must see the value too, not just CascadeConfig.
        assert orch._scanner._interval == 55.0
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
        "scan_interval_seconds",
    }
    assert not any("timeout" in f or "deadline" in f for f in exposed)


async def test_watcher_can_be_disabled_without_losing_the_worker(
    runtime: MemoryRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``EVEROS_DISABLE_CASCADE_WATCHER`` drops inotify, keeps md -> LanceDB.

    Forced by a shared host: inotify watches are a per-user kernel resource, the
    ceiling lives in read-only ``/proc/sys``, and an IDE indexer can hold 350k of
    524k on its own -- every server then dies at startup with ``Errno 28``. The
    scanner re-derives the same truth every 30s, so what is lost is latency. The
    worker must survive, or an ingesting run writes md that never gets indexed.
    """
    monkeypatch.setenv("EVEROS_DISABLE_CASCADE_WATCHER", "1")
    with patch("everos.memory.cascade.orchestrator.CascadeWatcher") as watcher_cls:
        orch = _make_orchestrator(runtime)
        await orch.start()
        assert watcher_cls.call_count == 0, "watcher was constructed anyway"
        assert orch._watcher is None
        assert orch._started is True
        assert orch._worker is not None
        assert orch._scanner is not None
        await orch.stop()


class _SharedQueue:
    """A ledger two drainers claim from: this orchestrator's passes and, inside
    a server, the background worker. ``held`` is the batch the worker has
    claimed (``processing``), which ``pending`` counts too."""

    def __init__(self, rows: int, bound: int = 5000) -> None:
        self.rows, self.bound, self.held = rows, bound, 0
        self.drains: list[int] = []
        self.summaries: list[int] = []

    async def drain(self, **_: object) -> int:
        took = min(self.rows, self.bound)
        self.rows -= took
        self.drains.append(took)
        return took

    async def summary(self) -> object:
        pending = self.rows + self.held
        self.summaries.append(pending)
        return type("S", (), {"pending": pending})()


def _patched(orch: CascadeOrchestrator, q: _SharedQueue) -> tuple:
    return (
        patch.object(orch, "sync_once", side_effect=q.drain),
        patch.object(orch, "drain_once", side_effect=q.drain),
        patch.object(orch, "queue_summary", side_effect=q.summary),
    )


async def test_drain_fully_keeps_draining_past_the_worker_bound(
    runtime: MemoryRoot,
) -> None:
    """The worker bounds one drain (``max_passes`` x batch size). The ledger
    the one-time reconcile queues after an upgrade can exceed that bound, and
    a caller that stops there reports success with rows still pending.
    """
    orch = _make_orchestrator(runtime)
    q = _SharedQueue(rows=5003)
    a, b, c = _patched(orch, q)
    with a, b, c:
        assert await orch.drain_fully() == 5003
    assert q.rows == 0
    assert q.drains == [
        5000,
        3,
    ]  # the scan's drain, then the rest; the ledger says when to stop


async def test_drain_fully_waits_for_a_batch_another_drainer_holds(
    runtime: MemoryRoot,
) -> None:
    """Inside a server the background worker claims batches too. A pass of
    ours can find nothing pending while the worker still holds the last
    batch; ``POST /cascade/quiesce`` cancels the worker right after, which
    would orphan that batch in ``processing`` and freeze a partial index
    (real host, 12000-row ledger: 50 rows left behind). So the ledger, not
    our own pass, says when the drain is done.
    """
    orch = _make_orchestrator(runtime)
    q = _SharedQueue(rows=5050)
    a, b, c = _patched(orch, q)
    with a, b, c:

        async def first_pass(**_: object) -> int:
            took = await q.drain()  # 5000 by us; the worker claims the last 50
            q.rows, q.held = 0, 50
            return took

        async def worker_finishes_later(**_: object) -> int:
            # our pass finds nothing pending; the worker's batch lands two polls later
            if len(q.drains) >= 3:
                q.held = 0
            q.drains.append(0)
            return 0

        orch.sync_once.side_effect = first_pass
        orch.drain_once.side_effect = worker_finishes_later
        orch._config = dataclasses.replace(
            orch._config, worker_poll_interval_seconds=0.01
        )
        processed = await orch.drain_fully()

    assert q.summaries and q.summaries[-1] == 0, (
        "returned while the worker held a batch"
    )
    assert processed == 5000


async def test_drain_fully_gives_up_when_the_ledger_stops_moving(
    runtime: MemoryRoot,
) -> None:
    """A row stuck in ``processing`` (a hung handler) must not hang the caller
    forever; after the idle limit the drain returns and the caller reports
    what is left."""
    orch = _make_orchestrator(runtime)
    q = _SharedQueue(rows=0)
    q.held = 1
    a, b, c = _patched(orch, q)
    with (
        a,
        b,
        c,
        patch("everos.memory.cascade.orchestrator._DRAIN_IDLE_LIMIT_SECONDS", 0.05),
    ):
        orch._config = dataclasses.replace(
            orch._config, worker_poll_interval_seconds=0.01
        )
        assert await orch.drain_fully() == 0
    assert q.summaries[-1] == 1  # gave up with the row still held, and said so
