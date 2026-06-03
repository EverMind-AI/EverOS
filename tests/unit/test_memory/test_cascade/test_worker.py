"""Tests for :class:`CascadeWorker` retry classification + optimize scheduler.

The pure-function pieces (registry / reconciler) get coverage in
their own files. Here we focus on the worker's branch behaviour
without touching the real handler / lancedb stack:

- ``RecoverableError`` retries up to ``max_retry`` and then marks
  ``retryable=TRUE``.
- Any other exception marks ``retryable=FALSE`` immediately.
- Successful handler ⇒ ``mark_done``.
- Unknown kind ⇒ ``mark_failed(retryable=False)``.

A second group covers the per-kind throttle + trailing-edge
optimize scheduler that fires LanceDB ``optimize()`` outside the
drain loop — coalescing under burst writes, re-running when dirty
is re-raised mid-optimize, and flushing on drain-until-empty / stop.

The repo singleton is monkey-patched onto a recording fake so the
test stays in-memory.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
import unittest.mock as mock
from dataclasses import dataclass

import pytest

from everos.memory.cascade.errors import RecoverableError, UnrecoverableError
from everos.memory.cascade.handlers import Handler, HandlerDeps
from everos.memory.cascade.types import HandlerOutcome
from everos.memory.cascade.worker import CascadeWorker


@dataclass
class _Row:
    """Minimal MdChangeState shape the worker reads off."""

    md_path: str
    kind: str = "episode"
    change_type: str = "added"
    retry_count: int = 0


class _FakeRepo:
    """Records every state-machine transition the worker drives."""

    def __init__(self, batch: list[_Row]) -> None:
        self.batch = list(batch)
        self.done: list[str] = []
        self.failed: list[tuple[str, bool, str, int]] = []

    async def claim_pending_batch(self, _limit: int) -> list[_Row]:
        items, self.batch = self.batch, []
        return items

    async def mark_done(self, md_path: str) -> None:
        self.done.append(md_path)

    async def mark_failed(
        self,
        md_path: str,
        *,
        retryable: bool,
        error: str,
        new_retry_count: int,
    ) -> None:
        self.failed.append((md_path, retryable, error, new_retry_count))


class _OkHandler(Handler):
    def __init__(self) -> None:  # noqa: D401 — no deps needed
        pass

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=1, deleted=0, skipped=0
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=0, deleted=1, skipped=0
        )


class _RecoverableHandler(_OkHandler):
    """Always raises RecoverableError."""

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        raise RecoverableError("embedding 503")


class _UnrecoverableHandler(_OkHandler):
    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        raise UnrecoverableError("YAML parse error")


class _BareExceptionHandler(_OkHandler):
    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        raise RuntimeError("unexpected boom")


@pytest.fixture
def patched_repo(monkeypatch: pytest.MonkeyPatch) -> _FakeRepo:
    """Drop a fake repo onto the module the worker imports."""
    from everos.memory.cascade import worker as worker_mod

    repo = _FakeRepo(batch=[])
    monkeypatch.setattr(worker_mod, "md_change_state_repo", repo)
    return repo


async def test_ok_handler_marks_done(patched_repo: _FakeRepo) -> None:
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
    await w.drain_once()
    assert patched_repo.done == ["a.md"]
    assert patched_repo.failed == []


async def test_recoverable_handler_marks_retryable_after_max_retry(
    patched_repo: _FakeRepo,
) -> None:
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker(
        {"episode": _RecoverableHandler()}, max_retry=2, retry_backoff_seconds=0
    )
    await w.drain_once()
    assert patched_repo.done == []
    assert len(patched_repo.failed) == 1
    path, retryable, _err, retry_count = patched_repo.failed[0]
    assert path == "a.md"
    assert retryable is True
    assert retry_count == 2  # 2 retries after the initial attempt


async def test_unrecoverable_handler_marks_permanent(
    patched_repo: _FakeRepo,
) -> None:
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker({"episode": _UnrecoverableHandler()}, retry_backoff_seconds=0)
    await w.drain_once()
    _path, retryable, err, _retry = patched_repo.failed[0]
    assert retryable is False
    assert "UnrecoverableError" in err or "YAML parse error" in err


async def test_bare_exception_marked_permanent(patched_repo: _FakeRepo) -> None:
    """Anything that isn't RecoverableError counts as unrecoverable."""
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker({"episode": _BareExceptionHandler()}, retry_backoff_seconds=0)
    await w.drain_once()
    _path, retryable, _err, _retry = patched_repo.failed[0]
    assert retryable is False


async def test_unknown_kind_marks_permanent_without_handler(
    patched_repo: _FakeRepo,
) -> None:
    patched_repo.batch = [_Row(md_path="a.md", kind="mystery")]
    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
    await w.drain_once()
    assert patched_repo.failed[0][1] is False
    assert "no handler" in patched_repo.failed[0][2]


async def test_drain_until_empty_loops_until_no_batch(
    patched_repo: _FakeRepo,
) -> None:
    """Worker keeps draining until claim returns an empty list."""

    rows = [_Row(md_path=f"a{i}.md") for i in range(3)]

    class _ChunkedRepo(_FakeRepo):
        async def claim_pending_batch(self, _limit: int) -> list[_Row]:
            if not self.batch:
                return []
            head, self.batch = self.batch[:1], self.batch[1:]
            return head

    chunked = _ChunkedRepo(rows)
    from everos.memory.cascade import worker as worker_mod

    with mock.patch.object(worker_mod, "md_change_state_repo", chunked):
        w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
        total = await w.drain_until_empty()
    assert total == 3
    assert len(chunked.done) == 3


def test_worker_handler_deps_construct_with_real_classes() -> None:
    """Sanity: HandlerDeps accepts the real provider Protocols."""
    # No instantiation needed — just verifies the dataclass shape.
    assert {"memory_root", "embedder", "tokenizer"} == {
        f.name for f in HandlerDeps.__dataclass_fields__.values()
    }


# ── Optimize scheduler tests ───────────────────────────────────────────────


class _FakeLanceRepo:
    """Records every optimize() / rebuild_indexes() call.

    ``optimize_delay`` / ``rebuild_delay`` simulate slow operations.
    ``rebuild_raises`` makes ``rebuild_indexes`` raise (for crash-safety tests).
    Each ``optimize`` call's ``cleanup_older_than`` is preserved so
    prune-cadence tests can assert which calls took the heavy path.
    """

    def __init__(
        self,
        *,
        optimize_delay: float = 0.0,
        rebuild_delay: float = 0.0,
        rebuild_raises: bool = False,
    ) -> None:
        self.optimize_calls: list[float] = []
        self.optimize_cleanup_args: list[dt.timedelta | None] = []
        self.rebuild_calls: list[float] = []
        self.optimize_delay = optimize_delay
        self.rebuild_delay = rebuild_delay
        self.rebuild_raises = rebuild_raises

    async def optimize(self, *, cleanup_older_than: dt.timedelta | None = None) -> None:
        if self.optimize_delay > 0:
            await asyncio.sleep(self.optimize_delay)
        self.optimize_calls.append(time.monotonic())
        self.optimize_cleanup_args.append(cleanup_older_than)

    async def rebuild_indexes(self) -> None:
        if self.rebuild_delay > 0:
            await asyncio.sleep(self.rebuild_delay)
        if self.rebuild_raises:
            raise RuntimeError("rebuild boom")
        self.rebuild_calls.append(time.monotonic())


class _OkHandlerWithRepo(_OkHandler):
    """OK handler exposing a fake ``lance_repo`` for scheduler tests."""

    def __init__(self, repo: _FakeLanceRepo) -> None:
        super().__init__()
        self.lance_repo = repo


async def test_schedule_optimize_noop_when_handler_has_no_lance_repo(
    patched_repo: _FakeRepo,
) -> None:
    """Test stubs without ``lance_repo`` should not even register state."""
    w = CascadeWorker(
        {"episode": _OkHandler()},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.05,
    )
    w._schedule_optimize("episode")
    assert "episode" not in w._optimizer_states


async def test_schedule_optimize_collapses_burst_within_throttle_window(
    patched_repo: _FakeRepo,
) -> None:
    """A burst of synchronous schedules creates at most one in-flight task.

    The first call starts the optimize; subsequent calls during the
    same window only flip ``dirty``. With no time advance between
    schedules, the runner sees ``dirty=False`` after the first run
    and exits — total optimize() calls collapse to one.
    """
    fake = _FakeLanceRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.05,
    )
    for _ in range(10):
        w._schedule_optimize("episode")
    await w._flush_optimizers()
    assert fake.optimize_calls, "expected at least one optimize"
    assert len(fake.optimize_calls) == 1, (
        f"burst should collapse, got {len(fake.optimize_calls)} calls"
    )


async def test_schedule_optimize_reruns_when_dirty_set_during_optimize(
    patched_repo: _FakeRepo,
) -> None:
    """A write that lands mid-optimize re-raises ``dirty`` and triggers a re-run.

    Uses an artificially slow optimize so the second schedule fires
    while the first run is still in flight. Trailing-edge semantics
    guarantee the second run happens after the throttle interval.
    """
    fake = _FakeLanceRepo(optimize_delay=0.05)
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.02,
    )
    w._schedule_optimize("episode")
    await asyncio.sleep(0.01)  # ensure first task is mid-optimize
    w._schedule_optimize("episode")
    await w._flush_optimizers()
    assert len(fake.optimize_calls) == 2


async def test_concurrent_schedules_keep_one_task_per_kind(
    patched_repo: _FakeRepo,
) -> None:
    """LanceDB manifest contention guard: per-kind in-flight task is unique."""
    fake = _FakeLanceRepo(optimize_delay=0.05)
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.02,
    )
    w._schedule_optimize("episode")
    first_task = w._optimizer_states["episode"].task
    # Re-schedule while first task is still in flight; slot must not
    # be replaced.
    for _ in range(5):
        w._schedule_optimize("episode")
        assert w._optimizer_states["episode"].task is first_task
    await w._flush_optimizers()


async def test_flush_optimizers_awaits_pending_task(
    patched_repo: _FakeRepo,
) -> None:
    """flush_optimizers blocks until in-flight optimize commits and clears slot."""
    fake = _FakeLanceRepo(optimize_delay=0.05)
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.02,
    )
    w._schedule_optimize("episode")
    assert w._optimizer_states["episode"].task is not None
    await w._flush_optimizers()
    assert fake.optimize_calls, "flush should not return before optimize ran"
    assert w._optimizer_states["episode"].task is None


async def test_drain_until_empty_flushes_optimizers_before_returning(
    patched_repo: _FakeRepo,
) -> None:
    """CLI ``cascade sync`` expects FTS to be current when the call returns."""
    fake = _FakeLanceRepo(optimize_delay=0.03)
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.02,
    )
    await w.drain_until_empty()
    assert patched_repo.done == ["a.md"]
    assert len(fake.optimize_calls) == 1
    assert w._optimizer_states["episode"].task is None


async def test_drain_once_does_not_block_on_optimize(
    patched_repo: _FakeRepo,
) -> None:
    """drain_once is fire-and-forget — it must return before optimize commits."""
    fake = _FakeLanceRepo(optimize_delay=0.2)
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
    )
    started = time.monotonic()
    await w.drain_once()
    drain_elapsed = time.monotonic() - started
    # drain returned long before the 0.2s optimize would finish
    assert drain_elapsed < 0.1, f"drain blocked on optimize: {drain_elapsed:.3f}s"
    assert not fake.optimize_calls, "optimize should still be in flight"
    await w._flush_optimizers()
    assert len(fake.optimize_calls) == 1


async def test_stop_waits_for_in_flight_optimize(
    patched_repo: _FakeRepo,
) -> None:
    """stop() must give an in-flight optimize a chance to commit cleanly."""
    fake = _FakeLanceRepo(optimize_delay=0.05)
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.02,
        optimize_heartbeat_seconds=10.0,
        # Park rebuild interval — startup sweep still fires but we wait
        # for it before testing optimize semantics.
        optimize_rebuild_interval_seconds=10.0,
    )
    await w.start()
    # Let the startup rebuild sweep complete (instant for the fake repo)
    # before scheduling optimize — otherwise optimize would queue behind it.
    await asyncio.sleep(0.02)
    assert fake.rebuild_calls, "startup rebuild should have fired by now"
    w._schedule_optimize("episode")
    await asyncio.sleep(0.01)  # let optimize start
    await w.stop()
    assert len(fake.optimize_calls) == 1


async def test_optimize_failure_does_not_crash_drain_loop(
    patched_repo: _FakeRepo,
) -> None:
    """Repo.optimize() raising should be logged but never propagate."""

    class _FailingRepo:
        async def optimize(self) -> None:
            raise RuntimeError("simulated lancedb manifest conflict")

    class _HandlerWithFailingRepo(_OkHandler):
        def __init__(self) -> None:
            super().__init__()
            self.lance_repo = _FailingRepo()

    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker(
        {"episode": _HandlerWithFailingRepo()},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.02,
    )
    # If the failure propagated, drain_until_empty would raise.
    await w.drain_until_empty()
    assert patched_repo.done == ["a.md"]
    assert patched_repo.failed == []


async def test_heartbeat_schedules_every_handler_kind(
    patched_repo: _FakeRepo,
) -> None:
    """The heartbeat sweeps all kinds, even ones nobody wrote to.

    Drives the heartbeat manually via a short interval and asserts
    that ``optimize`` ran for both kinds at least once.
    """
    fake_a = _FakeLanceRepo()
    fake_b = _FakeLanceRepo()
    w = CascadeWorker(
        {
            "episode": _OkHandlerWithRepo(fake_a),
            "atomic_fact": _OkHandlerWithRepo(fake_b),
        },
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
        optimize_heartbeat_seconds=0.05,
    )
    await w.start()
    # Let at least one heartbeat tick happen.
    await asyncio.sleep(0.12)
    await w.stop()
    assert fake_a.optimize_calls, "heartbeat should have scheduled episode"
    assert fake_b.optimize_calls, "heartbeat should have scheduled atomic_fact"


async def test_optimize_prunes_on_first_call_then_throttles(
    patched_repo: _FakeRepo,
) -> None:
    """First optimize() per kind passes ``cleanup_older_than``; subsequent
    calls within ``optimize_prune_interval_seconds`` do not.

    Rationale lives in ``DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS``:
    LanceDB ``optimize()`` without ``cleanup_older_than`` leaves stale
    physical files on disk; passing it on every 1-second optimize tick
    is wasteful, but never passing it leaks files until FDs exhaust.
    A separate cadence — prune ≪ optimize — balances the two.
    """
    fake = _FakeLanceRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
        optimize_prune_interval_seconds=10.0,  # long — second call should NOT prune
    )
    # First call: state has never pruned, must include cleanup_older_than.
    w._schedule_optimize("episode")
    await w._flush_optimizers()
    assert len(fake.optimize_calls) == 1
    assert fake.optimize_cleanup_args[0] is not None, (
        "first optimize must prune to catch up from prior session"
    )
    assert fake.optimize_cleanup_args[0] == dt.timedelta(seconds=10.0)

    # Second call within the prune window: light path (no cleanup).
    await asyncio.sleep(0.02)  # exceed optimize throttle (0.01), not prune (10)
    w._schedule_optimize("episode")
    await w._flush_optimizers()
    assert len(fake.optimize_calls) == 2
    assert fake.optimize_cleanup_args[1] is None, (
        "second optimize within prune window should skip cleanup_older_than"
    )


# ── Rebuild scheduler tests ────────────────────────────────────────────────


async def test_rebuild_runs_on_startup_for_every_kind(
    patched_repo: _FakeRepo,
) -> None:
    """The first rebuild sweep fires on worker start, before any interval.

    Otherwise a daemon that restarts more often than the rebuild
    interval would never bound accumulated UUIDs.
    """
    fake_a = _FakeLanceRepo()
    fake_b = _FakeLanceRepo()
    w = CascadeWorker(
        {
            "episode": _OkHandlerWithRepo(fake_a),
            "atomic_fact": _OkHandlerWithRepo(fake_b),
        },
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
        optimize_heartbeat_seconds=10.0,  # park heartbeat
        optimize_rebuild_interval_seconds=10.0,  # only the startup sweep should fire
    )
    await w.start()
    # Allow the startup sweep to complete; the next tick is 10s away.
    await asyncio.sleep(0.1)
    await w.stop()
    # Exactly one rebuild per kind: the startup sweep. Next interval is 10s.
    assert len(fake_a.rebuild_calls) == 1
    assert len(fake_b.rebuild_calls) == 1


async def test_rebuild_runs_periodically(
    patched_repo: _FakeRepo,
) -> None:
    """After the startup sweep, rebuild repeats every interval."""
    fake = _FakeLanceRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
        optimize_heartbeat_seconds=10.0,
        optimize_rebuild_interval_seconds=0.05,  # ~tick every 50ms in this test
    )
    await w.start()
    await asyncio.sleep(0.2)  # ~4 ticks plus startup sweep
    await w.stop()
    # Startup sweep + at least 2 interval-driven sweeps.
    assert len(fake.rebuild_calls) >= 3, (
        f"expected ≥3 rebuilds (1 startup + ≥2 periodic), got {len(fake.rebuild_calls)}"
    )


async def test_rebuild_failure_does_not_crash_daemon(
    patched_repo: _FakeRepo,
) -> None:
    """A throwing rebuild is logged and absorbed; the worker keeps running."""
    fake = _FakeLanceRepo(rebuild_raises=True)
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
        optimize_heartbeat_seconds=0.05,
        optimize_rebuild_interval_seconds=10.0,
    )
    await w.start()
    # Give startup rebuild a chance to throw, then heartbeat to keep optimizing.
    await asyncio.sleep(0.12)
    # Optimize should still progress despite rebuild errors.
    assert fake.optimize_calls, "heartbeat optimize should run even when rebuild fails"
    await w.stop()
    # Worker is still alive (stop() returned cleanly).
    assert w._task is None
