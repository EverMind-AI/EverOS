"""Tests for :class:`CascadeWorker` retry classification + optimize scheduler.

The pure-function pieces (registry / reconciler) get coverage in
their own files. Here we focus on the worker's branch behaviour
without touching the real handler / lancedb stack:

- ``ExternalServiceError`` retries up to ``max_retry`` and then marks
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

from everos.core.errors import EmbeddingServiceError
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

    async def claim_pending_batch(
        self, _limit: int, *, kinds: set[str] | None = None
    ) -> list[_Row]:
        # Round-2 finding #3: worker now forwards a ``kinds`` scope to
        # the repo. The fake honors it so ``drain_once(kinds=...)``
        # exercises the plumbing end-to-end from tests.
        if kinds is not None:
            picked = [r for r in self.batch if r.kind in kinds]
            self.batch = [r for r in self.batch if r.kind not in kinds]
            return picked
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
    def __init__(self) -> None:
        pass

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=1, deleted=0, skipped=0
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=0, deleted=1, skipped=0
        )


class _BareExceptionHandler(_OkHandler):
    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        raise RuntimeError("unexpected boom")


class _ExternalServiceHandler(_OkHandler):
    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        raise EmbeddingServiceError("embedding 503")


class _VanishedFileHandler(_OkHandler):
    """Simulate a file removed after its modified event was queued."""

    def __init__(self) -> None:
        self.deleted_paths: list[str] = []

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        raise FileNotFoundError(md_path)

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        self.deleted_paths.append(md_path)
        return await super().handle_deleted(md_path)


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


async def test_bare_exception_marked_permanent(patched_repo: _FakeRepo) -> None:
    """Anything that isn't ExternalServiceError counts as unrecoverable."""
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker({"episode": _BareExceptionHandler()}, retry_backoff_seconds=0)
    await w.drain_once()
    _path, retryable, _err, _retry = patched_repo.failed[0]
    assert retryable is False


async def test_external_service_error_is_retried_then_retryable(
    patched_repo: _FakeRepo,
) -> None:
    """ExternalServiceError (embedding / LLM / rerank) retries up to
    max_retry then marks retryable=True."""
    patched_repo.batch = [_Row(md_path="a.md")]
    w = CascadeWorker(
        {"episode": _ExternalServiceHandler()}, max_retry=2, retry_backoff_seconds=0
    )
    await w.drain_once()
    assert patched_repo.done == []
    assert len(patched_repo.failed) == 1
    path, retryable, _err, retry_count = patched_repo.failed[0]
    assert path == "a.md"
    assert retryable is True
    assert retry_count == 2


async def test_modified_event_for_vanished_file_is_processed_as_delete(
    patched_repo: _FakeRepo,
) -> None:
    """A stale modified event must not leave the indexed row behind."""
    patched_repo.batch = [_Row(md_path="vanished.md", change_type="modified")]
    handler = _VanishedFileHandler()
    w = CascadeWorker({"episode": handler}, retry_backoff_seconds=0)

    await w.drain_once()

    assert handler.deleted_paths == ["vanished.md"]
    assert patched_repo.done == ["vanished.md"]
    assert patched_repo.failed == []


async def test_unknown_kind_marks_permanent_without_handler(
    patched_repo: _FakeRepo,
) -> None:
    patched_repo.batch = [_Row(md_path="a.md", kind="mystery")]
    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
    await w.drain_once()
    assert patched_repo.failed[0][1] is False
    assert "no handler" in patched_repo.failed[0][2]


async def test_retry_budget_exhausted_marks_unrecoverable(
    patched_repo: _FakeRepo,
) -> None:
    """When retry_count >= _MAX_TOTAL_RETRIES, the worker skips processing
    and marks the row retryable=False so the scanner stops re-enqueuing."""
    from everos.memory.cascade import worker as wmod

    threshold = wmod._MAX_TOTAL_RETRIES
    patched_repo.batch = [_Row(md_path="a.md", retry_count=threshold)]
    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
    await w.drain_once()
    assert patched_repo.done == []
    assert len(patched_repo.failed) == 1
    path, retryable, err, retry_count = patched_repo.failed[0]
    assert path == "a.md"
    assert retryable is False
    assert "retry budget exhausted" in err
    assert retry_count == threshold


async def test_external_service_error_at_budget_edge_demotes_in_place(
    patched_repo: _FakeRepo,
) -> None:
    """When inline retries push retry_count to the budget mid-batch, the
    row is marked retryable=False directly instead of retryable=True
    followed by a scanner-cycle demotion."""
    from everos.memory.cascade import worker as wmod

    threshold = wmod._MAX_TOTAL_RETRIES
    max_retry = 3
    # Enter with retry_count such that the inline retries bring it exactly
    # to the budget: increments happen on attempts 0..max_retry-1.
    starting = threshold - max_retry
    patched_repo.batch = [_Row(md_path="a.md", retry_count=starting)]
    w = CascadeWorker(
        {"episode": _ExternalServiceHandler()},
        max_retry=max_retry,
        retry_backoff_seconds=0,
    )
    await w.drain_once()
    assert patched_repo.done == []
    assert len(patched_repo.failed) == 1
    path, retryable, _err, retry_count = patched_repo.failed[0]
    assert path == "a.md"
    assert retry_count == threshold
    assert retryable is False, (
        "budget exhausted during inline retries → demote in place, "
        "do not require another scanner cycle to hit retryable=False"
    )


async def test_drain_until_empty_loops_until_no_batch(
    patched_repo: _FakeRepo,
) -> None:
    """Worker keeps draining until claim returns an empty list."""

    rows = [_Row(md_path=f"a{i}.md") for i in range(3)]

    class _ChunkedRepo(_FakeRepo):
        async def claim_pending_batch(
            self, _limit: int, *, kinds: set[str] | None = None
        ) -> list[_Row]:
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
    """Sanity: HandlerDeps accepts the real provider Protocols.

    Embedding is no longer part of this shape — it's a soft dependency
    handlers fetch themselves via ``get_embedding_capability()``.
    """
    # No instantiation needed — just verifies the dataclass shape.
    assert {"memory_root", "tokenizer"} == {
        f.name for f in HandlerDeps.__dataclass_fields__.values()
    }


# ── Optimize scheduler tests ───────────────────────────────────────────────


class _FakeLanceRepo:
    """Records every optimize() / prune() / rebuild_indexes() call.

    The optimize path is split: ``optimize()`` is the light lock-free
    compaction (no args); ``prune(older_than)`` is the heavy write-locked
    reclaim. ``beats`` combines both in call order — most scheduler tests
    only care that *a maintenance beat* ran, not which. The first beat per
    kind is always a prune (``last_prune_at`` starts at 0).

    ``optimize_delay`` / ``rebuild_delay`` simulate slow operations (the
    delay applies to both maintenance beats). ``rebuild_raises`` makes
    ``rebuild_indexes`` raise (crash-safety tests).
    """

    def __init__(
        self,
        *,
        optimize_delay: float = 0.0,
        rebuild_delay: float = 0.0,
        rebuild_raises: bool = False,
    ) -> None:
        self.optimize_calls: list[float] = []
        self.prune_calls: list[float] = []
        self.prune_args: list[dt.timedelta] = []
        self.rebuild_calls: list[float] = []
        self.optimize_delay = optimize_delay
        self.rebuild_delay = rebuild_delay
        self.rebuild_raises = rebuild_raises

    @property
    def beats(self) -> list[float]:
        """All maintenance beats (optimize + prune) in call order."""
        return sorted(self.optimize_calls + self.prune_calls)

    async def optimize(self) -> None:
        if self.optimize_delay > 0:
            await asyncio.sleep(self.optimize_delay)
        self.optimize_calls.append(time.monotonic())

    async def prune(self, older_than: dt.timedelta) -> None:
        if self.optimize_delay > 0:
            await asyncio.sleep(self.optimize_delay)
        self.prune_calls.append(time.monotonic())
        self.prune_args.append(older_than)

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
    assert fake.beats, "expected at least one optimize"
    assert len(fake.beats) == 1, f"burst should collapse, got {len(fake.beats)} calls"


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
    assert len(fake.beats) == 2


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
    assert fake.beats, "flush should not return before optimize ran"
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
    assert len(fake.beats) == 1
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
    assert not fake.beats, "optimize should still be in flight"
    await w._flush_optimizers()
    assert len(fake.beats) == 1


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
    assert len(fake.beats) == 1


async def test_optimize_failure_does_not_crash_drain_loop(
    patched_repo: _FakeRepo,
) -> None:
    """Repo.optimize() raising should be logged but never propagate."""

    class _FailingRepo:
        async def optimize(self) -> None:
            raise RuntimeError("simulated lancedb manifest conflict")

        async def prune(self, older_than: dt.timedelta) -> None:
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
    assert fake_a.beats, "heartbeat should have scheduled episode"
    assert fake_b.beats, "heartbeat should have scheduled atomic_fact"


async def test_optimize_prunes_on_first_call_then_throttles(
    patched_repo: _FakeRepo,
) -> None:
    """First maintenance beat per kind is a heavy ``prune()``; subsequent
    beats within ``optimize_prune_interval_seconds`` take the light
    lock-free ``optimize()`` path.

    Rationale lives in ``DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS``:
    ``prune`` (write-locked, ``delete_unverified``) physically reclaims
    stale files but briefly stalls writes; running it on every 1-second
    tick is wasteful, but never pruning leaks files until FDs / disk
    exhaust. A separate cadence — prune ≪ optimize — balances the two.
    """
    fake = _FakeLanceRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
        optimize_prune_interval_seconds=10.0,  # cadence: long — 2nd beat is light
        optimize_prune_retention_seconds=45.0,  # retention: decoupled from cadence
    )
    # First beat: state has never pruned, must take the heavy prune path.
    w._schedule_optimize("episode")
    await w._flush_optimizers()
    assert len(fake.prune_calls) == 1, "first beat must prune to catch up"
    assert not fake.optimize_calls
    # prune is passed the RETENTION window, not the cadence.
    assert fake.prune_args[0] == dt.timedelta(seconds=45.0)

    # Second beat within the prune window: light lock-free optimize.
    await asyncio.sleep(0.02)  # exceed optimize throttle (0.01), not prune (10)
    w._schedule_optimize("episode")
    await w._flush_optimizers()
    assert len(fake.prune_calls) == 1, "second beat within window must not re-prune"
    assert len(fake.optimize_calls) == 1, "second beat is the light path"


async def test_failed_prune_backs_off_a_cadence_and_keeps_health_signal(
    patched_repo: _FakeRepo,
) -> None:
    """A prune that fails (e.g. killed by the write-lock timeout on a hung
    lance cleanup) advances the *attempt* clock but not the *success* clock:

    - attempt clock advances → the next beat waits a full cadence and takes
      the light lock-free path instead of immediately re-pruning, so a hung
      prune can't pin the write lock ~97% of the time (review N1);
    - success clock (``last_prune_at``) does NOT advance → the prune-staleness
      health signal still climbs, so a persistently failing prune surfaces as
      degraded rather than being masked.
    """

    class _PruneRaisesRepo(_FakeLanceRepo):
        async def prune(self, older_than: dt.timedelta) -> None:
            self.prune_calls.append(time.monotonic())
            self.prune_args.append(older_than)
            raise TimeoutError("simulated hung cleanup killed by write-lock timeout")

    fake = _PruneRaisesRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
        optimize_prune_interval_seconds=10.0,  # long cadence: 2nd beat is light
        optimize_prune_retention_seconds=45.0,
    )
    # First beat: prune is attempted and raises.
    w._schedule_optimize("episode")
    await w._flush_optimizers()
    st = w._optimizer_states["episode"]
    assert len(fake.prune_calls) == 1, "first beat attempts a prune"
    assert st.last_prune_attempt_at > 0, "attempt clock advances even on failure"
    assert st.last_prune_at == 0.0, "success clock must NOT advance on a failed prune"

    # Second beat within the cadence: must fall to the light path, not re-prune.
    await asyncio.sleep(0.02)  # exceeds optimize throttle (0.01), not cadence (10)
    w._schedule_optimize("episode")
    await w._flush_optimizers()
    assert len(fake.prune_calls) == 1, "failed prune must not immediately retry (N1)"
    assert len(fake.optimize_calls) == 1, "second beat backs off to the light path"


async def test_prune_recurs_once_per_cadence_across_light_beats(
    patched_repo: _FakeRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A heavy (prune) beat must fire again after a full cadence, no matter
    how many light beats ran in between.

    The attempt clock advances only on the heavy path. If a light beat also
    pushed it forward, frequent light beats would keep resetting the cadence
    and prune would run exactly once per process lifetime — version cleanup
    silently stops and the index dir grows unbounded, which is the incident
    this whole change exists to prevent.
    """
    from everos.memory.cascade import worker as wmod

    clock = {"t": 1_000.0}
    monkeypatch.setattr(wmod.time, "monotonic", lambda: clock["t"])
    fake = _FakeLanceRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.0,
        optimize_prune_interval_seconds=10.0,
    )

    w._schedule_optimize("episode")
    await w._flush_optimizers()
    assert len(fake.prune_calls) == 1, "first beat prunes (never pruned yet)"

    # 11 light-ish beats, one simulated second apart — they cross the cadence.
    for _ in range(11):
        clock["t"] += 1.0
        w._schedule_optimize("episode")
        await w._flush_optimizers()

    assert len(fake.prune_calls) == 2, (
        "prune must recur one cadence after the last prune ATTEMPT; light "
        "beats in between must not push the cadence forward"
    )
    assert len(fake.optimize_calls) == 10, "the other beats took the light path"


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
    assert fake.beats, "heartbeat optimize should run even when rebuild fails"
    await w.stop()
    # Worker is still alive (stop() returned cleanly).
    assert w._task is None


class _OptimizeFailingRepo(_FakeLanceRepo):
    """Fake repo whose ``optimize()`` AND ``prune()`` raise until ``fail``
    is cleared. ``error`` selects the exception so a test can distinguish a
    genuine failure from a benign commit conflict."""

    def __init__(self, *, error: Exception | None = None, **kw) -> None:  # type: ignore[no-untyped-def]
        super().__init__(**kw)
        self.fail = True
        self._error = error or RuntimeError(
            "Max offset of 9 exceeds length of values 3"
        )

    async def optimize(self) -> None:
        if self.fail:
            raise self._error
        await super().optimize()

    async def prune(self, older_than: dt.timedelta) -> None:
        if self.fail:
            raise self._error
        await super().prune(older_than)


async def test_optimize_failures_counted_escalated_and_reset(
    patched_repo: _FakeRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Layer-2 stop-gap for lance-format/lance#7653.

    Consecutive ``optimize()`` failures are counted, escalate warning→error once
    the threshold is hit, and reset to 0 **only** when the next optimize
    succeeds — not by the fallback rebuild the threshold triggers. Zeroing the
    alert counter inside the branch that fires at the threshold made
    ``optimize_failure_streak >= threshold`` effectively unobservable: the
    counter cycled 1..threshold -> 0 -> 1.. and the only window where a poller
    could see the threshold value was the sub-second rebuild itself. The rate
    limiter lives in ``failures_since_fallback`` instead.
    """
    from everos.memory.cascade import worker as wmod

    calls: list[tuple[str, str]] = []

    class _SpyLogger:
        def __getattr__(self, level: str):  # type: ignore[no-untyped-def]
            def rec(event: str, **_kw) -> None:  # type: ignore[no-untyped-def]
                calls.append((level, event))

            return rec

    monkeypatch.setattr(wmod, "logger", _SpyLogger())

    repo = _OptimizeFailingRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(repo)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.05,
    )
    w._optimizer_states["episode"] = wmod._KindOptimizerState()

    threshold = wmod._OPTIMIZE_FAILURE_ALERT_THRESHOLD
    # Run up to (threshold - 1) failures, then check state.
    for _ in range(threshold - 1):
        await w._run_optimize_once("episode")

    state = w._optimizer_states["episode"]
    assert state.optimize_failures == threshold - 1

    fail_logs = [lvl for lvl, ev in calls if ev == "cascade_lancedb_optimize_failed"]
    assert fail_logs == ["warning"] * (threshold - 1)

    # One more failure triggers the fallback rebuild. The alert counter must
    # keep climbing across it — that is what the health verdict reads.
    await w._run_optimize_once("episode")
    assert state.optimize_failures == threshold, (
        "the fallback rebuild must not reset the alert counter — doing so is "
        "what made the threshold unreachable"
    )
    assert state.failures_since_fallback == 0, "the rate limiter resets, not the alert"
    rebuild_logs = [
        lvl for lvl, ev in calls if ev == "cascade_lancedb_optimize_fallback_rebuild"
    ]
    assert len(rebuild_logs) == 1

    # A success is the only thing that clears either counter.
    repo.fail = False
    await w._run_optimize_once("episode")
    assert state.optimize_failures == 0
    assert state.failures_since_fallback == 0


async def test_optimize_fallback_rebuild_on_sustained_failure(
    patched_repo: _FakeRepo,
) -> None:
    """Consecutive optimize failures >= threshold trigger a fallback rebuild.

    The rebuild drops + recreates indexes, bypassing the Rust panic path. The
    rate limiter (``failures_since_fallback``) resets after the rebuild whether
    it succeeded or not, so the fallback fires at most once per threshold
    failures rather than on every subsequent tick.
    """
    from everos.memory.cascade import worker as wmod

    repo = _OptimizeFailingRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(repo)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.05,
    )
    w._optimizer_states["episode"] = wmod._KindOptimizerState()

    threshold = wmod._OPTIMIZE_FAILURE_ALERT_THRESHOLD
    for _i in range(threshold):
        await w._run_optimize_once("episode")

    state = w._optimizer_states["episode"]
    assert state.failures_since_fallback == 0, "rebuild resets the rate limiter"
    assert len(repo.rebuild_calls) == 1, "exactly one fallback rebuild expected"


async def test_persistent_optimize_failure_stays_visible_in_health(
    patched_repo: _FakeRepo,
) -> None:
    """A table failing 100% of the time must reach the health threshold.

    Regression guard for a reachability bug, not a counting bug: the fallback
    rebuild used to zero the same counter the health verdict reads, so the
    threshold value existed only during the sub-second rebuild. Against a 30s
    scrape that is ~1% observable, i.e. ``cascade.healthy`` — the field
    operators are told to alert on — stayed green while the table never
    reclaimed a version. Sibling of the run7 cross-kind ``max()`` masking bug:
    a remediation path refreshing the very signal meant to report it.

    Driven past the threshold on purpose: the point is that the streak survives
    the remediation, so the check has to run *after* a fallback has fired.
    """
    from everos.memory.cascade import worker as wmod

    repo = _OptimizeFailingRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(repo)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.05,
    )
    w._optimizer_states["episode"] = wmod._KindOptimizerState()

    threshold = wmod._OPTIMIZE_FAILURE_ALERT_THRESHOLD
    for _i in range(threshold * 2 + 1):
        await w._run_optimize_once("episode")

    assert len(repo.rebuild_calls) == 2, (
        "fallback stays rate-limited to once per threshold failures"
    )
    health = w.health()
    assert health.optimize_failure_streak >= threshold
    reasons = health.reasons()
    assert any("optimize" in r for r in reasons), (
        f"health must name the stuck optimize; got {reasons}"
    )


# ── health signals ───────────────────────────────────────────────────────────


def test_worker_health_dataclass_thresholds() -> None:
    """reasons() fires exactly on each threshold, one entry per crossed signal."""
    from everos.memory.cascade import worker as wmod

    mk = wmod.CascadeWorkerHealth
    assert mk(0, 0, 0, 0.0).reasons() == []
    assert mk(wmod._DRAIN_FAILURE_ALERT_THRESHOLD, 0, 0, 0.0).reasons()
    assert mk(0, 0, wmod._OPTIMIZE_FAILURE_ALERT_THRESHOLD, 0.0).reasons()
    assert mk(0, 0, 0, wmod._PRUNE_STALE_SECONDS_ALERT).reasons()

    reasons = mk(
        wmod._DRAIN_FAILURE_ALERT_THRESHOLD, 0, 0, wmod._PRUNE_STALE_SECONDS_ALERT
    ).reasons()
    assert len(reasons) == 2  # drain + prune, not the sub-threshold optimize


def test_worker_health_idle_is_not_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker with no optimize activity is never prune-stale (nothing to
    reclaim), even if it started long ago."""
    from everos.memory.cascade import worker as wmod

    # Freeze the clock: a bare `monotonic() - N` goes negative on a runner
    # whose uptime is below N, making the "started long ago" premise fiction.
    now = 10_000.0
    monkeypatch.setattr(wmod.time, "monotonic", lambda: now)
    w = CascadeWorker({"episode": _OkHandlerWithRepo(_FakeLanceRepo())})
    w._started_at = now - (wmod._PRUNE_STALE_SECONDS_ALERT + 1000)
    h = w.health()
    assert h.prune_stale_seconds == 0.0
    assert h.prune_stale_kind is None
    assert h.reasons() == []


def test_worker_health_reports_worst_kind_not_newest_prune(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Staleness is the WORST kind's, and ``reasons`` names it.

    Production registers several lance-backed kinds. Reporting the newest
    prune across kinds let one healthy kind mask a kind whose cleanup had
    died — the dead kind's index dir grows unbounded while ``/health`` stays
    green, which is the incident this signal exists to catch.
    """
    from everos.memory.cascade import worker as wmod

    now = 100_000.0
    monkeypatch.setattr(wmod.time, "monotonic", lambda: now)
    w = CascadeWorker(
        {
            "episode": _OkHandlerWithRepo(_FakeLanceRepo()),
            "atomic_fact": _OkHandlerWithRepo(_FakeLanceRepo()),
        }
    )
    w._started_at = now - 10_000.0
    # atomic_fact pruned just now; episode has not pruned in 3x the threshold.
    fresh = wmod._KindOptimizerState()
    fresh.last_prune_at = now - 10.0
    dead = wmod._KindOptimizerState()
    dead.last_prune_at = now - 3 * wmod._PRUNE_STALE_SECONDS_ALERT
    w._optimizer_states["atomic_fact"] = fresh
    w._optimizer_states["episode"] = dead

    h = w.health()
    assert h.prune_stale_kind == "episode", "must report the worst kind"
    assert h.prune_stale_seconds >= wmod._PRUNE_STALE_SECONDS_ALERT
    reasons = h.reasons()
    assert any("cleanup stalled" in r for r in reasons)
    assert any("episode" in r for r in reasons), "operator needs the kind named"


def test_worker_health_reports_prune_staleness(monkeypatch: pytest.MonkeyPatch) -> None:
    """An active kind that has never successfully pruned since start goes
    stale once past the alert threshold."""
    from everos.memory.cascade import worker as wmod

    # Freeze the monotonic clock so staleness is deterministic. Real
    # ``time.monotonic()`` returns process/boot uptime, which is huge on a
    # long-lived dev box but only ~100s on a fresh CI runner — a bare
    # ``monotonic() - 1000`` would go negative there and read as not-stale.
    now = 10_000.0
    monkeypatch.setattr(wmod.time, "monotonic", lambda: now)
    w = CascadeWorker({"episode": _OkHandlerWithRepo(_FakeLanceRepo())})
    w._started_at = now - (wmod._PRUNE_STALE_SECONDS_ALERT + 100)
    w._optimizer_states["episode"] = wmod._KindOptimizerState()  # last_prune_at=0
    h = w.health()
    assert h.prune_stale_seconds >= wmod._PRUNE_STALE_SECONDS_ALERT
    assert any("cleanup stalled" in r for r in h.reasons())


def test_worker_health_forwards_counters() -> None:
    """drain / unrecoverable / optimize-streak counters surface verbatim."""
    from everos.memory.cascade import worker as wmod

    w = CascadeWorker({"episode": _OkHandlerWithRepo(_FakeLanceRepo())})
    w._drain_consecutive_failures = 2
    w._unrecoverable_total = 7
    st = wmod._KindOptimizerState()
    st.optimize_failures = 4
    w._optimizer_states["episode"] = st
    h = w.health()
    assert h.drain_consecutive_failures == 2
    assert h.unrecoverable_total == 7
    assert h.optimize_failure_streak == 4


async def test_light_beat_commit_conflict_is_debug_and_uncounted(
    patched_repo: _FakeRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A benign light-beat commit conflict must not pollute the signal.

    The lock-free compaction can lose the optimistic-concurrency race
    against a live writer; that is expected under churn and self-heals
    next beat. It is logged at ``debug``, does NOT increment the failure
    streak, and does NOT trigger a fallback rebuild.
    """
    from everos.memory.cascade import worker as wmod

    calls: list[tuple[str, str]] = []

    class _SpyLogger:
        def __getattr__(self, level: str):  # type: ignore[no-untyped-def]
            def rec(event: str, **_kw) -> None:  # type: ignore[no-untyped-def]
                calls.append((level, event))

            return rec

    monkeypatch.setattr(wmod, "logger", _SpyLogger())

    repo = _OptimizeFailingRepo(error=RuntimeError("Retryable commit conflict"))
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(repo)},
        retry_backoff_seconds=0,
    )
    state = wmod._KindOptimizerState()
    # Force the LIGHT beat: pretend we just attempted a prune so should_prune=False
    # (scheduling reads the attempt clock, not the success clock — see N1 split).
    state.last_prune_attempt_at = time.monotonic()
    w._optimizer_states["episode"] = state

    await w._run_optimize_once("episode")

    assert state.optimize_failures == 0, "benign conflict must not count"
    events = [ev for _, ev in calls]
    levels = {lvl for lvl, _ in calls}
    assert "cascade_lancedb_optimize_conflict" in events
    assert "cascade_lancedb_optimize_failed" not in events
    assert "cascade_lancedb_optimize_fallback_rebuild" not in events
    assert "error" not in levels and "warning" not in levels


async def test_heavy_beat_commit_conflict_is_also_benign(
    patched_repo: _FakeRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A HEAVY-beat (prune) commit conflict is benign too.

    The per-table write lock is in-process only, so a second process (a long
    ``cascade backfill``, a ``cascade sync``) can preempt prune's Rewrite
    commit. Counting those as real failures let sustained cross-process churn
    reach the alert threshold and fire a spurious fallback rebuild, which
    drops every index before recreating it — leaving the table without an FTS
    index (and `/search` 500ing on that kind) if the rebuild lost the race
    too. A prune that genuinely stops succeeding is caught by the per-kind
    prune-staleness signal instead.
    """
    from everos.memory.cascade import worker as wmod

    class _ConflictRepo(_FakeLanceRepo):
        """Records which beat was attempted, then loses the commit race."""

        def __init__(self) -> None:
            super().__init__()
            self.attempts: list[str] = []

        async def optimize(self) -> None:
            self.attempts.append("optimize")
            raise RuntimeError("Retryable commit conflict for version 215")

        async def prune(self, older_than: dt.timedelta) -> None:
            self.attempts.append("prune")
            raise RuntimeError("Retryable commit conflict for version 215")

    # Freeze the clock: which beat runs must not depend on the runner's uptime
    # (`monotonic()` is boot-relative — a fresh CI runner reads ~100s).
    now = 10_000.0
    monkeypatch.setattr(wmod.time, "monotonic", lambda: now)
    repo = _ConflictRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(repo)},
        retry_backoff_seconds=0,
        optimize_prune_interval_seconds=10.0,
    )
    state = wmod._KindOptimizerState()  # last_prune_attempt_at=0 → HEAVY beat
    w._optimizer_states["episode"] = state

    await w._run_optimize_once("episode")

    assert repo.attempts == ["prune"], "must have taken the heavy (prune) path"
    assert state.optimize_failures == 0, (
        "a heavy-beat commit conflict is a lost cross-process race, not a "
        "failure — counting it re-arms the spurious fallback rebuild"
    )


async def test_conflict_log_names_the_lost_beat(
    patched_repo: _FakeRepo,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conflict log must say WHICH beat lost the race.

    Lance labels both beats' commit identically ("This Rewrite transaction
    was preempted by concurrent transaction ..."), so the error message
    cannot tell them apart — yet the cost differs sharply:

    - a lost LIGHT beat is free (compaction retries ~10s later);
    - a lost HEAVY beat means that table skipped a whole prune cadence, so
      its superseded files stay on disk until the next one lands.

    Without ``pruned`` on this line, attributing index-dir growth means
    back-inferring which beats were heavy from the 300s cadence.
    """
    from everos.memory.cascade import worker as wmod

    calls: list[tuple[str, dict[str, object]]] = []

    class _SpyLogger:
        def __getattr__(self, level: str):  # type: ignore[no-untyped-def]
            def rec(event: str, **kw) -> None:  # type: ignore[no-untyped-def]
                calls.append((event, kw))

            return rec

    monkeypatch.setattr(wmod, "logger", _SpyLogger())

    class _ConflictRepo(_FakeLanceRepo):
        """Loses the commit race on both beats."""

        async def optimize(self) -> None:
            raise RuntimeError("Retryable commit conflict for version 215")

        async def prune(self, older_than: dt.timedelta) -> None:
            raise RuntimeError("Retryable commit conflict for version 215")

    # Freeze the clock: which beat runs must not depend on the runner's uptime
    # (`monotonic()` is boot-relative — a fresh CI runner reads ~100s).
    monkeypatch.setattr(wmod.time, "monotonic", lambda: 10_000.0)
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(_ConflictRepo())},
        retry_backoff_seconds=0,
        optimize_prune_interval_seconds=300.0,
    )
    state = wmod._KindOptimizerState()  # last_prune_attempt_at=0 → HEAVY first
    w._optimizer_states["episode"] = state

    await w._run_optimize_once("episode")  # heavy: prune loses the race
    # The attempt clock advanced to the (frozen) now, so the cadence has not
    # elapsed — the next beat is the light one.
    await w._run_optimize_once("episode")

    lost = [kw for ev, kw in calls if ev == "cascade_lancedb_optimize_conflict"]
    assert len(lost) == 2, "both beats must have logged a conflict"
    assert lost[0]["pruned"] is True, (
        "the heavy beat lost — this is the one that costs a prune cadence"
    )
    assert lost[1]["pruned"] is False, (
        "the light beat lost — free, the next beat retries it"
    )


async def test_non_conflict_failure_counts_even_when_message_says_retryable(
    patched_repo: _FakeRepo,
) -> None:
    """The benign filter must match ONLY ``commit conflict``.

    Widening it to a bare ``retryable`` substring would swallow unrelated
    recoverable errors — an ``ExternalServiceError`` repr carries
    ``retryable=True`` — so a genuinely stuck optimize would log at debug
    forever: no streak, no escalation, no fallback rebuild, health green.
    """
    from everos.memory.cascade import worker as wmod

    repo = _OptimizeFailingRepo(
        error=RuntimeError("ExternalServiceError(provider='x', retryable=True): boom")
    )
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(repo)},
        retry_backoff_seconds=0,
    )
    state = wmod._KindOptimizerState()
    state.last_prune_attempt_at = time.monotonic()  # light beat
    w._optimizer_states["episode"] = state

    await w._run_optimize_once("episode")

    assert state.optimize_failures == 1, (
        "an error whose message merely contains 'retryable' is NOT a commit "
        "conflict and must count toward the streak"
    )


# ── background-loop supervision ──────────────────────────────────────────────


async def test_a_crashed_loop_is_restarted_instead_of_dying_silently(
    patched_repo: _FakeRepo,
) -> None:
    """A background loop that raises must be restarted, not lost.

    The three long-lived loops are plain ``create_task`` coroutines. Without
    supervision one uncaught exception ends that loop permanently: nothing
    restarts it, and because the worker keeps a strong reference to the task the
    interpreter never prints "Task exception was never retrieved" either (that
    fires on GC). The loop's job just stops happening, with zero output.
    """
    from everos.memory.cascade import worker as wmod

    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(wmod, "_LOOP_RESTART_BACKOFF_SECONDS", (0.0, 0.0, 0.0))

    runs = 0

    async def _body() -> None:
        nonlocal runs
        runs += 1
        if runs < 3:
            raise RuntimeError("boom")
        w._stop.set()  # third run exits cleanly

    try:
        await w._supervise("test-loop", _body)
    finally:
        monkeypatched.undo()

    assert runs == 3, "the loop must be restarted after each crash"


async def test_a_permanently_crashing_loop_asks_the_process_to_exit(
    patched_repo: _FakeRepo,
) -> None:
    """Once the restart budget is spent, the worker asks the process to exit.

    Deployments are expected to run under a restarting supervisor (systemd
    ``Restart=always``, Docker ``restart: unless-stopped``, a k8s Deployment).
    Continuing to serve with a dead projection pipeline is worse: searches
    answer from a silently frozen index while ``/health`` shows nothing wrong.
    """
    from everos.memory.cascade import worker as wmod

    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
    exits: list[str] = []
    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(wmod, "_LOOP_RESTART_BACKOFF_SECONDS", (0.0, 0.0))
    monkeypatched.setattr(w, "_request_process_exit", exits.append)

    attempts = 0

    async def _always_raises() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("deterministic failure")

    try:
        await w._supervise("test-loop", _always_raises)
    finally:
        monkeypatched.undo()

    assert attempts == 3, "initial run + one per backoff entry"
    assert exits == ["test-loop"], "process exit must be requested exactly once"


async def test_a_stable_run_refills_the_restart_budget(
    patched_repo: _FakeRepo,
) -> None:
    """Independent transients days apart must not pool into a process exit.

    The restart budget counts consecutive *quick* crashes, not crashes over
    the process lifetime. Without the reset, a loop that hits one recoverable
    transient every few days — each cleared by a single restart — spends the
    budget strike by strike and the crash after the last one SIGTERMs a
    healthy server weeks in, which punishes exactly the case supervision
    exists to absorb. A body that ran at least ``_LOOP_STABLE_RUN_SECONDS``
    before raising is a fresh incident and gets the full ladder again.
    """
    from types import SimpleNamespace

    from everos.memory.cascade import worker as wmod

    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
    exits: list[str] = []
    clock = {"now": 0.0}
    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(wmod, "_LOOP_RESTART_BACKOFF_SECONDS", (0.0, 0.0))
    # Only the worker module sees the fake clock; the event loop keeps real
    # time, so the zero backoffs above still pass through _wait_or_stop.
    monkeypatched.setattr(wmod, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
    monkeypatched.setattr(w, "_request_process_exit", exits.append)

    runs = 0

    async def _body() -> None:
        nonlocal runs
        runs += 1
        if runs == 3:
            # A long, honest run before this crash — an independent incident.
            clock["now"] += wmod._LOOP_STABLE_RUN_SECONDS + 1
            raise RuntimeError("independent transient")
        if runs == 5:
            w._stop.set()  # clean exit
            return
        raise RuntimeError("quick crash")

    try:
        await w._supervise("test-loop", _body)
    finally:
        monkeypatched.undo()

    # Budget is 2 restarts here: crashes 1-2 spend it, crash 3 (after a
    # stable run) must start over rather than exceed it, leaving room for
    # crash 4 and the clean run 5. A lifetime budget exits after run 3.
    assert runs == 5, "the stable run must refill the budget"
    assert exits == [], "no process exit for independent incidents"


async def test_supervision_does_not_swallow_cancellation(
    patched_repo: _FakeRepo,
) -> None:
    """``stop()`` cancels these tasks, so ``CancelledError`` must propagate.

    Catching it as a "crash" would restart the loop during shutdown and make
    ``stop()`` hang instead of returning.
    """
    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)

    async def _body() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await w._supervise("test-loop", _body)


async def test_all_three_loops_are_supervised(patched_repo: _FakeRepo) -> None:
    """Supervision must cover every long-lived loop, not just the drain one.

    ``_run_loop`` already had an inner ``try``; ``_heartbeat_loop`` and
    ``_rebuild_loop`` did not, which is the asymmetry this guards.
    """
    w = CascadeWorker({"episode": _OkHandler()}, retry_backoff_seconds=0)
    await w.start()
    try:
        names = {
            t.get_name()
            for t in (w._task, w._heartbeat_task, w._rebuild_task)
            if t is not None
        }
        assert names == {
            "cascade-worker",
            "cascade-worker-heartbeat",
            "cascade-worker-rebuild",
        }
        for task in (w._task, w._heartbeat_task, w._rebuild_task):
            assert task is not None
            assert task.get_coro().__qualname__.endswith("_supervise"), (
                f"{task.get_name()} is not running under _supervise"
            )
    finally:
        await w.stop()


async def test_optimize_does_not_park_forever_on_a_rebuild(
    patched_repo: _FakeRepo,
) -> None:
    """The optimize runner's wait on a rebuild must be bounded, like its mirror.

    Whichever maintenance job arrives second parks on the first, so the two
    waits are the same hazard seen from opposite ends: while the runner waits,
    its task slot stays occupied, ``_schedule_optimize`` keeps short-circuiting
    on it, and that table quietly stops being pruned. The rebuild-side wait was
    bounded; this one was left open on the argument that ``rebuild_indexes``
    carries its own deadline — true of its critical section only, not of the
    task's dispatch and teardown, so nothing actually bounded it.
    """
    from everos.memory.cascade import worker as wmod

    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(wmod, "_MAINTENANCE_TASK_TIMEOUT_SECONDS", 0.05)

    fake = _FakeLanceRepo()
    w = CascadeWorker(
        {"episode": _OkHandlerWithRepo(fake)},
        retry_backoff_seconds=0,
        optimize_min_interval_seconds=0.01,
    )
    state = wmod._KindOptimizerState()
    w._optimizer_states["episode"] = state
    state.dirty = True
    # A rebuild that never finishes: exactly the state the runner used to wait
    # out forever.
    state.rebuild_task = asyncio.create_task(asyncio.sleep(30))

    try:
        async with asyncio.timeout(5):
            await w._optimize_runner("episode", initial_delay=0)
    finally:
        state.rebuild_task.cancel()
        monkeypatched.undo()

    assert not fake.optimize_calls, (
        "the runner must skip the beat, not compact under a live rebuild — "
        "both commit on the same manifest"
    )
