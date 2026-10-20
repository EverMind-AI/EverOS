"""Model-based test of the cascade worker's index-maintenance cycle.

Third instance of the pattern in
``tests/unit/test_service/test_memorize_queue_stateful.py``; the sibling
``test_worker_stateful.py`` covers the ``md_change_state`` row cycle. The
subject here is everything downstream of a drained batch: the per-kind
throttle (:meth:`CascadeWorker._schedule_optimize`), the runner with its
trailing edge and cooldown, the light / heavy beat split (``optimize`` vs
``prune`` + ``ensure_vector_indexes``), the maintenance-call timeout, the
failure streak and its fallback rebuild, the rebuild's commit-conflict
backoff ladder, the optimize / rebuild mutual exclusion with its bounded
waits in both directions, the startup rebuild sweep, ``start`` / ``stop``
and :meth:`CascadeWorker.health`.

What is real: all of ``worker.py``, including its three supervised loops
(``start()`` launches them; their timers are set to an hour so the rules,
not wall time, decide when a heartbeat or a rebuild sweep fires -- the
startup sweep is the loop's own). What is scripted: the LanceDB repository
behind ``handler.index_repo`` (its four maintenance methods), the md queue
(always empty: the row cycle has its own test) and the worker's clock,
which ``advance_clock`` moves so a 300 s prune cadence or a 900 s staleness
alert does not cost 900 s of wall time (asyncio keeps its own clock, so the
timers below still run in real seconds).

Rules, i.e. what the outside world does to this subsystem: ``touch`` (a
drained batch wrote to a kind's table; the beats it causes are scripted,
including a second write landing mid-beat and a rebuild sweep arriving
mid-beat), ``heartbeat`` (the 60 s sweep's tick), ``rebuild`` (the 12 h
sweep or a due conflict retry reached this kind, optionally with a write
landing during the rebuild), ``advance_clock`` and ``restart``.

Invariants after every rule (the machine first waits for the worker to
settle, so every call the rule caused has been made):

1. the scripted repository saw exactly the calls the spec predicts, in
   order -- the ledger: one beat per touch and one more per mid-beat
   write, heavy on a kind's first beat and again once ``prune_interval``
   has passed, ``ensure`` only after a successful prune, a fallback
   rebuild on the fifth non-benign failure and not before, a commit
   conflict never counted, a sweep's rebuild skipped when the runner it
   waits on outlives the timeout;
2. no two maintenance calls were ever in flight on one kind's table;
3. ``dirty`` without a runner only when the runner gave up waiting on a
   long rebuild (the write is then picked up by the next schedule);
4. ``health()``: the streak is the run of non-benign failures since the
   last success, the staleness signal names the kind that has gone
   longest without a successful prune, both alert reasons fire at their
   thresholds and not below;
5. the rebuild bookkeeping the loop reads (attempt, retry deadline)
   follows the backoff ladder;
6. consecutive beats of one kind are at least ``min_interval`` apart on
   the worker's clock.

Timing the model relies on: a normal call takes 0.01 s, the maintenance
timeout is patched to 0.3 s, a "slow" rebuild takes 0.6 s and a "hang"
is cut by the timeout. A rebuild arriving mid-beat therefore proceeds
when the runner ends within ~0.15 s (two more beats and cooldowns) and
is skipped when a later beat hangs (0.3 s) or a slow fallback rebuild
runs inside the runner (0.6 s); either side has at least 0.1 s of margin
and the two deadlines are asyncio timers on one loop, which fire in
deadline order. The one combination with no margin -- a rebuild arriving
during a beat that itself hangs -- is not scripted.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import random
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from hypothesis import HealthCheck, currently_in_test_context, given, settings
from hypothesis import event as _hypothesis_event
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from everos.memory.cascade import worker as worker_mod
from everos.memory.cascade.handlers import Handler
from everos.memory.cascade.types import HandlerOutcome
from everos.memory.cascade.worker import CascadeWorker

settings.register_profile(
    "ci",
    max_examples=30,
    stateful_step_count=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "deep",
    max_examples=1000,
    stateful_step_count=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

INDEXED = ("episode", "profile")
KINDS = (*INDEXED, "plain")  # "plain": a handler without index_repo
# Duplicates weight the draw; each beat is scripted independently.
BEAT_OUTCOMES = (
    "ok",
    "ok",
    "conflict",
    "fail",
    "fail",
    "hang",
    "ensure_fail",
    "rebuild_during",
)
REBUILD_OUTCOMES = ("ok", "ok", "conflict", "fail", "touch", "slow_touch")
MIN_INTERVAL = 0.03
PRUNE_INTERVAL = 100_000.0
TIMEOUT = 0.3  # patched over _MAINTENANCE_TASK_TIMEOUT_SECONDS
CALL = 0.01  # a normal maintenance call
SLOW = 0.6  # a rebuild the optimize runner gives up waiting for
HANG = 5.0  # a call the timeout cuts short
CLOCK_STEPS = (1_000.0, 200_000.0)  # below / above the prune cadence
TOL = 2.0  # seconds of slack between the model's clock reads and the worker's
THRESHOLD = worker_mod._OPTIMIZE_FAILURE_ALERT_THRESHOLD
BACKOFFS = worker_mod._REBUILD_CONFLICT_BACKOFFS_SECONDS
ALERT = worker_mod._PRUNE_STALE_SECONDS_ALERT


def event(label: str) -> None:
    if currently_in_test_context():
        _hypothesis_event(label)


class _Clock:
    """``time`` as the worker sees it: real monotonic plus a movable offset."""

    def __init__(self) -> None:
        self.offset = 0.0

    def monotonic(self) -> float:
        return time.monotonic() + self.offset

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


class _EmptyQueue:
    """The md queue, always empty: the drain loop finds nothing to do."""

    async def claim_pending_batch(
        self, limit: int, *, kinds: set[str] | None = None
    ) -> list[Any]:
        return []


@dataclass
class _Play:
    """One call the model expects, and what the repository should do with it."""

    op: str
    served: str  # ok | conflict | fail | hang
    actions: tuple[str, ...] = ()  # "touch" | "spawn_rebuild", at call entry
    slow: bool = False


class _ScriptedLanceRepo:
    """Plays the calls the model queued for one kind and records what it saw."""

    def __init__(self, machine: MaintenanceMachine, kind: str) -> None:
        self.machine = machine
        self.kind = kind
        self.queue: list[_Play] = []
        self.calls: list[tuple[str, str]] = []
        self.beat_starts: list[float] = []  # worker clock, per worker life
        self.active = 0

    async def optimize(self) -> None:
        await self._serve("optimize")

    async def prune(self, older_than: Any) -> None:
        await self._serve("prune")

    async def ensure_vector_indexes(self) -> list[str]:
        await self._serve("ensure")
        return []

    async def rebuild_indexes(self) -> None:
        await self._serve("rebuild")

    async def _serve(self, op: str) -> None:
        m = self.machine
        play = self.queue.pop(0) if self.queue else None
        if play is None or play.op != op:
            m.violations.append(f"{self.kind}: {op} called, script had {play}")
            play = _Play(op, "ok")
        self.active += 1
        if self.active > 1:
            m.violations.append(f"{self.kind}: {op} overlapped another call")
        self.calls.append((op, play.served))
        if op in ("optimize", "prune"):
            self.beat_starts.append(m.clock.monotonic())
        m.trace.append(f"  {self.kind}.{op} -> {play.served} {play.actions}")
        try:
            for action in play.actions:
                if action == "touch":
                    m.worker._schedule_optimize(self.kind)
                else:
                    m.side.append(
                        asyncio.create_task(m.worker._run_rebuild_once(self.kind))
                    )
            if play.served == "hang":
                await asyncio.sleep(HANG)
            else:
                await asyncio.sleep(SLOW if play.slow else CALL)
            if play.served == "conflict":
                raise RuntimeError("Retryable commit conflict: preempted by a writer")
            if play.served == "fail":
                raise RuntimeError("index maintenance boom")
        finally:
            self.active -= 1


class _IndexedHandler(Handler):
    def __init__(self, repo: _ScriptedLanceRepo) -> None:  # no HandlerDeps
        self.index_repo = repo

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=1, deleted=0, skipped=0
        )

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=0, deleted=1, skipped=0
        )


class _PlainHandler(_IndexedHandler):
    def __init__(self) -> None:
        pass


@dataclass
class _KindModel:
    """The spec's bookkeeping for one kind; the ledger survives restarts."""

    expected: list[tuple[str, str]] = field(default_factory=list)
    failures: int = 0
    since_fallback: int = 0
    last_prune_attempt: float | None = None
    last_prune_ok: float | None = None
    rebuild_attempt: int = 0
    retry_at: float = 0.0
    gave_up: bool = False

    def forget(self) -> None:
        """A restart loses the in-memory state, not the table."""
        self.failures = self.since_fallback = self.rebuild_attempt = 0
        self.last_prune_attempt = self.last_prune_ok = None
        self.retry_at = 0.0
        self.gave_up = False


class MaintenanceMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.clock = _Clock()
        self._patches = [
            mock.patch.object(worker_mod, "time", self.clock),
            mock.patch.object(worker_mod, "md_change_state_repo", _EmptyQueue()),
            mock.patch.object(worker_mod, "_MAINTENANCE_TASK_TIMEOUT_SECONDS", TIMEOUT),
        ]
        for p in self._patches:
            p.start()
        self.repos = {kind: _ScriptedLanceRepo(self, kind) for kind in INDEXED}
        self.handlers: dict[str, Handler] = {
            kind: _IndexedHandler(self.repos[kind]) for kind in INDEXED
        }
        self.handlers["plain"] = _PlainHandler()
        self.m = {kind: _KindModel() for kind in INDEXED}
        self.violations: list[str] = []
        self.side: list[asyncio.Task[None]] = []
        self.trace: list[str] = []
        self.started_at = 0.0
        self.worker: CascadeWorker
        self.run(self._start("ok"))

    def run(self, coro: Any) -> Any:
        # Bounded: the first deep run of this machine sat 85 minutes at 0 %
        # CPU with no trace to read. A step that does not finish in a minute
        # is a failure with the trace, not a hang.
        return self.loop.run_until_complete(asyncio.wait_for(coro, timeout=60.0))

    def teardown(self) -> None:
        try:
            self.run(self.worker.stop())
        finally:
            for p in self._patches:
                p.stop()
            self.loop.close()

    def _now(self) -> float:
        return self.clock.monotonic()

    # ── driving the worker ─────────────────────────────────────────────────

    async def _start(self, sweep: str) -> None:
        self.worker = CascadeWorker(
            self.handlers,
            retry_backoff_seconds=0.0,
            poll_interval_seconds=3600.0,
            optimize_min_interval_seconds=MIN_INTERVAL,
            optimize_heartbeat_seconds=3600.0,
            optimize_prune_interval_seconds=PRUNE_INTERVAL,
            optimize_rebuild_interval_seconds=3600.0,
        )
        self.started_at = self._now()
        for kind in INDEXED:
            self.m[kind].forget()
            self.repos[kind].beat_starts.clear()
            self._rebuild(kind, sweep, inside_runner=False)  # the startup sweep
        await self.worker.start()
        await self._settle(sweep=True)

    async def _schedule(self, *kinds: str) -> None:
        for kind in kinds:  # create_task inside: needs the running loop
            self.worker._schedule_optimize(kind)
        await self._settle()

    async def _settle(self, *, sweep: bool = False) -> None:
        """Wait until nothing the last rule caused is still running."""
        deadline = time.perf_counter() + 20.0
        while time.perf_counter() < deadline:
            await self.worker._flush_optimizers()
            if self.side:
                side, self.side = self.side, []
                await asyncio.gather(*side, return_exceptions=True)
                continue
            states = self.worker._optimizer_states.values()
            quiet = all(s.task is None and s.rebuild_task is None for s in states)
            quiet = quiet and all(r.active == 0 for r in self.repos.values())
            if sweep:  # the startup sweep runs inside the rebuild loop's task
                quiet = quiet and all(not r.queue for r in self.repos.values())
            if quiet:
                return
            await asyncio.sleep(0.005)
        raise RuntimeError("worker never settled\n" + "\n".join(self.trace[-30:]))

    # ── the model ──────────────────────────────────────────────────────────

    def _play(
        self, kind: str, op: str, served: str, actions: tuple[str, ...], slow: bool
    ) -> None:
        self.repos[kind].queue.append(_Play(op, served, actions, slow))
        self.m[kind].expected.append((op, served))

    def _touch_model(
        self, kind: str, outcomes: list[str], rebuild_outcome: str
    ) -> None:
        """What one ``_schedule_optimize`` must cause: the runner's beats."""
        km = self.m[kind]
        km.gave_up = False
        queue = list(outcomes)
        spawned = runner_long_after_spawn = dirty = False
        while queue or dirty:
            o = queue.pop(0) if queue else "ok"
            actions: list[str] = ["touch"] if queue else []  # a write mid-beat
            if o == "rebuild_during" and spawned:
                o = "ok"  # the rebuild loop is one task: it cannot arrive twice
            if o == "rebuild_during":
                actions.append("spawn_rebuild")
                spawned = True
            elif spawned and o == "hang":
                runner_long_after_spawn = True
            fallback, dirty = self._beat(kind, o, tuple(actions), rebuild_outcome)
            if fallback and spawned and rebuild_outcome == "slow_touch":
                runner_long_after_spawn = True
            dirty = dirty or bool(queue)
        if spawned and runner_long_after_spawn:
            event("rebuild during beat: skipped, runner outlived the wait")
        elif spawned:
            self._rebuild(kind, rebuild_outcome, inside_runner=False)

    def _beat(
        self, kind: str, o: str, actions: tuple[str, ...], rebuild_outcome: str
    ) -> tuple[bool, bool]:
        """One ``_run_optimize_once``; returns (fallback ran, dirty re-raised)."""
        km = self.m[kind]
        now = self._now()
        heavy = (
            km.last_prune_attempt is None
            or now - km.last_prune_attempt >= PRUNE_INTERVAL
        )
        served = {"rebuild_during": "ok", "ensure_fail": "ok"}.get(o, o)
        self._play(kind, "prune" if heavy else "optimize", served, actions, False)
        failed = served in ("fail", "hang")
        if heavy:
            km.last_prune_attempt = now
            if served == "ok":
                km.last_prune_ok = now
                ensure = "fail" if o == "ensure_fail" else "ok"
                self._play(kind, "ensure", ensure, (), False)
                failed = ensure == "fail"
        event(f"beat: {'heavy' if heavy else 'light'} {o}")
        if served == "conflict":
            return False, False
        if not failed:
            km.failures = km.since_fallback = 0
            return False, False
        km.failures += 1
        km.since_fallback += 1
        if km.since_fallback < THRESHOLD:
            return False, False
        km.since_fallback = 0
        event("fallback rebuild")
        return True, self._rebuild(kind, rebuild_outcome, inside_runner=True)

    def _rebuild(self, kind: str, ro: str, *, inside_runner: bool) -> bool:
        """One ``_run_rebuild_once``; returns whether it re-raised ``dirty``."""
        km = self.m[kind]
        now = self._now()
        served = {"touch": "ok", "slow_touch": "ok"}.get(ro, ro)
        actions = ("touch",) if ro in ("touch", "slow_touch") else ()
        self._play(kind, "rebuild", served, actions, ro == "slow_touch")
        if served == "ok":
            km.rebuild_attempt, km.retry_at = 0, 0.0
        elif served == "conflict" and km.rebuild_attempt < len(BACKOFFS):
            km.retry_at = now + BACKOFFS[km.rebuild_attempt]
            km.rebuild_attempt += 1
        else:
            km.rebuild_attempt, km.retry_at = 0, 0.0
        event(f"rebuild: {ro}" + (" (fallback)" if inside_runner else ""))
        if inside_runner:
            # The write lands while the runner is alive: dirty, one more beat.
            return ro in ("touch", "slow_touch")
        if ro == "touch":  # a new runner waits out the short rebuild, then beats
            km.gave_up = False
            self._beat(kind, "ok", (), "ok")
        elif ro == "slow_touch":  # the new runner gives up waiting; dirty stays
            km.gave_up = True
            event("runner gave up waiting on a rebuild")
        return False

    # ── rules ──────────────────────────────────────────────────────────────

    @rule(
        kind=st.sampled_from(KINDS),
        outcomes=st.lists(st.sampled_from(BEAT_OUTCOMES), min_size=1, max_size=3),
        rebuild_outcome=st.sampled_from(REBUILD_OUTCOMES),
    )
    def touch(self, kind: str, outcomes: list[str], rebuild_outcome: str) -> None:
        """A drained batch wrote to ``kind``'s table."""
        self.trace.append(f"touch {kind} {outcomes} rebuild={rebuild_outcome}")
        if kind in self.m:
            self._touch_model(kind, outcomes, rebuild_outcome)
        else:
            event("touch: kind without index_repo")
        self.run(self._schedule(kind))

    @rule(
        kind=st.sampled_from(INDEXED), rebuild_outcome=st.sampled_from(REBUILD_OUTCOMES)
    )
    def optimize_keeps_failing(self, kind: str, rebuild_outcome: str) -> None:
        """Writes keep landing while every beat fails: reaches the fallback."""
        outcomes = ["fail"] * THRESHOLD
        self.trace.append(f"touch {kind} {outcomes} rebuild={rebuild_outcome}")
        self._touch_model(kind, outcomes, rebuild_outcome)
        self.run(self._schedule(kind))

    @rule(outcome=st.sampled_from(("ok", "conflict", "fail")))
    def heartbeat(self, outcome: str) -> None:
        """The heartbeat loop's tick: every handler kind through the gate."""
        self.trace.append(f"heartbeat {outcome}")
        for kind in self.handlers:
            if kind in self.m:
                self._touch_model(kind, [outcome], "ok")
        self.run(self._schedule(*self.handlers))

    @rule(kind=st.sampled_from(KINDS), outcome=st.sampled_from(REBUILD_OUTCOMES))
    def rebuild(self, kind: str, outcome: str) -> None:
        """The rebuild loop reached ``kind``: a sweep, or a due conflict retry."""
        self.trace.append(f"rebuild {kind} {outcome}")
        if kind in self.m:
            self._rebuild(kind, outcome, inside_runner=False)
        self.run(self.worker._run_rebuild_once(kind))
        self.run(self._settle())

    @rule(seconds=st.sampled_from(CLOCK_STEPS))
    def advance_clock(self, seconds: float) -> None:
        self.clock.offset += seconds
        self.trace.append(f"clock +{int(seconds)}")
        event(f"clock +{int(seconds)}")

    @rule(sweep=st.sampled_from(REBUILD_OUTCOMES))
    def restart(self, sweep: str) -> None:
        self.trace.append(f"restart sweep={sweep}")
        self.run(self.worker.stop())
        self.run(self._start(sweep))
        event(f"restart: sweep {sweep}")

    # ── invariants ─────────────────────────────────────────────────────────

    def _tail(self) -> str:
        return "\n".join(self.trace[-40:])

    @invariant()
    def repo_saw_what_the_spec_predicts(self) -> None:
        assert not self.violations, "\n".join(self.violations) + "\n" + self._tail()
        for kind, repo in self.repos.items():
            assert not repo.queue, (
                f"{kind}: calls never made: {repo.queue}\n{self._tail()}"
            )
            got, want = repo.calls, self.m[kind].expected
            if got != want:
                first = next(
                    (
                        i
                        for i, (g, w) in enumerate(zip(got, want, strict=False))
                        if g != w
                    ),
                    min(len(got), len(want)),
                )
                raise AssertionError(
                    f"{kind}: diverges at call {first}\n"
                    f"  repo saw:  {got[first - 3 : first + 4]}\n"
                    f"  expected:  {want[first - 3 : first + 4]}\n{self._tail()}"
                )

    @invariant()
    def dirty_only_after_a_runner_gave_up(self) -> None:
        states = self.worker._optimizer_states
        assert set(states) <= set(INDEXED), set(states)
        for kind, s in states.items():
            assert s.task is None and s.rebuild_task is None, kind
            assert s.dirty == self.m[kind].gave_up, (kind, s.dirty, self._tail())

    @invariant()
    def health_matches_model(self) -> None:
        h = self.worker.health()
        reasons = h.reasons()
        streak = max(km.failures for km in self.m.values())
        assert h.optimize_failure_streak == streak, (h, streak, self._tail())
        assert any("optimize stuck" in r for r in reasons) == (streak >= THRESHOLD)
        assert h.drain_consecutive_failures == 0 and h.unrecoverable_total == 0
        # staleness: the kind that has gone longest without a successful prune
        baselines = {
            kind: max(
                km.last_prune_ok if km.last_prune_ok is not None else float("-inf"),
                self.started_at,
            )
            for kind, km in self.m.items()
        }
        worst = min(baselines, key=baselines.__getitem__)
        expected = self._now() - baselines[worst]
        assert abs(h.prune_stale_seconds - expected) < TOL, (h, expected)
        ties = sum(1 for b in baselines.values() if abs(b - baselines[worst]) < TOL)
        if ties == 1:
            assert h.prune_stale_kind == worst, (h, baselines)
        stalled = any("cleanup stalled" in r for r in reasons)
        if abs(expected - ALERT) > TOL:
            assert stalled == (expected >= ALERT), (reasons, expected)

    @invariant()
    def rebuild_ladder_bookkeeping(self) -> None:
        for kind, s in self.worker._optimizer_states.items():
            km = self.m[kind]
            assert s.rebuild_attempt == km.rebuild_attempt, (kind, s, km)
            if km.retry_at == 0.0:
                assert s.rebuild_retry_at == 0.0, (kind, s.rebuild_retry_at)
            else:
                assert abs(s.rebuild_retry_at - km.retry_at) < TOL, (kind, s, km)

    @invariant()
    def beats_are_throttled(self) -> None:
        # The worker's clock is ``time.monotonic``; on a coarse clock
        # (Windows, 15.6 ms ticks) a timer may fire one tick early and a gap
        # may read one tick short, so the slack is two ticks on top of half
        # the interval. On a fine clock that is half the interval.
        slack = MIN_INTERVAL / 2 + 2 * time.get_clock_info("monotonic").resolution
        for kind, repo in self.repos.items():
            for a, b in itertools.pairwise(repo.beat_starts):
                assert b - a >= MIN_INTERVAL - slack, (kind, b - a, self._tail())


TestMaintenanceMachine = MaintenanceMachine.TestCase


def test_machine_catches_a_conflict_counted_as_a_failure() -> None:
    """Self-check: treat a commit conflict as a real failure and the health
    invariant must trip (the streak would pin high on a busy table)."""
    with mock.patch.object(worker_mod, "_is_benign_commit_conflict", lambda exc: False):
        machine = MaintenanceMachine()
        try:
            machine.touch("episode", ["conflict"], "ok")
            with pytest.raises(AssertionError):
                machine.health_matches_model()
        finally:
            machine.teardown()


def test_machine_catches_the_fallback_rebuild_zeroing_the_streak() -> None:
    """Self-check: the historical bug -- the fallback rebuild reset the
    failure counter, so the alert that fires at the threshold never fired."""
    real = CascadeWorker._run_rebuild_once

    async def resetting(self: CascadeWorker, kind: str) -> None:
        await real(self, kind)
        if kind in self._optimizer_states:
            self._optimizer_states[kind].optimize_failures = 0

    with mock.patch.object(CascadeWorker, "_run_rebuild_once", resetting):
        machine = MaintenanceMachine()
        try:
            for _ in range(THRESHOLD):
                machine.touch("episode", ["fail"], "ok")
            with pytest.raises(AssertionError):
                machine.health_matches_model()
        finally:
            machine.teardown()


# ── mode 2: the real loops on their own timers, raced by real writers ───────
#
# The machine above drives the subsystem one event at a time so that every
# call can be predicted. This mode gives up prediction for interleaving: the
# worker's three loops run on short real intervals (heartbeat, rebuild sweep
# with conflict retries, drain), several writer tasks land writes at random
# moments -- directly, as a drained batch does, or as md rows the drain loop
# picks up itself -- and the repository decides each call's fate at random
# (ok / conflict / fail / hang cut by the timeout / rebuild slower than the
# wait). What is checked does not depend on timing:
#
# - no two maintenance calls in flight on one kind's table, ever;
# - no call is cancelled unless the worker is stopping (a hang cut by the
#   timeout is the one expected cancellation) -- the wait that killed the
#   task it waited for shows up here as a cancelled rebuild;
# - the three supervised loops are alive until ``stop()`` -- a loop that
#   ends with a ``CancelledError`` nobody raised is the silent death;
# - every kind written to gets a beat after its last write (an optimizer
#   whose task slot never frees stops pruning that table silently);
# - consecutive beats of one kind are at least ``min_interval`` apart;
# - after ``stop()`` nothing is in flight and nothing starts;
# - ``health()`` reports the failure streak the repository's own log
#   implies for the final worker's life.
#
# It cannot shrink or replay (real time), so it keeps a millisecond trace.

_DEEP = os.environ.get("HYPOTHESIS_PROFILE") == "deep"
_J_PLAN_MAX = 40 if _DEEP else 16
_J_TIMEOUT = 0.1  # maintenance-call timeout (patched)
_J_SLOW = 0.25  # a rebuild longer than the wait, cut by nothing
_J_BACKOFFS = (0.05, 0.1, 0.2)  # rebuild conflict retry ladder (patched)
_J_MIN_INTERVAL = 0.05


class _RandomLanceRepo:
    """Decides each call's fate at random; records everything it saw."""

    def __init__(self, run: _JitterRun, kind: str, rng: random.Random) -> None:
        self.run = run
        self.kind = kind
        self.rng = rng
        self.calls: list[tuple[int, str, str]] = []  # (life, op, outcome)
        self.beat_starts: list[tuple[int, float]] = []  # (life, perf_counter)
        self.last_beat_at = -1.0
        self.active = 0

    async def optimize(self) -> None:
        await self._serve("optimize")

    async def prune(self, older_than: Any) -> None:
        await self._serve("prune")

    async def ensure_vector_indexes(self) -> list[str]:
        await self._serve("ensure")
        return []

    async def rebuild_indexes(self) -> None:
        await self._serve("rebuild")

    def _pick(self, op: str) -> str:
        r = self.run
        if op == "rebuild":
            return self.rng.choices(
                ("ok", "conflict", "fail", "slow"), weights=r.rebuild_weights
            )[0]
        return self.rng.choices(
            ("ok", "conflict", "fail", "hang"), weights=r.beat_weights
        )[0]

    async def _serve(self, op: str) -> None:
        r = self.run
        outcome = self._pick(op)
        if r.stopped:
            r.violations.append(f"{self.kind}: {op} started after stop() returned")
        self.active += 1
        if self.active > 1:
            r.violations.append(f"{self.kind}: {op} overlapped another call")
        self.calls.append((r.life, op, outcome))
        if op in ("optimize", "prune"):
            now = time.perf_counter()
            self.beat_starts.append((r.life, now))
            self.last_beat_at = now
        r.note(f"{self.kind}.{op} -> {outcome} (life {r.life})")
        try:
            if outcome == "hang":
                await asyncio.sleep(5.0)  # the timeout cuts this short
            else:
                await asyncio.sleep(
                    _J_SLOW if outcome == "slow" else self.rng.random() * 0.01
                )
            if outcome == "conflict":
                raise RuntimeError("Retryable commit conflict: preempted by a writer")
            if outcome == "fail":
                raise RuntimeError("index maintenance boom")
        except asyncio.CancelledError:
            if outcome != "hang" and not r.stopping:
                r.violations.append(f"{self.kind}: {op} was cancelled mid-flight")
            raise
        finally:
            self.active -= 1


class _MdRows:
    """The md queue: writers push rows, the real drain loop claims them."""

    def __init__(self, run: _JitterRun) -> None:
        self.run = run
        self.pending: list[Any] = []
        self.in_flight = 0

    async def claim_pending_batch(
        self, limit: int = 100, *, kinds: set[str] | None = None
    ) -> list[Any]:
        batch, self.pending = self.pending[:limit], self.pending[limit:]
        self.in_flight += len(batch)
        return batch

    async def mark_done(self, md_path: str) -> None:
        self.in_flight -= 1

    async def mark_failed(self, md_path: str, **kw: Any) -> None:
        self.in_flight -= 1
        self.run.violations.append(f"row {md_path} failed: {kw.get('error')}")


class _JitterRun:
    def __init__(self, seed: int, beat_weights: Any, rebuild_weights: Any) -> None:
        self.rng = random.Random(seed)
        self.beat_weights = beat_weights
        self.rebuild_weights = rebuild_weights
        self.life = 0
        self.stopping = False  # stop() called: cancellations are expected
        self.stopped = False  # stop() returned: nothing may start
        self.violations: list[str] = []
        self.trace: list[str] = []
        self.t0 = time.perf_counter()
        self.repos = {
            kind: _RandomLanceRepo(self, kind, random.Random(self.rng.random()))
            for kind in INDEXED
        }
        self.rows = _MdRows(self)
        self.last_touch_at: dict[str, float] = {}
        self.worker: CascadeWorker | None = None

    def note(self, what: str) -> None:
        self.trace.append(f"{(time.perf_counter() - self.t0) * 1000:7.1f}ms {what}")

    def dump(self) -> str:
        return "\n".join(self.trace[-80:])

    def new_worker(self) -> CascadeWorker:
        handlers: dict[str, Handler] = {
            kind: _IndexedHandler(self.repos[kind])
            for kind in INDEXED  # type: ignore[arg-type]
        }
        handlers["plain"] = _PlainHandler()
        self.worker = CascadeWorker(
            handlers,
            retry_backoff_seconds=0.0,
            poll_interval_seconds=0.02,
            optimize_min_interval_seconds=_J_MIN_INTERVAL,
            optimize_heartbeat_seconds=0.05,
            optimize_prune_interval_seconds=0.08,
            optimize_rebuild_interval_seconds=0.15,
        )
        return self.worker

    def loops_alive(self) -> list[str]:
        w = self.worker
        assert w is not None
        dead = []
        for name, task in (
            ("drain", w._task),
            ("heartbeat", w._heartbeat_task),
            ("rebuild", w._rebuild_task),
        ):
            if task is None or task.done():
                dead.append(f"{name}: {task}")
        return dead


async def _jittered_maintenance(
    plan: list[tuple[str, str, float]],
    seed: int,
    restart_at: float | None,
    beat_weights: tuple[int, int, int, int],
    rebuild_weights: tuple[int, int, int, int],
    quiet_tail: float = 0.0,
) -> None:
    """``plan``: (kind, "touch" | "row", pause before it) per writer step,
    split round-robin over three writers. ``restart_at`` (0..1) stops and
    restarts the worker that far through the plan, or never. ``quiet_tail``
    is how long the loops run on alone after the last write."""
    run = _JitterRun(seed, beat_weights, rebuild_weights)
    patches = [
        mock.patch.object(worker_mod, "md_change_state_repo", run.rows),
        mock.patch.object(worker_mod, "_MAINTENANCE_TASK_TIMEOUT_SECONDS", _J_TIMEOUT),
        mock.patch.object(
            worker_mod, "_REBUILD_CONFLICT_BACKOFFS_SECONDS", _J_BACKOFFS
        ),
    ]
    for p in patches:
        p.start()
    w = run.new_worker()
    await w.start()
    run.note("started")
    done_steps = 0
    restart_step = (
        None if restart_at is None else int(restart_at * max(1, len(plan) - 1))
    )
    reached_restart = asyncio.Event()
    try:

        async def writer(steps: list[tuple[str, str, float]]) -> None:
            nonlocal done_steps
            for kind, how, pause in steps:
                await asyncio.sleep(pause)
                assert run.worker is not None
                if how == "row":
                    run.rows.pending.append(
                        SimpleNamespace(
                            md_path=f"{kind}/{done_steps}.md",
                            kind=kind,
                            change_type="modified",
                            retry_count=0,
                        )
                    )
                else:
                    run.worker._schedule_optimize(kind)
                if kind in run.repos:
                    run.last_touch_at[kind] = time.perf_counter()
                run.note(f"write {kind} via {how}")
                done_steps += 1
                if restart_step is not None and done_steps >= restart_step:
                    reached_restart.set()

        async def restarter() -> None:
            if restart_step is None:
                return
            await reached_restart.wait()
            assert run.worker is not None
            dead = run.loops_alive()
            assert not dead, f"loop dead before restart: {dead}\n{run.dump()}"
            run.stopping = True
            run.note("restart: stopping")
            await run.worker.stop()
            run.stopped = True
            for repo in run.repos.values():
                assert repo.active == 0, run.dump()
            await asyncio.sleep(0.05)  # nothing may start after stop() returned
            # Rows the stop cancelled mid-batch stay ``processing`` in the real
            # table until the orchestrator's startup recovery re-queues them;
            # that recovery is the row-cycle test's subject, so here they are
            # simply forgotten.
            run.rows.in_flight = 0
            run.life += 1
            run.stopping = run.stopped = False
            await run.new_worker().start()
            run.note("restart: started")
            event("jitter: restarted mid-run")

        writers = [writer(plan[k::3]) for k in range(3)]
        await asyncio.gather(*writers, restarter())
        if quiet_tail:
            await asyncio.sleep(quiet_tail)  # heartbeat, prune cadence, sweeps

        # Trailing edge: every written kind gets a beat after its last write,
        # and the md rows all drain. Bounded: a stuck optimizer never does.
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            drained = not run.rows.pending and run.rows.in_flight == 0
            caught_up = all(
                run.repos[k].last_beat_at >= t for k, t in run.last_touch_at.items()
            )
            if drained and caught_up:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError(
                "never settled after the last write: "
                f"rows pending={len(run.rows.pending)} in_flight={run.rows.in_flight} "
                f"last_touch={run.last_touch_at} "
                f"last_beat={ {k: r.last_beat_at for k, r in run.repos.items()} }\n"
                + run.dump()
            )
        dead = run.loops_alive()
        assert not dead, f"loop dead: {dead}\n{run.dump()}"
        assert run.worker is not None
        h = run.worker.health()
        assert h.drain_consecutive_failures == 0, (h, run.dump())

        run.stopping = True
        run.note("stop")
        await run.worker.stop()
        run.stopped = True
        for kind, repo in run.repos.items():
            assert repo.active == 0, f"{kind} still in flight after stop\n{run.dump()}"
        states = run.worker._optimizer_states
        assert all(s.task is None and s.rebuild_task is None for s in states.values())
        await asyncio.sleep(0.15)  # nothing may start after stop (the fake checks)

        assert not run.violations, "\n".join(run.violations) + "\n" + run.dump()
        for kind, repo in run.repos.items():
            # Throttle, per worker life (slack for coarse clocks, see the
            # machine's invariant of the same name).
            slack = 0.02 + 2 * time.get_clock_info("monotonic").resolution
            starts = [t for life, t in repo.beat_starts if life == run.life]
            for a, b in itertools.pairwise(starts):
                assert b - a >= _J_MIN_INTERVAL - slack, (kind, b - a, run.dump())
            # The streak health reports is what this life's beats imply.
            beats: list[str] = []
            calls = [(op, o) for life, op, o in repo.calls if life == run.life]
            for i, (op, o) in enumerate(calls):
                if op == "optimize":
                    beats.append(o)
                elif op == "prune":
                    if o == "ok" and i + 1 < len(calls) and calls[i + 1][0] == "ensure":
                        beats.append(calls[i + 1][1])
                    else:
                        beats.append(o)
            streak = 0
            for o in reversed(beats):
                if o == "conflict":
                    continue
                if o == "ok":
                    break
                streak += 1
            # A kind the stopped sweep never reached has no state yet: streak 0.
            state = states.get(kind)
            got = state.optimize_failures if state is not None else 0
            assert got == streak, (kind, got, streak, beats[-8:], run.dump())
        reasons = h.reasons()
        assert any("optimize stuck" in r for r in reasons) == (
            h.optimize_failure_streak >= THRESHOLD
        )
        served = {o for repo in run.repos.values() for _, _, o in repo.calls}
        for o in ("hang", "conflict", "fail", "slow"):
            if o in served:
                event(f"jitter: {o} served")
        rebuilds = max(
            sum(1 for life, op, _ in repo.calls if op == "rebuild" and life == run.life)
            for repo in run.repos.values()
        )
        if rebuilds >= 2:
            event("jitter: a kind rebuilt twice in one life (sweep or retry)")
        n_calls = sum(len(r.calls) for r in run.repos.values())
        bucket = "<10" if n_calls < 10 else "10-29" if n_calls < 30 else "30+"
        event(f"jitter: calls {bucket}")
    finally:
        if run.worker is not None and run.worker._task is not None:
            run.stopping = True
            await run.worker.stop()
        for p in patches:
            p.stop()


@given(
    plan=st.lists(
        st.tuples(
            st.sampled_from(KINDS),
            st.sampled_from(("touch", "row")),
            st.sampled_from((0.0, 0.01, 0.03, 0.1)),
        ),
        min_size=1,
        max_size=_J_PLAN_MAX,
    ),
    seed=st.integers(0, 10_000),
    restart_at=st.one_of(st.none(), st.floats(0.0, 1.0)),
    quiet_tail=st.sampled_from((0.0, 0.2, 0.6)),
    beat_weights=st.sampled_from(
        ((6, 1, 1, 0), (3, 2, 3, 1), (1, 1, 6, 2), (2, 6, 1, 0))
    ),
    rebuild_weights=st.sampled_from(
        ((6, 1, 1, 0), (3, 3, 1, 2), (1, 1, 1, 6), (2, 6, 0, 1))
    ),
)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_jittered_maintenance_loops(
    plan: list[tuple[str, str, float]],
    seed: int,
    restart_at: float | None,
    beat_weights: tuple[int, int, int, int],
    rebuild_weights: tuple[int, int, int, int],
    quiet_tail: float,
) -> None:
    asyncio.run(
        _jittered_maintenance(
            plan, seed, restart_at, beat_weights, rebuild_weights, quiet_tail
        )
    )


def test_jitter_catches_the_wait_that_cancels() -> None:
    """Self-check: with the pre-fix wait (``asyncio.timeout`` around
    ``await task``) a slow startup rebuild plus one write kills the rebuild
    and the rebuild loop; the scout must report it."""

    class _CancellingWait:
        """``asyncio`` as the worker sees it, with the old wait semantics."""

        @staticmethod
        async def wait(fs: set[asyncio.Task[Any]], **kw: float) -> Any:
            (task,) = fs
            async with asyncio.timeout(kw["timeout"]):
                await task
            return {task}, set()

        def __getattr__(self, name: str) -> Any:
            return getattr(asyncio, name)

    plan = [("episode", "touch", 0.03)] * 6
    with (
        mock.patch.object(worker_mod, "asyncio", _CancellingWait()),
        pytest.raises(AssertionError, match=r"cancelled mid-flight|loop dead"),
    ):
        asyncio.run(_jittered_maintenance(plan, 1, None, (1, 0, 0, 0), (0, 0, 0, 1)))
