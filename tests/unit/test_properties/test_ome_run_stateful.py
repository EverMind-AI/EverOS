"""Model-based test of an OME strategy run's lifecycle: attempts, retries, recovery.

The subject is the chain ``emit → dispatch → Runner → RunRecordStore`` of the
offline memory engine, run for real: a real :class:`OfflineEngine` on a temp
SQLite jobstore (through :class:`StrategyTestHarness`), APScheduler, the
run-record store, the attempt loop with its retry and dead-letter rules,
``wait_for_event``, ``stop`` / ``start`` with crash recovery. Only the
strategy body is scripted: the event carries a plan and the body follows it
(succeed; fail once then succeed; fail every time; break the emit contract;
hang past the run timeout).

Rules: emit a batch of jobs and drain (they run concurrently under the
engine's semaphore), emit one job and ``wait_for_event`` on it, leave a
stale ``RUNNING`` row behind as a crashed process would and restart, and a
plain restart.

Invariants after every rule, from the runner's and store's docstrings:

1. an idle engine has no ``RUNNING`` row and no in-flight count;
2. every job's rows are exactly the attempt chain its plan implies:
   ``SUCCESS`` at attempt 0; ``FAILED`` at 0 then ``SUCCESS`` at 1; ``FAILED``
   at 0 and 1 then ``DEAD_LETTER`` at 2 (``max_retries = 2``); a contract
   violation is ``DEAD_LETTER`` at attempt 0 with no retry; a timeout counts
   as a failure and retries like one; a crashed row is marked ``CRASHED``
   and the resumed chain starts again at attempt 0;
3. the strategy body ran exactly as many times as that chain has attempts;
4. every non-success row carries an error, every row the retry snapshot;
5. ``wait_for_event`` returns the chain's final rows, never a prefix of it.

The ring-buffer trim of the run-record table is left to its own unit test:
``max_records_per_strategy`` stays above anything one sequence can write.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import threading
import time
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import pytest
from hypothesis import HealthCheck, currently_in_test_context, settings
from hypothesis import event as _hypothesis_event
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from everos.core.utils.datetime import get_utc_now, to_iso_format
from everos.infra.ome import RunStatus
from everos.infra.ome._stores import storage as storage_mod
from everos.memory.deps import MemoryDeps
from everos.memory.ports import Event, Immediate, StrategySpec
from tests.fakes import make_deps
from tests.helpers.ome import StrategyTestHarness, strategy_spec

settings.register_profile(
    "ci",
    max_examples=25,
    stateful_step_count=12,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "deep",
    max_examples=500,
    stateful_step_count=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

PLANS = ("ok", "flaky", "doomed", "contract", "slow")
MAX_RETRIES = 2
RUN_TIMEOUT = 0.1
S, F, D, C = (
    RunStatus.SUCCESS,
    RunStatus.FAILED,
    RunStatus.DEAD_LETTER,
    RunStatus.CRASHED,
)
CHAIN: dict[str, list[tuple[RunStatus, int]]] = {
    "ok": [(S, 0)],
    "flaky": [(F, 0), (S, 1)],
    "doomed": [(F, 0), (F, 1), (D, 2)],
    "contract": [(D, 0)],
    "slow": [(F, 0), (F, 1), (D, 2)],
}


def event(label: str) -> None:
    if currently_in_test_context():
        _hypothesis_event(label)


class _Job(Event):
    plan: str


class _Undeclared(Event):
    """Never in ``emits``: emitting it is a contract violation."""


_calls: dict[str, int] = defaultdict(int)

# Diagnostics for the intermittent Windows red: every OME connection is
# registered while open (when, from where), so a drain timeout can say which
# connections were still open and for how long.
_open_connections: dict[int, tuple[float, str]] = {}
_real_connect = storage_mod.OMEStorage.connect


@asynccontextmanager
async def _registered_connect(self: Any) -> Any:
    key = id(object())
    origin = "".join(traceback.format_stack(limit=8)[:-1])
    _open_connections[key] = (time.monotonic(), origin)
    try:
        async with _real_connect(self) as conn:
            yield conn
    finally:
        _open_connections.pop(key, None)


storage_mod.OMEStorage.connect = _registered_connect  # type: ignore[method-assign]


@strategy_spec(
    name="scripted",
    trigger=Immediate(on=[_Job]),
    emits=[],
    max_retries=MAX_RETRIES,
)
async def _scripted(event: Event, deps: MemoryDeps) -> list[Event]:
    assert isinstance(event, _Job)
    _calls[event.event_id] += 1
    n = _calls[event.event_id]
    if event.plan == "ok" or (event.plan == "flaky" and n > 1):
        return []
    if event.plan == "contract":
        return [_Undeclared()]
    if event.plan == "slow":
        await asyncio.sleep(5.0)  # the run timeout cuts this short
    raise RuntimeError(f"{event.plan}: attempt {n} fails")


class OmeRunMachine(RuleBasedStateMachine):
    SPEC: StrategySpec = _scripted
    """What the engine runs; the model always expects ``_scripted``'s budget."""

    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        _calls.clear()
        self.h = StrategyTestHarness()
        self.h._engine._config.run_timeout_seconds = RUN_TIMEOUT
        # The ring-buffer trim is the store's own unit test's subject; keep
        # it out of reach of anything one sequence can write (a deep
        # sequence writes a few hundred rows, the default cap is 100).
        self.h._engine._config.max_records_per_strategy = 100_000
        self.h.register(self.SPEC, make_deps())
        self.run(self.h.start())
        self.expected: dict[str, list[tuple[RunStatus, int]]] = {}
        self.plan_of: dict[str, str] = {}
        self.trace: list[str] = []

    def run(self, coro: Any) -> Any:
        return self.loop.run_until_complete(coro)

    def teardown(self) -> None:
        try:
            self.run(self.h.__aexit__(None, None, None))
        finally:
            self.loop.close()

    @property
    def engine(self) -> Any:
        return self.h._engine

    def _drain(self) -> None:
        """Drain, and when that times out say what was still in flight.

        Windows CI timed out here with one run in flight and only an APS log
        line (``database is locked`` after the 5 s busy wait) to go on.
        """
        try:
            self.run(self.h.drain(timeout=20))
        except TimeoutError as e:
            try:
                running = self.run(self.engine._run_record_store.find_running())
                rows = [
                    (
                        self.plan_of.get(r.event_id, "?"),
                        r.run_id[:8],
                        r.attempt,
                        r.started_at,
                    )
                    for r in running
                ]
            except Exception as read_error:
                rows = [f"find_running failed: {read_error!r}"]
            threads = sorted(t.name for t in threading.enumerate())
            now = time.monotonic()
            held = "\n".join(
                f"--- open for {now - opened:.1f}s, from:\n{origin}"
                for opened, origin in _open_connections.values()
            )
            raise TimeoutError(
                f"{e}; active_runs={self.engine._in_flight.active}; running={rows}; "
                f"threads={threads}; trace={self.trace[-8:]}\n"
                f"open connections ({len(_open_connections)}):\n{held}"
            ) from e

    # ── rules ──────────────────────────────────────────────────────────────

    @rule(plans=st.lists(st.sampled_from(PLANS), min_size=1, max_size=4))
    def emit_and_drain(self, plans: list[str]) -> None:
        jobs = [_Job(plan=p) for p in plans]
        for job in jobs:
            self.expected[job.event_id] = list(CHAIN[job.plan])
            self.plan_of[job.event_id] = job.plan
            self.trace.append(f"emit {job.plan} {job.event_id[:8]}")
        self.run(self._emit_all(jobs))
        self._drain()
        for p in plans:
            event(f"run: {p}")

    async def _emit_all(self, jobs: list[_Job]) -> None:
        await asyncio.gather(*(self.h.emit(job) for job in jobs))

    @rule(plan=st.sampled_from(PLANS))
    def emit_and_wait(self, plan: str) -> None:
        job = _Job(plan=plan)
        self.expected[job.event_id] = list(CHAIN[plan])
        self.plan_of[job.event_id] = plan
        self.trace.append(f"emit+wait {plan} {job.event_id[:8]}")
        self.run(self.h.emit(job))
        returned = self.run(
            self.engine.wait_for_event(job.event_id, timeout=20, poll_interval=0.01)
        )
        self._drain()
        final = self.run(self.engine.list_runs_by_event_id(job.event_id))
        got = sorted((r.status, r.attempt) for r in returned)
        want = sorted((r.status, r.attempt) for r in final)
        assert got == want, f"wait_for_event returned a prefix: {got} != {want}"
        event(f"wait: {plan}")

    @rule(plan=st.sampled_from(PLANS))
    def crash_and_restart(self, plan: str) -> None:
        """A process died mid-run and left a RUNNING row, seconds old. On
        restart the row is CRASHED and the event re-enqueued (#540: age is
        no longer a condition)."""
        job = _Job(plan=plan)
        stale_run_id = f"stale-{job.event_id}"
        self.run(self._insert_stale_running(job, stale_run_id, 1.0))
        self.expected[job.event_id] = [(C, 0), *CHAIN[plan]]
        self.plan_of[job.event_id] = plan
        self.trace.append(f"crash {plan} {job.event_id[:8]}")
        self.run(self.engine.stop())
        self.run(self.engine.start())
        self._drain()
        event(f"crash: {plan}")

    async def _insert_stale_running(self, job: _Job, run_id: str, age: float) -> None:
        started = to_iso_format(get_utc_now() - timedelta(seconds=age))
        async with self.engine._storage.transaction() as conn:
            await conn.execute(
                "INSERT INTO run_record (run_id, strategy_name, status, attempt, "
                "started_at, event_topic, event_payload, max_retries_snapshot, "
                "event_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    "scripted",
                    RunStatus.RUNNING.value,
                    0,
                    started,
                    _Job.topic(),
                    job.model_dump_json(),
                    MAX_RETRIES,
                    job.event_id,
                ),
            )

    @rule()
    def restart(self) -> None:
        self.trace.append("restart")
        self.run(self.engine.stop())
        self.run(self.engine.start())
        event("restart")

    # ── invariants ─────────────────────────────────────────────────────────

    @invariant()
    def idle_engine_has_nothing_running(self) -> None:
        assert self.engine._in_flight.active == 0
        running = self.run(self.engine._run_record_store.find_running())
        assert not running, [r.run_id for r in running]

    @invariant()
    def every_job_has_exactly_its_attempt_chain(self) -> None:
        for event_id, chain in self.expected.items():
            rows = self.run(self.engine.list_runs_by_event_id(event_id))
            got = sorted((r.status, r.attempt) for r in rows)
            assert got == sorted(chain), (
                self.plan_of[event_id],
                got,
                chain,
                "\n".join(self.trace[-10:]),
            )
            for r in rows:
                assert r.max_retries_snapshot == MAX_RETRIES
                assert r.event_topic == _Job.topic()
                assert (r.error is not None) == (r.status != S), (r.status, r.error)
                assert (r.finished_at is not None) == (r.status != RunStatus.RUNNING)

    @invariant()
    def the_body_ran_once_per_attempt(self) -> None:
        for event_id, chain in self.expected.items():
            attempts = sum(1 for status, _ in chain if status != C)
            assert _calls[event_id] == attempts, (
                self.plan_of[event_id],
                _calls[event_id],
            )

    @invariant()
    def no_stray_rows(self) -> None:
        # Straight from the store with an explicit limit: the harness's
        # list_runs caps at 100 rows, which a deep run read as "a row went
        # missing" once a sequence had written more than that.
        rows = self.run(
            self.engine._run_record_store.list_runs(
                strategy_name="scripted", limit=1_000_000
            )
        )
        assert len(rows) == sum(len(c) for c in self.expected.values())


TestOmeRunMachine = OmeRunMachine.TestCase


class _NoRetryMachine(OmeRunMachine):
    """The engine makes every failed attempt terminal."""

    SPEC = dataclasses.replace(_scripted, max_retries=0)


def test_machine_catches_a_dropped_retry() -> None:
    """Self-check: an engine that drops the retry of a failed attempt must
    trip the attempt-chain invariant on a flaky job."""
    machine = _NoRetryMachine()
    try:
        machine.emit_and_drain(["flaky"])
        with pytest.raises(AssertionError):
            machine.every_job_has_exactly_its_attempt_chain()
    finally:
        machine.teardown()
