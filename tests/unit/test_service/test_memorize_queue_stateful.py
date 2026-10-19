"""Model-based test of the ``async_mode`` queue: hypothesis drives the worker.

The unit tests next door pin the sequences someone thought of. This file
lets ``hypothesis`` generate thousands of operation sequences against the
real inbox repo (SQLite on a temp file) and the real worker, with only three
things replaced: ``memorize()`` (the LLM pipeline), the clock, and the one
second pause between status-write retries.

How the pieces divide the work (the pattern to copy for another component):

- **Rules** are only what the outside world can do to the system: enqueue,
  a drain whose rows succeed / fail / die mid-``memorize()`` (optionally
  with requests arriving *while* a row is inside ``memorize()``, or a status
  write that fails all retries), restart, sweep, moving the clock. Internal
  chains (wake -> claim -> task -> mark -> claim again) are left to the real
  code; the rule parameters exist to create the situations where those
  chains interleave.
- **The model** (``self.model``) is the spec's state machine (plan §4.4)
  written as assignments: what each row *should* be after each step. It is
  computed from the inputs, never read back from the table, and never copies
  the implementation -- that is what makes the comparison a check.
- **Invariants** run after every step and compare the real table with the
  model:

  1. no ``processing`` row while the process is alive and idle; while rows
     run, never two of one session and never more than ``max_concurrency``;
  2. a session's rows reach ``memorize()`` in arrival order, and a row starts
     only when no earlier row of its session is still queued; a replay after
     a crash repeats the crashed row, it never skips ahead;
  3. ``success`` / ``failed`` never change again except by purge; a delivered
     row has no body, a failed row keeps it;
  4. the caps hold: 429 is raised exactly when the model says the session or
     the process is full (asserted between steps; during a drain the table
     is a moving target and the product's verdict is taken as the fact);
  5. ``GET /tasks/{task_id}`` reports what the table holds.

Concurrency is the product's own: the worker creates one asyncio task per
row. The test only decides what happens at the moment a row is inside
``memorize()`` -- return, raise, let a request arrive (a real insert, so a
real await), or die -- and keeps that moment deterministic (a crash waits
until it is the only row in flight; outcomes are keyed by request id, not by
call order), so every failing sequence shrinks and replays.

Three tests:

- ``TestMemorizeQueueMachine`` -- the rule-based machine above. Deterministic,
  replayable, shrinking; a failure prints the shortest sequence plus the
  machine's own trace of what happened.
- ``test_jittered_real_tasks`` -- the scout: real producers racing the real
  worker loop with random pauses, slow memorize(), lost status writes and a
  mid-run stop()/start(). True interleaving, not replayable; a failure prints
  a millisecond trace so the interleaving can be rebuilt as a machine
  sequence (``mid=`` puts "who arrived during whose memorize()" into words).
- ``test_machine_catches_a_lost_replay`` -- the self-check: inject one bug and
  pin that an invariant trips. A machine that cannot fail proves nothing.

``--hypothesis-show-statistics`` lists how many examples hit each branch
(crash, lost write, enqueue during memorize(), replay, purge, restart);
a branch at 0% was not exercised. ``HYPOTHESIS_PROFILE=deep`` runs 1000
sequences, 1000 jittered runs with plans up to 30 requests and memorize()
up to 0.3 s; CI runs 60 of each in about 15 s.
"""

from __future__ import annotations

import asyncio
import os
import random
import shutil
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from hypothesis import HealthCheck, currently_in_test_context, given, settings
from hypothesis import event as _hypothesis_event
from hypothesis import strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
)
from sqlalchemy import select
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.config.settings import AsyncQueueSettings
from everos.core.errors import QueueSaturatedError
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite.repos.memorize_inbox import _MemorizeInboxRepo
from everos.infra.persistence.sqlite.tables import MemorizeInbox
from everos.service import memorize_queue as mq
from everos.service.memorize import MemorizeResult

settings.register_profile(
    "ci",
    max_examples=60,
    stateful_step_count=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "deep",
    max_examples=1000,
    stateful_step_count=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))

SESSIONS = ("s1", "s2", "s3")
MAX_CONCURRENCY = 2
PER_SESSION_CAP = 2
TOTAL_CAP = 4
RETENTION_HOURS = 2.0
LOST_WRITE_ERROR = "status write lost; outcome unknown"
MAX_REPLAYS = 3  # the settings default; a crashed row is replayed this often
REPLAYS_EXHAUSTED_ERROR = f"replayed {MAX_REPLAYS} times without finishing; giving up"
# Clock steps whose sums never land within minutes of the retention edge,
# so model and worker agree on what is expired.
CLOCK_STEPS_HOURS = (0.3, 3.0)
PENDING = ("queued", "processing")


def event(label: str) -> None:
    """Count a rule outcome in ``--hypothesis-show-statistics``.

    The self-check test drives the machine by hand, outside hypothesis,
    where ``hypothesis.event`` refuses to record.
    """
    if currently_in_test_context():
        _hypothesis_event(label)


class _Crash(BaseException):
    """The process dies inside ``memorize()``.

    Not an ``Exception``, so the worker's failure path does not run and the
    row stays ``processing`` -- exactly what a ``kill -9`` leaves behind.
    """


def _settings(**overrides: Any) -> AsyncQueueSettings:
    base: dict[str, Any] = {
        "max_concurrency": MAX_CONCURRENCY,
        "max_pending_per_session": PER_SESSION_CAP,
        "max_pending_total": TOTAL_CAP,
        "row_retention_hours": RETENTION_HOURS,
        "retry_backoff_seconds": 0.0,
        "scan_interval_seconds": 60.0,
    }
    base.update(overrides)
    return AsyncQueueSettings(**base)


def _payload(session: str, rid: str) -> dict[str, Any]:
    # ``rid`` rides inside the body so the fake memorize knows which row it is.
    return {
        "session_id": session,
        "app_id": "default",
        "project_id": "default",
        "messages": [{"content": rid}],
        "rid": rid,
    }


class _Queue:
    """One real queue on one temp SQLite file; module collaborators patched."""

    def __init__(self, memorize: Any, cfg: AsyncQueueSettings) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="mq_stateful_"))
        root = MemoryRoot(self.tmp)
        root.ensure()
        self.engine = create_system_engine(root.system_db, SqliteSettings())
        self.factory = create_session_factory(self.engine)
        self.repo = _MemorizeInboxRepo(session_factory=self.factory)
        self.clock_offset = timedelta()
        self.worker = mq.MemorizeQueueWorker(cfg)
        self._patches = [
            mock.patch.object(mq, "memorize_inbox_repo", self.repo),
            mock.patch.object(mq, "memorize", memorize),
            mock.patch.object(mq, "get_utc_now", self.now),
            mock.patch.object(mq, "_MARK_RETRY_SECONDS", 0.0),
        ]

    def now(self) -> datetime:
        return datetime.now(UTC) + self.clock_offset

    async def setup(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        for p in self._patches:
            p.start()

    async def teardown(self) -> None:
        for p in self._patches:
            p.stop()
        await self.engine.dispose()
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def rows(self) -> list[MemorizeInbox]:
        async with self.factory() as s:
            stmt = select(MemorizeInbox).order_by(MemorizeInbox.id)  # type: ignore[arg-type]
            return list((await s.execute(stmt)).scalars().all())


# ── mode 1: the rule-based machine ─────────────────────────────────────────


class MemorizeQueueMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.q = _Queue(self._fake_memorize, _settings())
        self.run(self.q.setup())
        # The model: request id -> expected state, plus the delivery log as
        # (session, arrival number). Purge lets SQLite reuse row ids, so the
        # arrival number, not the id, is what delivery order is checked on.
        self.model: dict[str, dict[str, Any]] = {}
        self.seen: set[str] = set()
        self.deliveries: list[tuple[str, int]] = []
        self.requeued: dict[str, int] = defaultdict(int)
        self.n = 0
        self.crashed = False
        # Outcome per request id, fixed when the drain starts: with rows
        # running side by side, "the k-th memorize() call" is timing.
        self.outcome_of: dict[str, str] = {}
        self.mid_events: list[tuple[str, bool]] = []
        self.lose_marks = 0
        self.lose_rid: str | None = None
        self.lose_row_id: int | None = None
        self.die_after_loss = False
        self.violations: list[str] = []
        self.trace: list[str] = []  # what happened, for a failure's message
        self._real_mark = self.q.repo.mark
        self.q.repo.mark = self._mark_maybe_lost  # type: ignore[method-assign]

    def run(self, coro: Any) -> Any:
        return self.loop.run_until_complete(coro)

    def teardown(self) -> None:
        self.run(self.q.teardown())
        self.loop.close()

    # ── the doubles ────────────────────────────────────────────────────────

    async def _fake_memorize(
        self,
        payload: dict[str, Any],
        *,
        is_final: bool = False,
        defer_extraction: bool = False,
    ) -> MemorizeResult:
        rid = payload["rid"]
        m = self.model[rid]
        session = payload["session_id"]
        self.deliveries.append((session, m["seq"]))
        self.trace.append(
            f"memorize({rid}/{session}) in_flight={sorted(self.q.worker._in_flight)}"
        )
        # claim_next flipped the row before calling us; a crash leaves it so.
        m["status"] = "processing"
        self._observe_scheduler(rid, session)
        # Requests that arrive while this row is inside memorize(): the only
        # interleaving a single event loop allows, and the one that matters.
        events, self.mid_events = self.mid_events, []
        for mid_session, final in events:
            await self._enqueue_async(mid_session, final, in_memorize=True)
        outcome = self.outcome_of.get(rid, "ok")
        if outcome == "crash":
            # The whole process dies, so nothing else may be running or about
            # to be claimed when it does: let the other rows in flight finish
            # and let the scheduler drain every other session first, which
            # keeps the table state after a crash deterministic.
            await self._wait_until_alone(rid, session)
        if self.lose_rid == rid and self.die_after_loss:
            # The status write is lost AND the process dies before the sweep
            # can reap it: the row stays processing, the next start() requeues
            # it and it is replayed (the documented at-least-once case).
            await self._wait_until_alone(rid, session)
            outcome = "fail" if outcome == "fail" else "ok"
        elif self.lose_rid == rid:
            # Whatever happens next, the status write is lost: the sweep
            # fails the row and the body stays.
            m["status"], m["error"], m["body"] = "failed", LOST_WRITE_ERROR, True
            outcome = "fail" if outcome == "fail" else "ok"
        self.trace.append(f"  {rid} -> {outcome}")
        if outcome == "crash":
            raise _Crash
        if outcome == "fail":
            if self.lose_rid != rid:
                m["status"], m["body"] = "failed", True
                m["error"] = "RuntimeError: boom"  # what _process stores
            raise RuntimeError("boom")
        if self.lose_rid != rid:
            m["status"], m["body"] = "success", False
        return MemorizeResult(message_count=1, status="extracted")

    def _observe_scheduler(self, rid: str, session: str) -> None:
        """Check the scheduler's rules at the moment a row starts."""
        in_flight = [row for row, _ in self.q.worker._in_flight.values()]
        sessions = [row.session_id for row in in_flight]
        if len(sessions) != len(set(sessions)):
            self.violations.append(f"{rid}: two rows of one session in flight")
        if len(in_flight) > MAX_CONCURRENCY:
            self.violations.append(f"{rid}: {len(in_flight)} rows in flight")
        seq = self.model[rid]["seq"]
        earlier = [
            r
            for r, m in self.model.items()
            if m["session"] == session and m["status"] == "queued" and m["seq"] < seq
        ]
        if earlier:
            self.violations.append(f"{rid} started before {earlier} of its session")

    async def _wait_until_alone(self, rid: str, session: str) -> None:
        deadline = time.perf_counter() + 30.0  # a hang must fail, not freeze
        while time.perf_counter() < deadline:
            others = [
                task
                for row, task in self.q.worker._in_flight.values()
                if row.request_id != rid and not task.done()
            ]
            claimable = any(
                m["status"] == "queued" and m["session"] != session
                for m in self.model.values()
            )
            if others:
                await asyncio.wait(others, timeout=1.0)
            elif len(self.q.worker._in_flight) > 1 or claimable:
                await asyncio.sleep(0)  # let the scheduler collect / claim
            else:
                return
        raise RuntimeError(f"{rid}: other rows never left the scheduler")

    async def _mark_maybe_lost(self, row_id: int, **values: Any) -> None:
        # Only the chosen row's writes are lost; with two rows in flight a
        # budget shared by both would land on whichever marks first.
        if self.lose_marks and row_id == self.lose_row_id:
            self.lose_marks -= 1
            if self.lose_marks == 0 and self.die_after_loss:
                raise _Crash  # kill -9 lands between the lost write and the sweep
            raise RuntimeError("database is locked")
        await self._real_mark(row_id, **values)

    # ── model helpers ──────────────────────────────────────────────────────

    def _pending(self, session: str | None = None) -> int:
        return sum(
            1
            for m in self.model.values()
            if m["status"] in PENDING and (session is None or m["session"] == session)
        )

    def _enqueue(self, session: str, is_final: bool) -> None:
        self.run(self._enqueue_async(session, is_final))

    async def _enqueue_async(
        self, session: str, is_final: bool, *, in_memorize: bool = False
    ) -> None:
        rid, seq = f"r{self.n}", self.n
        self.n += 1
        full = self._pending(session) >= PER_SESSION_CAP or self._pending() >= TOTAL_CAP
        # Register the row before the insert: the scheduler may claim it and
        # call memorize() for it before enqueue() even returns.
        self.model[rid] = {
            "seq": seq,
            "row_id": None,
            "session": session,
            "status": "queued",
            "body": True,
            "error": None,
            "created_at": None,
        }
        try:
            pending = await self.q.worker.enqueue(
                _payload(session, rid),
                request_id=rid,
                is_final=is_final,
                defer_extraction=False,
            )
        except QueueSaturatedError:
            del self.model[rid]
            self.trace.append(f"enqueue({rid}/{session}) -> 429 model_full={full}")
            # Between steps nothing moves, so the model's verdict is exact.
            # During a drain other rows finish while enqueue() awaits its
            # count, so the model's snapshot is stale by the time the table
            # is read: take the product's decision as the fact.
            assert full or in_memorize, f"429 for {rid} but the model had room"
            event("enqueue: 429")
            return
        self.trace.append(f"enqueue({rid}/{session}) -> accepted model_full={full}")
        assert not full or in_memorize, f"{rid} accepted but the model says full"
        event("enqueue: accepted")
        row = await self.q.repo.find_one(request_id=rid)
        assert row is not None and row.id is not None
        if not in_memorize:
            # Between steps nothing runs, so the depth enqueue() reported is
            # exactly the model's; during a drain it is a moving target.
            assert pending == self._pending(session)
        self.seen.add(rid)
        self.model[rid]["row_id"] = row.id
        self.model[rid]["created_at"] = row.created_at

    def _model_purge(self) -> None:
        before = self.q.now() - timedelta(hours=RETENTION_HOURS)
        for rid in [
            rid
            for rid, m in self.model.items()
            if m["status"] != "processing" and m["created_at"] < before
        ]:
            del self.model[rid]

    # ── rules ──────────────────────────────────────────────────────────────

    @precondition(lambda self: not self.crashed)
    @rule(session=st.sampled_from(SESSIONS), is_final=st.booleans())
    def enqueue(self, session: str, is_final: bool) -> None:
        self._enqueue(session, is_final)

    @precondition(lambda self: not self.crashed)
    @rule(
        outcomes=st.lists(st.sampled_from(["ok", "fail", "crash"]), max_size=4),
        mid=st.lists(st.tuples(st.sampled_from(SESSIONS), st.booleans()), max_size=2),
        lose_write=st.booleans(),
        die_after_loss=st.booleans(),
    )
    def drain(
        self,
        outcomes: list[str],
        mid: list[tuple[str, bool]],
        lose_write: bool,
        die_after_loss: bool,
    ) -> None:
        """One ``drain_once``: rows deliver in id order with these outcomes."""
        queued = sorted(
            (m["row_id"], rid)
            for rid, m in self.model.items()
            if m["status"] == "queued"
        )
        # One crash per drain: the first one ends the process.
        if "crash" in outcomes:
            first = outcomes.index("crash")
            outcomes = outcomes[: first + 1] + [
                "ok" if o == "crash" else o for o in outcomes[first + 1 :]
            ]
        if lose_write and die_after_loss and queued:
            # The lost-write row's death is this drain's crash. Two rows both
            # waiting to be the only one in flight before dying would wait
            # forever, so no other row crashes in this drain.
            outcomes = ["ok" if o == "crash" else o for o in outcomes]
        # Outcomes go to the queued rows in id order; rows enqueued during
        # this drain succeed.
        self.outcome_of = {
            rid: outcome for (_, rid), outcome in zip(queued, outcomes, strict=False)
        }
        self.mid_events = list(mid)
        self.trace.append(
            f"drain outcomes={self.outcome_of} mid={mid} lose={self.lose_rid} "
            f"model={ {r: m['status'] for r, m in self.model.items()} }"
        )
        for outcome in self.outcome_of.values():
            event(f"drain: row {outcome}")
        if mid and queued:
            event("drain: enqueue during memorize()")
        if lose_write and queued:
            # The first row's status write fails all three attempts.
            self.lose_marks = 3
            self.lose_row_id, self.lose_rid = queued[0]
            self.die_after_loss = die_after_loss
            event(
                "drain: status write lost, then crash"
                if die_after_loss
                else "drain: status write lost"
            )
        try:
            self.run(self.q.worker.drain_once())
        except _Crash:
            self.crashed = True
        finally:
            self.lose_marks, self.lose_rid, self.lose_row_id = 0, None, None
            self.die_after_loss = False
            self.outcome_of, self.mid_events = {}, []
        if not self.crashed:
            # A drain only stops when nothing is queued, and the lost-write
            # row was reaped before the next claim or after the loop.
            pending = {
                rid: m["status"]
                for rid, m in self.model.items()
                if m["status"] in PENDING
            }
            table = {
                r.request_id: (r.status, r.last_error) for r in self.run(self.q.rows())
            }
            assert not pending, (
                f"model pending={pending} table={table} "
                f"in_flight={list(self.q.worker._in_flight)}\n"
                + "\n".join(self.trace[-40:])
            )

    @rule()
    def restart(self) -> None:
        """What ``start()`` does: requeue the in-flight row, then sweep."""
        self.trace.append("restart")
        requeued, gave_up = self.run(
            self.q.repo.requeue_processing(max_replays=MAX_REPLAYS)
        )
        in_flight = [
            (rid, m) for rid, m in self.model.items() if m["status"] == "processing"
        ]
        poison = [rid for rid, _ in in_flight if self.requeued[rid] >= MAX_REPLAYS]
        assert (requeued, gave_up) == (len(in_flight) - len(poison), len(poison))
        if poison:
            event("restart: gave up on a poison row")
        elif in_flight:
            event("restart: replayed a crashed row")
        else:
            event("restart: clean")
        for rid, m in in_flight:
            if rid in poison:
                # Replayed MAX_REPLAYS times without finishing: failed, body kept.
                m["status"], m["error"], m["body"] = (
                    "failed",
                    REPLAYS_EXHAUSTED_ERROR,
                    True,
                )
            else:
                m["status"] = "queued"
                self.requeued[rid] += 1
        self.q.worker = mq.MemorizeQueueWorker(_settings())
        self.run(self.q.worker._sweep())
        self._model_purge()
        self.crashed = False

    @precondition(lambda self: not self.crashed)
    @rule()
    def sweep(self) -> None:
        self.trace.append("sweep")
        before = len(self.model)
        self.run(self.q.worker._sweep())
        self._model_purge()
        event("sweep: purged rows" if len(self.model) < before else "sweep: nothing")

    @rule(hours=st.sampled_from(CLOCK_STEPS_HOURS))
    def advance_clock(self, hours: float) -> None:
        self.q.clock_offset += timedelta(hours=hours)

    # ── invariants (plan §4.4) ─────────────────────────────────────────────

    @invariant()
    def table_matches_model(self) -> None:
        rows = self.run(self.q.rows())
        assert {r.request_id: r.status for r in rows} == {
            rid: m["status"] for rid, m in self.model.items()
        }
        for r in rows:
            m = self.model[r.request_id]
            assert (r.payload_json != "") == m["body"], (r.request_id, r.status)
            if m["error"] is not None:
                assert r.last_error == m["error"]
            if r.status == "success":
                assert r.last_error is None, r.request_id
            assert r.updated_at >= r.created_at, r.request_id

    @invariant()
    def delivered_at_most_once_per_life(self) -> None:
        # A row reaches memorize() once per time it was (re)queued: never
        # twice in one life, never silently skipped on a replay.
        seq_to_rid = {m["seq"]: rid for rid, m in self.model.items()}
        seen: dict[str, int] = defaultdict(int)
        for _session, seq in self.deliveries:
            if seq in seq_to_rid:
                seen[seq_to_rid[seq]] += 1
        for rid, n in seen.items():
            assert n <= 1 + self.requeued[rid], (rid, n, self.requeued[rid])

    @invariant()
    def backlog_readers_agree(self) -> None:
        # The route's backlog rule and GET /tasks read the same table the
        # model describes.
        for session in SESSIONS:
            expected = self._pending(session) > 0
            got = self.run(self.q.worker.has_pending(session_id=session))
            assert got == expected, (session, got, expected)
            page = self.run(mq.list_tasks(session_id=session, page_size=100))
            want = defaultdict(int)
            for m in self.model.values():
                if m["session"] == session:
                    want[m["status"]] += 1
            assert page.total == sum(want.values()), (session, page.total, want)
            seen = defaultdict(int)
            for item in page.items:
                seen[item.status] += 1
            assert dict(seen) == dict(want), (session, dict(seen), dict(want))

    @invariant()
    def at_most_one_in_flight(self) -> None:
        # Between steps nothing runs; a crash leaves exactly the dying row.
        processing = [m for m in self.model.values() if m["status"] == "processing"]
        assert len(processing) <= 1
        if not self.crashed:
            assert not processing

    @invariant()
    def scheduler_kept_its_rules(self) -> None:
        assert not self.violations, self.violations

    @invariant()
    def delivered_in_arrival_order(self) -> None:
        per_session: dict[str, list[int]] = defaultdict(list)
        for session, seq in self.deliveries:
            per_session[session].append(seq)
        for seq_list in per_session.values():
            assert seq_list == sorted(seq_list), seq_list

    @invariant()
    def caps_hold(self) -> None:
        assert self._pending() <= TOTAL_CAP
        for s in SESSIONS:
            assert self._pending(s) <= PER_SESSION_CAP

    @invariant()
    def status_reader_agrees(self) -> None:
        for rid in self.seen:
            got = self.run(mq.get_task(rid))
            m = self.model.get(rid)
            if m is None:
                assert got is None, rid
            else:
                assert got is not None and got.status == m["status"], rid
                assert (got.finished_at is not None) == (
                    m["status"] in ("success", "failed")
                ), rid


TestMemorizeQueueMachine = MemorizeQueueMachine.TestCase


# A machine that cannot fail proves nothing. Inject one bug -- a restart that
# fails the in-flight row instead of requeueing it -- and pin which invariant
# trips: the model carries the spec (crash -> queued), the table disagrees.
def test_machine_catches_a_lost_replay() -> None:
    machine = MemorizeQueueMachine()
    try:
        real_fail_abandoned = machine.q.repo.fail_abandoned

        async def forgets_the_row(**kwargs: Any) -> tuple[int, int]:
            return await real_fail_abandoned(), 0  # the bug: failed, not requeued

        machine.q.repo.requeue_processing = forgets_the_row  # type: ignore[method-assign]
        machine.enqueue("s1", False)
        machine.drain(["crash"], [], False, False)
        machine.restart()
        with pytest.raises(AssertionError):
            machine.table_matches_model()
    finally:
        machine.teardown()


def test_machine_catches_two_rows_of_one_session() -> None:
    """Second self-check: a claim that ignores busy sessions must trip the
    scheduler observation, not slip through as a different ordering."""
    machine = MemorizeQueueMachine()
    try:
        real_claim = machine.q.repo.claim_next

        async def claim_ignoring_busy_sessions(**kwargs: Any) -> MemorizeInbox | None:
            return await real_claim(exclude_sessions=())  # the bug

        machine.q.repo.claim_next = claim_ignoring_busy_sessions  # type: ignore[method-assign]
        machine.enqueue("s1", False)
        machine.enqueue("s1", False)
        # Both rows get claimed at once; the first one's memorize() lets the
        # second start by waiting on an enqueue, so they overlap for sure.
        machine.drain(["ok", "ok"], [("s2", False)], False, False)
        with pytest.raises(AssertionError):
            machine.scheduler_kept_its_rules()
    finally:
        machine.teardown()


# ── mode 2: real asyncio tasks racing the running loop ─────────────────────

# The deep profile widens the plan and slows memorize(); the scout should
# wander further locally than CI has time for.
_DEEP = os.environ.get("HYPOTHESIS_PROFILE") == "deep"
_PLAN_MAX = 30 if _DEEP else 10
_SLOW_MAX = 0.3 if _DEEP else 0.05  # seconds a memorize() may take


async def _jittered(
    plan: list[tuple[str, bool, str]],
    seed: int,
    restart_at: float | None,
    lose: list[int],
    slow: float,
) -> None:
    """Real producers, the real loop, random pauses.

    ``restart_at`` (0..1) stops and restarts the worker that far into the
    run, or never. ``lose`` names plan entries whose terminal status write
    fails all three attempts (sqlite locked / disk full). ``slow`` is the
    longest a memorize() may take, so rows of different sessions really do
    overlap.
    """
    rng = random.Random(seed)
    outcome_of: dict[str, str] = {}
    deliveries: list[str] = []
    calls: dict[str, int] = defaultdict(int)
    rid_of_row: dict[int, str] = {}
    lose_rids = {f"r{k}" for k in lose if k < len(plan)}
    lost_budget = {rid: 3 for rid in lose_rids}  # _MARK_ATTEMPTS
    settled = asyncio.Event()
    violations: list[str] = []
    t0 = time.perf_counter()
    trace: list[str] = []

    def note(event: str) -> None:
        trace.append(f"{(time.perf_counter() - t0) * 1000:7.1f}ms {event}")

    async def fake_memorize(
        payload: dict[str, Any],
        *,
        is_final: bool = False,
        defer_extraction: bool = False,
    ) -> MemorizeResult:
        rid = payload["rid"]
        session = payload["session_id"]
        deliveries.append(rid)
        calls[rid] += 1
        busy = sorted(q.worker._in_flight)
        note(f"memorize {rid}/{session} #{calls[rid]} in_flight={busy}")
        same = [
            r.request_id
            for r, _ in q.worker._in_flight.values()
            if r.session_id == session
        ]
        if len(same) > 1:
            violations.append(f"{rid}: {same} of {session} in flight together")
        await asyncio.sleep(rng.random() * (0.005 + slow))
        note(f"  {rid} -> {outcome_of[rid]}")
        if outcome_of[rid] == "fail":
            raise RuntimeError("boom")
        return MemorizeResult(message_count=1, status="extracted")

    cfg = _settings(
        max_pending_per_session=50, max_pending_total=200, scan_interval_seconds=0.05
    )
    q = _Queue(fake_memorize, cfg)
    await q.setup()
    real_mark = q.repo.mark
    real_fail_abandoned = q.repo.fail_abandoned
    terminal = 0

    def maybe_settled() -> None:
        if terminal >= len(plan):
            settled.set()

    async def rid_for(row_id: int) -> str:
        rid = rid_of_row.get(row_id)
        if rid is None:  # the producer may not have mapped it yet
            row = await q.repo.get_by_id(row_id)
            rid = row.request_id if row is not None else f"row{row_id}"
            rid_of_row[row_id] = rid
        return rid

    async def counting_mark(row_id: int, **values: Any) -> None:
        nonlocal terminal
        rid = await rid_for(row_id)
        status = values.get("status")
        if status in ("success", "failed") and lost_budget.get(rid, 0) > 0:
            lost_budget[rid] -= 1
            note(f"mark {rid} {status} LOST ({3 - lost_budget[rid]}/3)")
            raise RuntimeError("database is locked")
        await real_mark(row_id, **values)
        note(f"mark {rid} {status}")
        if status in ("success", "failed"):
            terminal += 1
            maybe_settled()

    async def counting_fail_abandoned(**kwargs: Any) -> int:
        nonlocal terminal
        n = await real_fail_abandoned(**kwargs)
        if n:
            note(f"reap {n} lost row(s)")
            terminal += n
            maybe_settled()
        return n

    q.repo.mark = counting_mark  # type: ignore[method-assign]
    q.repo.fail_abandoned = counting_fail_abandoned  # type: ignore[method-assign]
    restarted = False
    event(f"jitter: slow={slow}")
    if lose_rids:
        event("jitter: status write lost")
    if restart_at is not None:
        event("jitter: restart mid-run")
    try:
        await q.worker.start()

        async def produce(k: int, session: str, is_final: bool, outcome: str) -> None:
            rid = f"r{k}"
            outcome_of[rid] = outcome
            await asyncio.sleep(rng.random() * 0.01)
            pending = await q.worker.enqueue(
                _payload(session, rid),
                request_id=rid,
                is_final=is_final,
                defer_extraction=False,
            )
            note(f"enqueue {rid}/{session} accepted pending={pending}")
            row = await q.repo.find_one(request_id=rid)
            if row is not None and row.id is not None:
                rid_of_row.setdefault(row.id, rid)

        async def peek() -> None:
            for _ in range(5):
                await asyncio.sleep(rng.random() * 0.01)
                await mq.list_tasks(session_id=rng.choice(SESSIONS))

        async def restart() -> None:
            nonlocal restarted
            if restart_at is None:
                return
            # Land somewhere inside the run, whatever its length.
            await asyncio.sleep(restart_at * (0.05 + slow * len(plan) / 2))
            note("restart: stop()")
            await q.worker.stop()
            note("restart: stopped, start()")
            await q.worker.start()
            restarted = True
            note("restart: started")

        await asyncio.gather(
            *(produce(k, *item) for k, item in enumerate(plan)), peek(), restart()
        )
        async with asyncio.timeout(60.0):
            await settled.wait()
        await q.worker.stop()
        dump = "\n".join(trace)
        assert not violations, f"{violations}\n{dump}"
        assert await q.repo.count_pending() == 0, dump

        rows = {r.request_id: r for r in await q.rows()}
        assert set(rows) == set(outcome_of), dump
        for rid, outcome in outcome_of.items():
            r = rows[rid]
            expected = "failed" if outcome == "fail" else "success"
            if rid in lose_rids and calls[rid] == 1:
                # Its own status write never landed: the sweep failed it and
                # kept the body, whatever memorize() had returned.
                event("jitter: lost row reaped by the sweep")
                assert (r.status, r.last_error) == ("failed", LOST_WRITE_ERROR), (
                    rid + "\n" + dump
                )
                assert r.payload_json != "", rid
                continue
            if rid in lose_rids:
                # Restarted before the sweep got to it: requeued and replayed
                # like a crash (documented at-least-once), and the replay's
                # own write landed.
                event("jitter: lost row replayed after restart")
                assert restarted and calls[rid] == 2, rid + "\n" + dump
            else:
                assert calls[rid] == 1, rid + "\n" + dump
            assert r.status == expected, rid + "\n" + dump
            assert (r.payload_json != "") == (outcome == "fail"), rid
        # Per session, arrival order is the order the inserts committed, i.e.
        # row id order; a replay repeats a row, it never skips ahead.
        per_session: dict[str, list[int]] = defaultdict(list)
        for rid in deliveries:
            per_session[rows[rid].session_id].append(rows[rid].id)  # type: ignore[arg-type]
        for ids in per_session.values():
            assert ids == sorted(ids), f"{ids}\n" + dump
    finally:
        await q.teardown()


@given(
    plan=st.lists(
        st.tuples(
            st.sampled_from(SESSIONS), st.booleans(), st.sampled_from(["ok", "fail"])
        ),
        min_size=1,
        max_size=_PLAN_MAX,
    ),
    seed=st.integers(0, 10_000),
    restart_at=st.one_of(st.none(), st.floats(0.0, 1.0)),
    lose=st.lists(st.integers(0, _PLAN_MAX - 1), unique=True, max_size=3),
    slow=st.sampled_from([0.0, 0.05, _SLOW_MAX]),
)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_jittered_real_tasks(
    plan: list[tuple[str, bool, str]],
    seed: int,
    restart_at: float | None,
    lose: list[int],
    slow: float,
) -> None:
    asyncio.run(_jittered(plan, seed, restart_at, lose, slow))
