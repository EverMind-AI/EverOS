"""Model-based test of the ``async_mode`` queue: hypothesis drives the worker.

The unit tests next door pin the sequences someone thought of. This file
lets ``hypothesis`` generate thousands of operation sequences -- enqueue
through the caps, a drain whose rows succeed / fail / die mid-``memorize()``,
requests that arrive *while* a row is inside ``memorize()`` (the real
single-loop interleaving), a lost status write, a restart, the retention
sweep with the clock moved -- against the real repo on a real SQLite file
and the real worker, and checks the state-machine invariants after every
step:

1. no ``processing`` row while the process is alive and idle; while rows
   run, never two of one session and never more than ``max_concurrency``;
2. a session's rows reach ``memorize()`` in arrival order, and a row starts
   only when no earlier row of its session is still queued; a replay after
   a crash repeats the crashed row, it never skips ahead;
3. ``success`` / ``failed`` never change again except by purge; a delivered
   row has no body, a failed row keeps it;
4. the caps hold: 429 is raised exactly when the model says the session or
   the process is full;
5. ``GET /tasks/{task_id}`` reports what the table holds.

Two modes. ``TestMemorizeQueueMachine`` is the rule-based machine: one step
at a time, deterministic, replayable, and a failure shrinks to the shortest
sequence. ``test_jittered_real_tasks`` runs the same kind of plan with real
``asyncio`` tasks racing the running worker loop: true interleaving, not
replayable. ``HYPOTHESIS_PROFILE=deep`` runs many more examples locally.
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
        self.n = 0
        self.crashed = False
        # Outcome per request id, fixed when the drain starts: with rows
        # running side by side, "the k-th memorize() call" is timing.
        self.outcome_of: dict[str, str] = {}
        self.mid_events: list[tuple[str, bool]] = []
        self.lose_marks = 0
        self.lose_rid: str | None = None
        self.lose_row_id: int | None = None
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
        if self.lose_rid == rid:
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
        for _ in range(100_000):
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
                await asyncio.wait(others)
            elif len(self.q.worker._in_flight) > 1 or claimable:
                await asyncio.sleep(0)  # let the scheduler collect / claim
            else:
                return
        raise RuntimeError("other rows never left the scheduler")

    async def _mark_maybe_lost(self, row_id: int, **values: Any) -> None:
        # Only the chosen row's writes are lost; with two rows in flight a
        # budget shared by both would land on whichever marks first.
        if self.lose_marks and row_id == self.lose_row_id:
            self.lose_marks -= 1
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
    )
    def drain(
        self, outcomes: list[str], mid: list[tuple[str, bool]], lose_write: bool
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
            event("drain: status write lost")
            # The first row's status write fails all three attempts.
            self.lose_marks = 3
            self.lose_row_id, self.lose_rid = queued[0]
        try:
            self.run(self.q.worker.drain_once())
        except _Crash:
            self.crashed = True
        finally:
            self.lose_marks, self.lose_rid, self.lose_row_id = 0, None, None
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
        requeued = self.run(self.q.repo.requeue_processing())
        in_flight = [m for m in self.model.values() if m["status"] == "processing"]
        assert requeued == len(in_flight)
        event("restart: replayed a crashed row" if in_flight else "restart: clean")
        for m in in_flight:
            m["status"] = "queued"
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


TestMemorizeQueueMachine = MemorizeQueueMachine.TestCase


# A machine that cannot fail proves nothing. Inject one bug -- a restart that
# fails the in-flight row instead of requeueing it -- and pin which invariant
# trips: the model carries the spec (crash -> queued), the table disagrees.
def test_machine_catches_a_lost_replay() -> None:
    machine = MemorizeQueueMachine()
    try:
        machine.q.repo.requeue_processing = machine.q.repo.fail_abandoned  # type: ignore[method-assign]
        machine.enqueue("s1", False)
        machine.drain(["crash"], [], False)
        machine.restart()
        with pytest.raises(AssertionError):
            machine.table_matches_model()
    finally:
        machine.teardown()


# ── mode 2: real asyncio tasks racing the running loop ─────────────────────

# The deep profile also widens the plan; the scout should wander further locally.
_PLAN_MAX = 30 if os.environ.get("HYPOTHESIS_PROFILE") == "deep" else 10


async def _jittered(
    plan: list[tuple[str, bool, str]], seed: int, restart_at: float | None
) -> None:
    """Real producers, the real loop, random pauses; ``restart_at`` (0..1)
    stops and restarts the worker that far into the run, or never."""
    rng = random.Random(seed)
    outcome_of: dict[str, str] = {}
    deliveries: list[str] = []
    rid_of_row: dict[int, str] = {}
    settled = asyncio.Event()
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
        deliveries.append(rid)
        session = payload["session_id"]
        note(f"memorize {rid}/{session} in_flight={sorted(q.worker._in_flight)}")
        await asyncio.sleep(rng.random() * 0.005)
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
    terminal = 0

    async def counting_mark(row_id: int, **values: Any) -> None:
        nonlocal terminal
        await real_mark(row_id, **values)
        note(f"mark {rid_of_row.get(row_id, f'row{row_id}')} {values.get('status')}")
        if values.get("status") in ("success", "failed"):
            terminal += 1
            if terminal == len(plan):
                settled.set()

    q.repo.mark = counting_mark  # type: ignore[method-assign]
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
                rid_of_row[row.id] = rid

        async def peek() -> None:
            for _ in range(5):
                await asyncio.sleep(rng.random() * 0.01)
                await mq.list_tasks(session_id=rng.choice(SESSIONS))

        async def restart() -> None:
            if restart_at is None:
                return
            await asyncio.sleep(restart_at * 0.05)
            note("restart: stop()")
            await q.worker.stop()
            note("restart: stopped, start()")
            await q.worker.start()
            note("restart: started")

        await asyncio.gather(
            *(produce(k, *item) for k, item in enumerate(plan)), peek(), restart()
        )
        async with asyncio.timeout(5.0):
            await settled.wait()
        await q.worker.stop()
        assert await q.repo.count_pending() == 0, "\n".join(trace)

        rows = {r.request_id: r for r in await q.rows()}
        assert set(rows) == set(outcome_of), "\n".join(trace)
        for rid, outcome in outcome_of.items():
            r = rows[rid]
            assert r.status == ("failed" if outcome == "fail" else "success"), (
                rid + "\n" + "\n".join(trace)
            )
            assert (r.payload_json != "") == (outcome == "fail"), rid
        # Per session, arrival order is the order the inserts committed, i.e.
        # row id order; sessions are independent.
        per_session: dict[str, list[int]] = defaultdict(list)
        for rid in deliveries:
            per_session[rows[rid].session_id].append(rows[rid].id)  # type: ignore[arg-type]
        for ids in per_session.values():
            assert ids == sorted(ids), f"{ids}\n" + "\n".join(trace)
        # A row that finished inside stop()'s grace is never replayed.
        assert len(deliveries) == len(plan), "\n".join(trace)
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
)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_jittered_real_tasks(
    plan: list[tuple[str, bool, str]], seed: int, restart_at: float | None
) -> None:
    asyncio.run(_jittered(plan, seed, restart_at))
