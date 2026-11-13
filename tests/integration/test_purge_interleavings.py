"""Random interleavings of purges, background events, queued requests and
restarts never break the purge invariants (A28, spec §5.8):

* I1 / I6 -- what lies outside a purge's range, or arrived after it ended, is
  kept: every item born after every purge covering its scope ended has its
  product.
* I2 -- nothing born before a purge of its scope survives it: no item born
  before the latest purge covering its scope has its product.

An item born while a purge of its scope runs may or may not keep its product
(writes during a purge may be partly deleted, spec D28): not checked.

Products are ``memcell`` rows -- SQLite-only writes; what keeps a stale one
out is the OME gate and the queue's tombstone check, and the purge waiting
for older work in flight before it deletes a second time. Items: OME events
(the real Runner, gate and run-record store), async memorize rows (the
real queue and inbox) and write requests in flight (tracked as the HTTP
middleware tracks them). A restart drops everything in memory, keeps the
stores, rebuilds the marks
from ``system.db`` and replays what was in flight, as crash recovery does.
Invariants are checked at stable points: no purge running, nothing queued
or in flight.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, Phase, currently_in_test_context, event, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from everos.config.settings import AsyncQueueSettings
from everos.core.locks import LOCKS
from everos.core.purge_marks import MARKS
from everos.core.utils.datetime import get_utc_now
from everos.infra.ome._dispatch.runner import Runner
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.config import OMEConfig
from everos.infra.persistence.sqlite import SqliteMemorizeInboxStore, session_scope
from everos.infra.persistence.sqlite._tables import Memcell
from everos.memory.deps import MemoryDeps
from everos.memory.ports import Event, Immediate
from everos.service import MemorizeResult
from everos.service.memory._queue import MemorizeQueueWorker
from everos.service.memory._queue_steps import recover_processing
from everos.service.purge import PurgeRunGate
from tests.fakes.storage import make_storage
from tests.helpers.ome import ome_strategy
from tests.unit.test_service.test_purge._harness import Purge, make_purge

# A step that cancels work (a restart is a process dying) can strand an
# aiosqlite connection. Its finaliser warns, then stops the connection's
# worker thread; raised as an error the warning aborts that stop, and a
# non-daemon worker thread then blocks the interpreter's exit (Python 3.13).
pytestmark = pytest.mark.filterwarnings(
    "default:.*was deleted before being closed:ResourceWarning"
)

SCOPES = [("a1", "p1"), ("a1", "p2"), ("a2", "p1")]
PURGES = [("project", "a1", "p1"), ("app", "a1", None), ("space", None, None)]


class ScopedEvent(Event):
    app_id: str
    project_id: str


@dataclass
class Item:
    key: str
    app: str
    project: str
    born: datetime


async def _write_product(key: str, app: str, project: str) -> None:
    async with session_scope(_STATE["factory"]) as s:
        if await s.get(Memcell, key) is None:
            s.add(
                Memcell(
                    memcell_id=key,
                    app_id=app,
                    project_id=project,
                    session_id="s",
                    track="user_memory",
                    raw_type="conversation",
                    message_ids_json="[]",
                    sender_ids_json="[]",
                    payload_json="{}",
                    timestamp=get_utc_now(),
                )
            )
            await s.commit()


_STATE: dict[str, Any] = {}


@ome_strategy(name="product", trigger=Immediate(on=[ScopedEvent]), emits=[])
async def _product(event: ScopedEvent, deps: MemoryDeps) -> list[Event]:
    await _write_product(f"ev_{event.event_id}", event.app_id, event.project_id)
    return []


async def _deliver(payload: dict[str, Any], **_: Any) -> MemorizeResult:
    await _write_product(
        f"rq_{payload['session_id']}", payload["app_id"], payload["project_id"]
    )
    return MemorizeResult(message_count=1, status="extracted")


class PurgeMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.tmp = Path(tempfile.mkdtemp(prefix="purge-prop-"))
        MARKS.reset()
        LOCKS.reset()
        self.storage = make_storage(self.tmp)
        self.items: list[Item] = []
        self.purges: list[tuple[tuple[str, str | None, str | None], datetime]] = []
        self.tasks: set[asyncio.Task[Any]] = set()
        self.writes: dict[asyncio.Task[Any], str] = {}
        """In-flight write requests and the item each writes."""
        self.crashed = False
        """A run died in flight (its RUNNING row is left) and no restart yet."""
        self.seq = 0
        self.run(self._boot(first=True))

    # ── plumbing ─────────────────────────────────────────────────────────

    def run(self, coro: Any) -> Any:
        return self.loop.run_until_complete(coro)

    async def _boot(self, *, first: bool) -> None:
        if first:
            await self.storage.create_schema()
            self.ome = OMEStorage(db_path=self.tmp / "ome.db")
            await self.ome.init()
            _STATE["factory"] = self.storage.database.session_factory
        self.records = RunRecordStore(storage=self.ome, max_records_per_strategy=10_000)
        self.purge: Purge = make_purge(self.storage)
        self.purge.service.engine = _Engine(self.records)  # type: ignore[assignment]
        self.runner = self.purge.runner()
        self.ome_runner = Runner(
            run_record_store=self.records,
            engine_sem=asyncio.Semaphore(4),
            emit_hook=_no_emit,
            config=OMEConfig(
                jobstore_path=self.tmp / "unused", retry_backoff_base_seconds=0
            ),
            gate=PurgeRunGate(self.purge.marks),
        )
        self.queue = MemorizeQueueWorker(
            AsyncQueueSettings(retry_backoff_seconds=0.0, scan_interval_seconds=60.0),
            inbox=SqliteMemorizeInboxStore(self.storage.sqlite.memorize_inbox),
            deliver=_deliver,
            marks=self.purge.marks,
        )
        await self.purge.service.restore()
        self.runner.start()

    def _spawn(self, event: ScopedEvent, run_id: str) -> asyncio.Task[Any]:
        task = self.loop.create_task(
            self.ome_runner.run(_product, event, run_id=run_id, max_retries_snapshot=0)
        )
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def _started(self, task: asyncio.Task[Any], run_id: str) -> None:
        """Until the run's RUNNING row is written (or the run ended): from
        then on a restart replays it, as crash recovery does. A run killed
        before that row exists is lost (as in the engine), so the machine
        waits for it rather than guessing how long it takes."""
        async with asyncio.timeout(10):
            while True:
                if task.done() or await self.records.get(run_id) is not None:
                    return
                await asyncio.sleep(0.005)

    async def _settle(self) -> None:
        """Run until no purge is active, nothing queued, nothing in flight.

        Not while a crashed run's row is left: a purge waits for it, and only
        a restart (crash recovery) ends it.
        """
        if self.crashed:
            return
        async with asyncio.timeout(20):
            while True:
                await self.queue.drain_once()
                if self.tasks:
                    await asyncio.wait(set(self.tasks), timeout=0.2)
                active = self.purge.marks.active_job_id
                queued = await self.queue._inbox.count_status("queued")
                if active is None and not self.tasks and queued == 0:
                    return
                await asyncio.sleep(0.02)

    def _key(self, prefix: str) -> str:
        self.seq += 1
        return f"{prefix}{self.seq}"

    # ── rules ────────────────────────────────────────────────────────────

    # Rules stay enabled whatever the background got to: what a rule can do
    # is decided inside it, so a replayed sequence draws the same data.
    @rule(which=st.sampled_from(PURGES))
    def confirm(self, which: tuple[str, str | None, str | None]) -> None:
        if len(self.purges) >= 3:
            return

        async def go() -> None:
            req = await self.purge.service.request(*which)
            job = await self.purge.service.confirm(req.nonce)
            if job.scope == which[0] and (job.app_id, job.project_id) == which[1:]:
                row = await self.purge.service.jobs.get(job.job_id)
                if (which, row.created_at) not in self.purges:  # type: ignore[union-attr]
                    self.purges.append((which, row.created_at))  # type: ignore[union-attr]

        self.run(go())

    @rule(where=st.sampled_from(SCOPES), stale=st.booleans())
    def submit_ome_event(self, where: tuple[str, str], stale: bool) -> None:
        born = get_utc_now() - (timedelta(hours=1) if stale else timedelta(0))
        event = ScopedEvent(app_id=where[0], project_id=where[1], ts=born)
        self.items.append(Item(f"ev_{event.event_id}", *where, born))
        run_id = self._key("run")
        self.run(self._started(self._spawn(event, run_id), run_id))

    @rule(where=st.sampled_from(SCOPES))
    def crash_mid_event(self, where: tuple[str, str]) -> None:
        """An event the process died running: its RUNNING row is all that is
        left, and crash recovery replays it after the next restart."""
        event = ScopedEvent(
            app_id=where[0], project_id=where[1], ts=get_utc_now() - timedelta(hours=1)
        )
        self.items.append(Item(f"ev_{event.event_id}", *where, event.ts))
        self.crashed = True
        self.run(
            self.records.mark_running(
                run_id=self._key("run"),
                strategy_name="product",
                attempt=0,
                event_topic=ScopedEvent.topic(),
                event_payload=event.model_dump_json(),
                max_retries_snapshot=0,
                event_id=event.event_id,
            )
        )

    @rule(where=st.sampled_from(SCOPES))
    def submit_inbox_row(self, where: tuple[str, str]) -> None:
        session = self._key("s")

        async def go() -> None:
            await self.queue.enqueue(
                {
                    "session_id": session,
                    "app_id": where[0],
                    "project_id": where[1],
                    "messages": [{"content": "m"}],
                },
                request_id=f"req-{session}",
                is_final=False,
                defer_extraction=False,
            )
            row = await self.queue._inbox.get_task(f"req-{session}")
            self.items.append(Item(f"rq_{session}", *where, row.created_at))  # type: ignore[union-attr]

        self.run(go())

    @rule(where=st.sampled_from(SCOPES), slow=st.booleans())
    def in_flight_write(self, where: tuple[str, str], slow: bool) -> None:
        """A write request: tracked from its arrival while it runs (as the
        HTTP middleware does), it writes its product when it ends -- possibly
        after a purge confirmed meanwhile, which must wait for it."""
        key = self._key("wr_")
        born = get_utc_now()
        self.items.append(Item(key, *where, born))
        marks = self.purge.marks

        async def go() -> None:
            with marks.track(born):
                await asyncio.sleep(0.3 if slow else 0)
                await _write_product(key, *where)

        task = self.loop.create_task(go())
        self.writes[task] = key
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        self.run(asyncio.sleep(0.01))

    @rule()
    def deliver_one(self) -> None:
        self.run(self.queue.drain_once())

    @rule()
    def claim_and_crash(self) -> None:
        """A row claimed and left ``processing`` (the process died in it)."""
        self.run(self.queue._inbox.claim_next())

    @rule()
    def restart(self) -> None:
        async def go() -> None:
            await self.runner.stop()
            # A request still running dies with the process: its client got
            # an error and its write never happened.
            died = {self.writes[t] for t in self.tasks if t in self.writes}
            self.items = [i for i in self.items if i.key not in died]
            for task in list(self.tasks):
                task.cancel()
            for task in list(self.tasks):
                with contextlib.suppress(BaseException):
                    await task
            MARKS.reset()
            await self._boot(first=False)
            await recover_processing(self.queue._inbox, max_replays=100)
            for record in await self.records.find_running():  # crash recovery
                await self.records.mark_crashed(
                    run_id=record.run_id, finished_at=get_utc_now(), error="crash"
                )
                event = ScopedEvent.model_validate_json(record.event_payload)
                run_id = self._key("run")
                await self._started(self._spawn(event, run_id), run_id)
            self.crashed = False

        self.run(go())

    @rule()
    def run_to_terminal(self) -> None:
        self.run(self._settle())

    # ── invariants (stable points only) ──────────────────────────────────

    def _stable(self) -> bool:
        return self.purge.marks.active_job_id is None and not self.tasks

    async def _covering_jobs(self, item: Item) -> list[tuple[datetime, Any]]:
        """``(confirmed_at, finished_at)`` of every purge covering ``item``."""
        return [
            (row.created_at, row.finished_at)
            for row in await self.purge.service.jobs.all_rows()
            if row.scope == "space"
            or (row.scope == "app" and row.app_id == item.app)
            or (
                row.scope == "project"
                and (row.app_id, row.project_id) == (item.app, item.project)
            )
        ]

    async def _unsettled(self) -> bool:
        """Queued work, or work that died and waits for a restart."""
        inbox = self.queue._inbox
        running = await self.records.find_running()
        busy = await inbox.count_status("processing") + await inbox.count_status(
            "queued"
        )
        return bool(running) or busy > 0

    async def _products(self) -> set[str]:
        async with session_scope(_STATE["factory"]) as s:
            rows = (await s.execute(Memcell.__table__.select())).all()
        return {r.memcell_id for r in rows}

    @invariant()
    def purge_invariants(self) -> None:
        if not self._stable() or self.run(self._unsettled()):
            return  # checked at stable points only; it never drives the system
        products = self.run(self._products())
        for item in self.items:
            jobs = self.run(self._covering_jobs(item))
            older = [at for at, _ in jobs if at > item.born]
            if older:
                _event("I2 checked: an item older than a purge of its scope")
                assert item.key not in products, f"I2: {item} survived {max(older)}"
            elif any(at <= item.born <= (end or item.born) for at, end in jobs):
                _event("not checked: an item born while a purge of its scope ran")
            else:
                _event("I1 checked: an item newer than every purge of its scope")
                assert item.key in products, f"I1: {item} lost; {self._trace(item)}"

    def _trace(self, item: Item) -> str:
        """What the run records say about a lost event (failure message only)."""
        if not item.key.startswith("ev_"):
            return "not an OME event"
        runs = self.run(self.records.list_by_event_id(item.key[3:]))
        return "runs: " + repr([(r.run_id, r.status, r.error) for r in runs])

    def teardown(self) -> None:
        async def go() -> None:
            await self.runner.stop()
            # As asyncio.run ends: every task still on the loop is finished
            # here, so none leaves an open connection to the garbage collector.
            left = asyncio.all_tasks() - {asyncio.current_task()}
            for task in left:
                task.cancel()
            await asyncio.gather(*left, return_exceptions=True)
            await self.storage.aclose()
            await self.loop.shutdown_asyncgens()

        self.run(go())
        self.loop.close()
        MARKS.reset()
        gc.collect()  # what a step stranded is stopped now, under pytestmark
        shutil.rmtree(self.tmp, ignore_errors=True)


def _event(label: str) -> None:
    if currently_in_test_context():  # the fixed sequences run outside one
        event(label)


class _Engine:
    def __init__(self, records: RunRecordStore) -> None:
        self._records = records

    async def running_runs(self) -> list[Any]:
        return await self._records.find_running()

    async def redact_payloads(self, match: Any, marker: str) -> int:
        return await self._records.redact_payloads(match, marker)


async def _no_emit(_event: Event) -> None:
    return None


TestPurgeInterleavings = PurgeMachine.TestCase
TestPurgeInterleavings.settings = settings(
    max_examples=int(os.environ.get("PURGE_PROP_EXAMPLES", "200")),
    phases=(
        [Phase.explicit, Phase.reuse, Phase.generate]
        if os.environ.get("PURGE_PROP_NOSHRINK")
        else list(Phase)
    ),
    stateful_step_count=12,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


def _sequence(steps: list[tuple[str, dict[str, Any]]]) -> None:
    machine = PurgeMachine()
    try:
        for name, kwargs in steps:
            getattr(machine, name)(**kwargs)
            machine.purge_invariants()
        machine.run_to_terminal()
        machine.purge_invariants()
    finally:
        machine.teardown()


def test_stale_event_after_the_purge_is_dropped() -> None:
    """I2, fixed: a backlog event older than the purge of its scope arrives
    after the purge ended; the gate drops it."""
    _sequence(
        [
            ("confirm", {"which": ("project", "a1", "p1")}),
            ("run_to_terminal", {}),
            ("submit_ome_event", {"where": ("a1", "p1"), "stale": True}),
        ]
    )


def test_stale_event_replayed_after_restart_stays_out() -> None:
    """I2, fixed: an event the process died running is replayed after a
    restart that came after its scope's purge; only the tombstone restored
    from system.db keeps it out."""
    _sequence(
        [
            ("crash_mid_event", {"where": ("a1", "p1")}),
            ("confirm", {"which": ("space", None, None)}),
            ("run_to_terminal", {}),
            ("restart", {}),
        ]
    )


def test_queued_row_claimed_at_crash_stays_out() -> None:
    """I2, fixed: a row claimed when the process died, of a scope purged
    meanwhile, is replayed after the restart and must be dropped."""
    _sequence(
        [
            ("submit_inbox_row", {"where": ("a1", "p1")}),
            ("claim_and_crash", {}),
            ("confirm", {"which": ("app", "a1", None)}),
            ("run_to_terminal", {}),
            ("restart", {}),
        ]
    )


def test_other_scope_event_during_purge_lands() -> None:
    """I1, fixed: an event of another scope that arrives during a purge
    runs and lands; nothing outside the scope waits for the purge."""
    _sequence(
        [
            ("confirm", {"which": ("project", "a1", "p1")}),
            ("submit_ome_event", {"where": ("a2", "p1"), "stale": False}),
            ("submit_inbox_row", {"where": ("a1", "p2")}),
        ]
    )
