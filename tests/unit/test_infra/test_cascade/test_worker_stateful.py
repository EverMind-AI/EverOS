"""Model-based test of the cascade worker's ``md_change_state`` machine.

Second instance of the pattern in
``tests/unit/test_service/test_memorize_queue_stateful.py`` (its module
docstring explains the split). Here the real :class:`CascadeWorker` runs
against the real ``MdChangeStateRepo`` on a temp SQLite file; only the
handlers are scripted (no LanceDB: a handler without ``index_repo`` makes
the optimize scheduler a no-op).

Rules (what the outside world does): the scanner / watcher ``upsert``-ing a
path (new, re-saved with a new mtime, re-saved unchanged), one ``drain``
batch whose handlers succeed / fail transiently / fail for good / find the
file gone / die mid-batch -- optionally with a path re-upserted *while* its
handler runs --, ``cascade fix --apply``, and a restart.

Invariants after every step, from the repo and worker docstrings:

1. the table matches the model: status, retryable, retry_count, lsn, error;
2. a re-enqueue during processing wins: the stale verdict is dropped and
   the row is pending again with a fresh lsn. When the re-enqueue and the
   verdict race (another row's handler re-saves this row), the verdict may be
   written first: the row was briefly ``failed``, and a re-save with an
   unchanged mtime then keeps its retry count. Both orders are valid, and
   the table decides which one happened (``_settle_races``);
3. retry bookkeeping: transient failures spend the budget (each attempt is
   persisted, so a mid-ladder restart keeps it), a row stays retryable only
   below ``MAX_TOTAL_RETRIES``, re-saves with new content reset it, ``fix``
   resets it;
4. ``queue_summary`` (what ``cascade status`` and ``/health`` show) agrees
   with the table;
5. no ``processing`` row while the process is alive and idle.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, currently_in_test_context, settings
from hypothesis import event as _hypothesis_event
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule
from sqlalchemy import select
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.core.errors import EmbeddingServiceError
from everos.infra.cascade._projections import Handler
from everos.infra.cascade._worker import CascadeWorker
from everos.infra.cascade._worker import _tuning as tuning
from everos.infra.cascade.models import HandlerOutcome
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import create_session_factory, create_system_engine
from everos.infra.persistence.sqlite._repos.md_change_state import (
    MTIME_TOLERANCE_SECONDS,
    MdChangeStateRepo,
)
from everos.infra.persistence.sqlite._tables import MdChangeState

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

PATHS = ("users/u/episodes/a.md", "users/u/episodes/b.md", "users/u/episodes/c.md")
KINDS = ("episode", "orphan")  # "orphan" has no handler registered
CHANGES = ("added", "modified", "deleted")
BATCH = 2
MAX_RETRY = 1  # inline attempts after the first
MAX_TOTAL = tuning.MAX_TOTAL_RETRIES
OUTCOMES = ("ok", "transient", "transient_then_ok", "boom", "vanished", "crash")


def event(label: str) -> None:
    if currently_in_test_context():
        _hypothesis_event(label)


class _Crash(BaseException):
    """The process dies inside a handler: rows of the batch stay processing."""


class _ScriptedHandler(Handler):
    """Plays the outcome the machine scripted for each path."""

    def __init__(self, machine: CascadeMachine) -> None:  # no HandlerDeps
        self.machine = machine
        self.active = 0

    async def handle_added_or_modified(self, md_path: str) -> HandlerOutcome:
        m = self.machine
        self.active += 1
        try:
            m.trace.append(f"handle {md_path} active={self.active}")
            events, m.mid_events = m.mid_events, []
            for path, kind, change, same_mtime in events:
                await m._upsert_async(path, kind, change, same_mtime, in_handler=True)
            outcome = m.outcome_of.get(md_path, "ok")
            attempt = m.attempts[md_path] = m.attempts[md_path] + 1
            m.trace.append(f"  {md_path} attempt {attempt} -> {outcome}")
            if outcome == "crash":
                await m._wait_until_alone(md_path)
                raise _Crash
            if outcome == "vanished":
                raise FileNotFoundError(md_path)
            if outcome == "boom":
                raise RuntimeError("unexpected boom")
            if outcome == "transient" or (
                outcome == "transient_then_ok" and attempt == 1
            ):
                raise EmbeddingServiceError("embedding 503")
            return HandlerOutcome(
                md_path=md_path, kind="episode", upserted=1, deleted=0, skipped=0
            )
        finally:
            self.active -= 1

    async def handle_deleted(self, md_path: str) -> HandlerOutcome:
        self.machine.trace.append(f"delete {md_path}")
        return HandlerOutcome(
            md_path=md_path, kind="episode", upserted=0, deleted=1, skipped=0
        )


def _verdict(m: dict[str, Any], outcome: str) -> dict[str, Any] | None:
    """What _process_one writes for a claimed row with this scripted outcome:
    the columns it sets, or ``None`` when it writes nothing (the crash)."""
    if m["kind"] != "episode" or m["retry_count"] >= MAX_TOTAL:
        return {"status": "failed", "retryable": False, "error": True}
    if m["change"] == "deleted" or outcome in ("ok", "vanished"):
        return {"status": "done", "retryable": None, "error": False}
    if outcome == "transient_then_ok":
        # The one retry it consumed is persisted before the successful
        # second call, so the done row keeps that count (#549).
        return {
            "status": "done",
            "retryable": None,
            "error": False,
            "retry_count": m["retry_count"] + 1,
        }
    if outcome == "boom":
        return {"status": "failed", "retryable": False, "error": True}
    if outcome == "transient":
        r = m["retry_count"] + MAX_RETRY
        return {
            "status": "failed",
            "retryable": r < MAX_TOTAL,
            "retry_count": r,
            "error": True,
        }
    return None  # crash: stays processing


class CascadeMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.tmp = Path(tempfile.mkdtemp(prefix="cascade_stateful_"))
        root = MemoryRoot(self.tmp)
        root.ensure()
        self.engine = create_system_engine(root.system_db, SqliteSettings())
        self.factory = create_session_factory(self.engine)
        self.repo = MdChangeStateRepo(session_factory=self.factory)
        self.run(self._create_tables())
        self.handler = _ScriptedHandler(self)
        self.worker = CascadeWorker(
            {"episode": self.handler},
            self.repo,
            batch_size=BATCH,
            handler_concurrency=2,
            max_retry=MAX_RETRY,
            retry_backoff_seconds=0.0,
            poll_interval_seconds=0.05,
        )
        # the model: path -> expected row; plus bookkeeping for the rules
        self.model: dict[str, dict[str, Any]] = {}
        self.next_lsn = 1
        self.crashed = False
        self.outcome_of: dict[str, str] = {}
        self.attempts: dict[str, int] = defaultdict(int)
        self.mid_events: list[tuple[str, str, str, bool]] = []
        self.claimed: set[str] = set()
        self.marked: set[str] = set()
        self.trace: list[str] = []
        real_done, real_failed = self.repo.mark_done, self.repo.mark_failed

        async def done_counting(md_path: str, **kw: Any) -> None:
            await real_done(md_path, **kw)
            self.marked.add(md_path)

        async def failed_counting(md_path: str, **kw: Any) -> None:
            await real_failed(md_path, **kw)
            self.marked.add(md_path)

        self.repo.mark_done = done_counting  # type: ignore[method-assign]
        self.repo.mark_failed = failed_counting  # type: ignore[method-assign]

    async def _create_tables(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    def run(self, coro: Any) -> Any:
        return self.loop.run_until_complete(coro)

    def teardown(self) -> None:
        self.run(self.engine.dispose())
        self.loop.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── model helpers ──────────────────────────────────────────────────────

    def _mtime_for(self, path: str, same: bool) -> float:
        old = self.model.get(path)
        if old is not None and same:
            return old["mtime"]
        return float(self.next_lsn) * 10.0  # any fresh value far from the old one

    async def _upsert_async(
        self, path: str, kind: str, change: str, same_mtime: bool, *, in_handler: bool
    ) -> None:
        mtime = self._mtime_for(path, same_mtime)
        lsn = await self.repo.upsert(path, kind=kind, change_type=change, mtime=mtime)
        assert lsn == self.next_lsn, (lsn, self.next_lsn)
        old = self.model.get(path)
        keep = (
            old is not None
            and old["status"] == "failed"
            and abs(old["mtime"] - mtime) < MTIME_TOLERANCE_SECONDS
        )
        retry_count = old["retry_count"] if (old is not None and keep) else 0
        if_failed_first = None
        if old is not None and old["status"] == "processing":
            # Re-enqueued while its handler runs: the verdict about to be
            # written is stale and must be dropped (mark_* are guarded).
            old["shielded"] = True
            if_failed_first = self._retry_count_if_failed_first(path, old, mtime)
            event("upsert: during processing")
        self.model[path] = {
            "status": "pending",
            "retryable": None,
            "retry_count": retry_count,
            "error": False,
            "lsn": lsn,
            "mtime": mtime,
            "kind": kind,
            "change": change,
            "shielded": old["shielded"]
            if (old is not None and old.get("shielded"))
            else False,
            "if_failed_first": if_failed_first,
        }
        self.next_lsn += 1
        where = " (in handler)" if in_handler else ""
        self.trace.append(
            f"upsert {path} kind={kind} change={change} same_mtime={same_mtime} "
            f"-> lsn={lsn} retry_count={retry_count}{where}"
            + (
                f" (or {if_failed_first} if the failure landed first)"
                if if_failed_first is not None
                else ""
            )
        )

    def _retry_count_if_failed_first(
        self, path: str, claim: dict[str, Any], mtime: float
    ) -> int | None:
        """The retry count a re-save of this claimed row keeps if the claim's
        failure is written before it; ``None`` when the order cannot matter.

        Inside a handler, the re-save of another row of the batch races that
        row's own terminal write (a row with no handler writes within a few
        statements). A re-save that lands first finds the row ``processing``:
        the count resets and the verdict is dropped. A failure that lands
        first leaves the row ``failed``, and a re-save with an unchanged mtime
        then keeps the count the failure recorded. A ``done`` verdict resets
        the count either way, so only a failure can differ.
        """
        verdict = _verdict(claim, self.outcome_of.get(path, "ok"))
        if verdict is None or verdict["status"] != "failed":
            return None
        if abs(claim["mtime"] - mtime) >= MTIME_TOLERANCE_SECONDS:
            return None
        return int(verdict.get("retry_count", claim["retry_count"]))

    async def _wait_until_alone(self, md_path: str) -> None:
        """Crash only when every other row of the batch has been marked."""
        deadline = time.perf_counter() + 30.0
        while time.perf_counter() < deadline:
            others = self.claimed - {md_path}
            if others <= self.marked and self.handler.active == 1:
                return
            await asyncio.sleep(0)
        raise RuntimeError(f"{md_path}: batch never settled around the crash")

    def _apply_outcome(
        self, path: str, m: dict[str, Any], scripted: dict[str, str]
    ) -> None:
        """What _process_one must have written for this claimed row."""
        if m.get("shielded"):
            m["shielded"] = False  # the stale verdict was dropped; row stays pending
            return
        verdict = _verdict(m, scripted.get(path, "ok"))
        if verdict is not None:
            m.update(verdict)

    def _settle_races(self) -> None:
        """Take the retry count each racing re-save left on its row.

        The valid counts are the model's (the re-save landed first and reset
        it) and ``if_failed_first`` (the failure landed first and the re-save
        kept it); the table says which happened. Any other count is still a
        mismatch, and every other column stays an exact comparison in
        ``table_matches_model``.
        """
        racing = {
            p: m for p, m in self.model.items() if m["if_failed_first"] is not None
        }
        if not racing:
            return
        rows = {r.md_path: r for r in self.run(self._rows())}
        for path, m in racing.items():
            reset, kept = m["retry_count"], m["if_failed_first"]
            got = rows[path].retry_count
            assert got in (reset, kept), (
                f"{path}: retry_count={got}, valid: {reset} (re-save first) or "
                f"{kept} (failure first)\n" + "\n".join(self.trace[-30:])
            )
            if kept != reset:
                event(
                    "re-save race: failure first"
                    if got == kept
                    else "re-save race: re-save first"
                )
            m["retry_count"], m["if_failed_first"] = got, None

    # ── rules ──────────────────────────────────────────────────────────────

    @precondition(lambda self: not self.crashed)
    @rule(
        path=st.sampled_from(PATHS),
        kind=st.sampled_from(KINDS),
        change=st.sampled_from(CHANGES),
        same_mtime=st.booleans(),
    )
    def upsert(self, path: str, kind: str, change: str, same_mtime: bool) -> None:
        self.run(self._upsert_async(path, kind, change, same_mtime, in_handler=False))
        event(
            "upsert: re-save unchanged"
            if same_mtime and path in self.model
            else "upsert"
        )

    @precondition(lambda self: not self.crashed)
    @rule(
        outcomes=st.lists(st.sampled_from(OUTCOMES), max_size=BATCH),
        mid=st.lists(
            st.tuples(
                st.sampled_from(PATHS),
                st.sampled_from(KINDS),
                st.sampled_from(CHANGES),
                st.booleans(),
            ),
            max_size=1,
        ),
    )
    def drain(self, outcomes: list[str], mid: list[tuple[str, str, str, bool]]) -> None:
        """One ``drain_once``: the BATCH smallest-lsn pending rows, scripted."""
        pending = sorted(
            (m["lsn"], path)
            for path, m in self.model.items()
            if m["status"] == "pending"
        )
        batch = [path for _, path in pending[:BATCH]]
        if "crash" in outcomes:
            first = outcomes.index("crash")
            outcomes = outcomes[: first + 1] + [
                "ok" if o == "crash" else o for o in outcomes[first + 1 :]
            ]
        self.outcome_of = dict(zip(batch, outcomes, strict=False))
        self.attempts = defaultdict(int)
        self.mid_events = list(mid) if batch else []
        self.claimed, self.marked = set(batch), set()
        for path in batch:
            self.model[path]["status"] = "processing"
            event(f"drain: {self.outcome_of.get(path, 'ok')}")
        self.trace.append(f"drain batch={batch} outcomes={self.outcome_of} mid={mid}")
        scripted = dict(self.outcome_of)
        try:
            handled = self.run(self.worker.drain_once())
        except _Crash:
            self.crashed = True
            handled = len(batch)
        finally:
            self.outcome_of, self.mid_events = {}, []
        assert handled == len(batch), (handled, batch)
        for path in batch:
            self._apply_outcome(path, self.model[path], scripted)
        self._settle_races()

    @precondition(lambda self: not self.crashed)
    @rule()
    def fix(self) -> None:
        n = self.run(self.repo.reset_retryable_to_pending())
        fixable = [
            m for m in self.model.values() if m["status"] == "failed" and m["retryable"]
        ]
        assert n == len(fixable), (n, len(fixable))
        for m in fixable:
            m.update(status="pending", retryable=None, retry_count=0, error=False)
        event("fix: re-enqueued" if fixable else "fix: nothing")

    @rule()
    def restart(self) -> None:
        n = self.run(self.repo.recover_orphan_processing())
        orphans = [m for m in self.model.values() if m["status"] == "processing"]
        assert n == len(orphans), (n, len(orphans))
        for m in orphans:
            m["status"] = "pending"
        self.crashed = False
        self.worker = CascadeWorker(
            {"episode": self.handler},
            self.repo,
            batch_size=BATCH,
            handler_concurrency=2,
            max_retry=MAX_RETRY,
            retry_backoff_seconds=0.0,
            poll_interval_seconds=0.05,
        )
        event("restart: recovered orphans" if orphans else "restart: clean")

    # ── invariants ─────────────────────────────────────────────────────────

    async def _rows(self) -> list[MdChangeState]:
        async with self.factory() as s:
            return list((await s.execute(select(MdChangeState))).scalars().all())

    @invariant()
    def table_matches_model(self) -> None:
        rows = {r.md_path: r for r in self.run(self._rows())}
        assert set(rows) == set(self.model), "\n".join(self.trace[-30:])
        for path, m in self.model.items():
            r = rows[path]
            got = (r.status, r.retryable, r.retry_count, r.lsn, r.error is not None)
            want = (m["status"], m["retryable"], m["retry_count"], m["lsn"], m["error"])
            assert got == want, f"{path}: table={got} model={want}\n" + "\n".join(
                self.trace[-30:]
            )

    @invariant()
    def summary_matches_table(self) -> None:
        s = self.run(self.repo.queue_summary())
        ms = self.model.values()
        assert s.pending == sum(
            1 for m in ms if m["status"] in ("pending", "processing")
        )
        assert s.done == sum(1 for m in ms if m["status"] == "done")
        assert s.failed_retryable == sum(
            1 for m in ms if m["status"] == "failed" and m["retryable"]
        )
        assert s.failed_permanent == sum(
            1 for m in ms if m["status"] == "failed" and m["retryable"] is False
        )
        assert s.max_lsn == (self.next_lsn - 1 if self.model else 0)
        terminal = [m["lsn"] for m in ms if m["status"] in ("done", "failed")]
        assert s.last_processed_lsn == (max(terminal) if terminal else 0)

    @invariant()
    def idle_means_nothing_processing(self) -> None:
        processing = [p for p, m in self.model.items() if m["status"] == "processing"]
        if not self.crashed:
            assert not processing, processing

    @invariant()
    def retry_budget_bookkeeping(self) -> None:
        for path, m in self.model.items():
            assert 0 <= m["retry_count"] <= MAX_TOTAL + MAX_RETRY, (
                path,
                m["retry_count"],
            )
            if m["status"] == "failed" and m["retryable"]:
                assert m["retry_count"] < MAX_TOTAL, (path, m)
            if m["status"] == "done":
                assert m["retryable"] is None and not m["error"], (path, m)
            if m["status"] == "failed":
                assert m["retryable"] is not None and m["error"], (path, m)


TestCascadeMachine = CascadeMachine.TestCase


def test_machine_catches_a_stale_verdict_overwriting_a_re_enqueue() -> None:
    """Self-check: drop the ``WHERE status='processing'`` guard on mark_done
    and the "re-enqueue during processing wins" invariant must trip."""
    machine = CascadeMachine()
    try:
        from types import SimpleNamespace

        from sqlalchemy import update

        from everos.infra.persistence.sqlite import session_scope

        async def stale_read_one(md_path: str) -> Any:
            # Pretend the re-enqueue never happened, so the worker still
            # takes the mark_done path. The model must catch that verdict.
            return SimpleNamespace(
                md_path=md_path,
                lsn=1,
                status="processing",
                kind="episode",
                mtime=0.0,
            )

        async def unguarded_mark_done(md_path: str, **_: Any) -> None:
            async with session_scope(machine.factory) as s:
                await s.execute(
                    update(MdChangeState)
                    .where(MdChangeState.md_path == md_path)
                    .values(status="done")
                )
                await s.commit()
            machine.marked.add(md_path)

        machine.repo.mark_done = unguarded_mark_done  # type: ignore[method-assign]
        machine.repo.read_one = stale_read_one  # type: ignore[method-assign]
        machine.upsert(PATHS[0], "episode", "added", False)
        machine.drain(["ok"], [(PATHS[0], "episode", "modified", False)])
        with pytest.raises(AssertionError):
            machine.table_matches_model()
    finally:
        machine.teardown()


def _pin_write_order(machine: CascadeMachine, *, failure_first: bool) -> None:
    """Fix the order of the next re-save and the next failure verdict, which
    naturally race on two connections."""
    first_landed = asyncio.Event()
    real_upsert, real_failed = machine.repo.upsert, machine.repo.mark_failed

    async def upsert(*args: Any, **kw: Any) -> int:
        if failure_first:
            await first_landed.wait()
        lsn = await real_upsert(*args, **kw)
        first_landed.set()
        return lsn

    async def mark_failed(md_path: str, **kw: Any) -> None:
        if not failure_first:
            await first_landed.wait()
        await real_failed(md_path, **kw)
        first_landed.set()

    machine.repo.upsert = upsert  # type: ignore[method-assign]
    machine.repo.mark_failed = mark_failed  # type: ignore[method-assign]


@pytest.mark.parametrize("failure_first", [False, True], ids=["re-save", "failure"])
def test_machine_accepts_either_order_of_a_re_save_and_a_failure(
    failure_first: bool,
) -> None:
    """A handler re-saves the path of a row with no handler, which fails at once.

    Whichever write lands first is valid. Re-save first: the row is still
    ``processing``, its count resets and the failure is dropped. Failure first:
    the row is ``failed`` when it is re-saved with the same mtime, so it keeps
    the count the failure recorded. The model takes whichever of the two the
    table shows and no other; each order is pinned here, so both are known to
    occur and each is held to its exact count.
    """
    orphan, worker_row = PATHS[0], PATHS[1]
    machine = CascadeMachine()
    try:
        machine.upsert(orphan, "episode", "added", False)
        machine.drain(["transient"], [])  # one retry spent: failed at count 1
        machine.upsert(orphan, "orphan", "added", True)  # keeps the count, no handler
        machine.upsert(worker_row, "episode", "added", False)
        _pin_write_order(machine, failure_first=failure_first)
        # batch [orphan, worker_row]: the worker row's handler re-saves the orphan
        machine.drain(["ok", "ok"], [(orphan, "orphan", "modified", True)])
        machine.table_matches_model()
        assert machine.model[orphan]["retry_count"] == (1 if failure_first else 0)
    finally:
        machine.teardown()
