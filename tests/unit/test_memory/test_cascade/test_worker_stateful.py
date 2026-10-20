"""Model-based test of the cascade worker's ``md_change_state`` machine.

Second instance of the pattern in
``tests/unit/test_service/test_memorize_queue_stateful.py`` (its module
docstring explains the split). Here the real :class:`CascadeWorker` runs
against the real ``_MdChangeStateRepo`` on a temp SQLite file; only the
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
   the row is pending again with a fresh lsn;
3. retry bookkeeping: transient failures spend the budget, a row stays
   retryable only below ``_MAX_TOTAL_RETRIES``, re-saves with new content
   reset it, ``fix`` resets it;
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
from unittest import mock

import pytest
from hypothesis import HealthCheck, currently_in_test_context, settings
from hypothesis import event as _hypothesis_event
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule
from sqlalchemy import select
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.core.errors import EmbeddingServiceError
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite.repos.md_change_state import (
    MTIME_TOLERANCE_SECONDS,
    _MdChangeStateRepo,
)
from everos.infra.persistence.sqlite.tables import MdChangeState
from everos.memory.cascade import worker as worker_mod
from everos.memory.cascade.handlers import Handler
from everos.memory.cascade.types import HandlerOutcome
from everos.memory.cascade.worker import CascadeWorker

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
MAX_TOTAL = worker_mod._MAX_TOTAL_RETRIES
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


class CascadeMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.tmp = Path(tempfile.mkdtemp(prefix="cascade_stateful_"))
        root = MemoryRoot(self.tmp)
        root.ensure()
        self.engine = create_system_engine(root.system_db, SqliteSettings())
        self.factory = create_session_factory(self.engine)
        self.repo = _MdChangeStateRepo(session_factory=self.factory)
        self.run(self._create_tables())
        self._patch = mock.patch.object(worker_mod, "md_change_state_repo", self.repo)
        self._patch.start()
        self.handler = _ScriptedHandler(self)
        self.worker = CascadeWorker(
            {"episode": self.handler},
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

        async def done_counting(md_path: str) -> None:
            await real_done(md_path)
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
        self._patch.stop()
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
        if old is not None and old["status"] == "processing":
            # Re-enqueued while its handler runs: the verdict about to be
            # written is stale and must be dropped (mark_* are guarded).
            old["shielded"] = True
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
        }
        self.next_lsn += 1
        where = " (in handler)" if in_handler else ""
        self.trace.append(
            f"upsert {path} kind={kind} change={change} same_mtime={same_mtime} "
            f"-> lsn={lsn} retry_count={retry_count}{where}"
        )

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
        if m["kind"] != "episode":
            m.update(status="failed", retryable=False, error=True)
            return
        if m["retry_count"] >= MAX_TOTAL:
            m.update(status="failed", retryable=False, error=True)
            return
        if m["change"] == "deleted":
            m.update(status="done", retryable=None, error=False)
            return
        outcome = scripted.get(path, "ok")
        if outcome in ("ok", "vanished", "transient_then_ok"):
            m.update(status="done", retryable=None, error=False)
        elif outcome == "boom":
            m.update(status="failed", retryable=False, error=True)
        elif outcome == "transient":
            r = m["retry_count"] + MAX_RETRY
            m.update(
                status="failed", retryable=r < MAX_TOTAL, retry_count=r, error=True
            )
        elif outcome == "crash":
            pass  # stays processing

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
        from sqlalchemy import update

        from everos.core.persistence.sqlite import session_scope

        async def unguarded_mark_done(md_path: str) -> None:
            async with session_scope(machine.factory) as s:
                await s.execute(
                    update(MdChangeState)
                    .where(MdChangeState.md_path == md_path)
                    .values(status="done")
                )
                await s.commit()
            machine.marked.add(md_path)

        machine.repo.mark_done = unguarded_mark_done  # type: ignore[method-assign]
        machine.upsert(PATHS[0], "episode", "added", False)
        machine.drain(["ok"], [(PATHS[0], "episode", "modified", False)])
        with pytest.raises(AssertionError):
            machine.table_matches_model()
    finally:
        machine.teardown()
