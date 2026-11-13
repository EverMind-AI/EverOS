"""The purge runner: one in-process loop carrying the active job to its end.

One attempt runs steps 1-9 (spec §5.6): clear the scope from the inbox,
markdown, the index, SQLite, the OME run history and the md change ledger;
wait until no work on input taken in before the purge is in flight; clear
again; persist the terminal state. Nothing is blocked meanwhile. Every step
is idempotent, so an attempt that stops anywhere (an error, a restart) is
simply run again.
"""

from __future__ import annotations

__all__ = ["PurgeRunner", "count_index", "payload_in_scope", "scope_filter"]

import asyncio
import contextlib
import json
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sqlalchemy.exc

from everos.core.errors import (
    AppError,
    InfrastructureError,
    VectorStoreBusyError,
    VectorStoreError,
)
from everos.core.failpoints import afailpoint
from everos.core.observability.logging import get_logger
from everos.core.purge_marks import PurgeMarks, PurgeScope
from everos.core.utils.datetime import from_iso_format, get_utc_now
from everos.memory.ports import Filter, all_of, eq

from ._views import PurgeStep, step_stores, steps_from_json, steps_to_json

if TYPE_CHECKING:
    from datetime import datetime

    from everos.config import PurgeSettings
    from everos.infra.cascade import DeleteFence
    from everos.infra.ome import OfflineEngine
    from everos.infra.persistence.index import IndexRepositories
    from everos.infra.persistence.markdown import MarkdownScopePurger
    from everos.infra.persistence.sqlite import PurgeJob, PurgeJobRepo, ScopePurgeRepo

logger = get_logger(__name__)

MAX_RETRIES = 5
"""Retryable failures a job survives; the next one ends it ``failed``."""
MAX_ATTEMPTS = 20
"""Attempts a job may start (a crash mid-step counts no failure, so a job
that keeps killing the server would otherwise rerun forever)."""
_DRAIN_POLL_S = 0.1
_DRAIN_SAVE_S = 5.0
"""How often the wait persists its running time (and so the heartbeat)."""


def _backoff(n: int) -> float:
    return min(2.0**n, 60.0)


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, sqlalchemy.exc.OperationalError):
        return "database is locked" in str(exc)
    return isinstance(exc, InfrastructureError | OSError)


@contextlib.contextmanager
def _index_errors() -> Iterator[None]:
    """A backend's own error type (pymilvus raises its own) becomes a
    retryable :class:`VectorStoreError`: an index that cannot be reached
    reruns the attempt."""
    try:
        yield
    except AppError:
        raise
    except Exception as exc:
        raise VectorStoreError(f"index: {exc!r}"[:300]) from exc


async def count_index(index: Any, scope: PurgeScope) -> int:
    """Rows of ``scope`` across every index table."""
    where = scope_filter(scope)
    total = 0
    with _index_errors():
        for repo in index.all():
            total += await (repo.count() if where is None else repo.count_where(where))
    return total


def scope_filter(scope: PurgeScope) -> Filter | None:
    """The index predicate of ``scope`` (``None``: every row)."""
    if scope.kind == "space":
        return None
    if scope.kind == "app":
        return eq("app_id", scope.app_id)
    return all_of(eq("app_id", scope.app_id), eq("project_id", scope.project_id))


def _event_ts(payload: str, fallback: datetime) -> datetime:
    """A stored OME event's ``ts`` (its input's origin); ``fallback`` if absent."""
    try:
        data = json.loads(payload)
        return from_iso_format(data["ts"])
    except (ValueError, TypeError, KeyError):
        return fallback


def payload_in_scope(scope: PurgeScope, payload: str) -> bool | None:
    """Whether a stored OME event belongs to ``scope``; ``None``: unscoped."""
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    app_id, project_id = data.get("app_id"), data.get("project_id")
    if not isinstance(app_id, str) or not isinstance(project_id, str):
        return None
    return scope.covers(app_id, project_id)


@dataclass
class _Run:
    """One attempt's view of its job."""

    job_id: str
    scope: PurgeScope
    at: datetime
    """``T``: when the purge was confirmed."""
    steps: dict[str, PurgeStep]
    run_seconds: float
    retries: int
    fence_tokens: frozenset[int] | None = None
    at_store: str | None = None
    """The store whose step runs or comes next (blamed when the budget ends)."""
    marker: str = field(init=False)

    def __post_init__(self) -> None:
        self.marker = json.dumps({"purged_by": self.job_id}, separators=(",", ":"))


class PurgeRunner:
    """Runs the active purge job of one root, healing its own loop."""

    def __init__(
        self,
        *,
        settings: PurgeSettings,
        marks: PurgeMarks,
        jobs: PurgeJobRepo,
        scope_repo: ScopePurgeRepo,
        markdown: MarkdownScopePurger,
        index: IndexRepositories,
        index_store: str,
        engine: OfflineEngine,
        fence: DeleteFence,
        scope_of: Callable[[str, str | None, str | None], PurgeScope],
        backoff: Callable[[int], float] = _backoff,
        poll_seconds: float = 5.0,
    ) -> None:
        self._settings = settings
        self._marks = marks
        self._jobs = jobs
        self._scope_repo = scope_repo
        self._markdown = markdown
        self._index = index
        self._index_store = index_store
        self._engine = engine
        self._fence = fence
        self._scope_of = scope_of
        self._backoff = backoff
        self._poll = poll_seconds
        self._task: asyncio.Task[None] | None = None
        self._wake: asyncio.Event | None = None
        self._stopping = False
        self._restarts = 0

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stopping = False
        self._wake = asyncio.Event()
        self._spawn()

    async def stop(self) -> None:
        """Cancel the loop: a job mid-attempt keeps ``running``."""
        self._stopping = True
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def wake(self) -> None:
        """A job was just confirmed; the 5 s poll covers a lost wake."""
        if self._wake is not None:
            self._wake.set()

    def _spawn(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="purge-runner")
        self._task.add_done_callback(self._on_loop_done)

    def _on_loop_done(self, task: asyncio.Task[None]) -> None:
        if self._stopping or task is not self._task:
            return
        self._restarts += 1
        logger.error(
            "purge_runner_loop_died",
            error="cancelled" if task.cancelled() else repr(task.exception()),
            restarts=self._restarts,
        )
        asyncio.get_running_loop().call_later(
            self._backoff(self._restarts), self._respawn
        )

    def _respawn(self) -> None:
        if not self._stopping:
            self._spawn()

    async def _loop(self) -> None:
        assert self._wake is not None
        while True:
            await afailpoint("purge.runner_loop")
            job = await self._jobs.active()
            if job is None:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), timeout=self._poll)
                self._wake.clear()
                continue
            await self._attempt(job)

    # ── one attempt ──────────────────────────────────────────────────────

    async def _attempt(self, job: PurgeJob) -> None:
        stores = step_stores(self._index_store)
        known = {s.store: s for s in steps_from_json(job.steps_json)}
        run = _Run(
            job_id=job.job_id,
            scope=self._scope_of(job.scope, job.app_id, job.project_id),
            at=job.created_at,
            steps={s: known.get(s, PurgeStep(s, "pending")) for s in stores},
            run_seconds=job.run_seconds,
            retries=job.retries,
        )
        left = self._settings.run_budget_seconds - job.run_seconds
        if left <= 0:
            await self._end(run, "failed", "purge run budget exhausted")
            return
        if job.attempts >= MAX_ATTEMPTS:
            await self._end(
                run, "failed", f"purge stopped after {job.attempts} attempts"
            )
            return
        await self._jobs.update_progress(
            job.job_id,
            status="running",
            attempts=job.attempts + 1,
            started_at=job.started_at or get_utc_now(),
        )
        outcome, error = await self._try_steps(run, left)
        if outcome == "retry":
            run.retries += 1
            if run.retries > MAX_RETRIES:
                await self._end(run, "failed", error)
                return
            logger.warning(
                "purge_attempt_failed",
                job_id=run.job_id,
                retries=run.retries,
                error=error,
            )
            await self._jobs.update_progress(
                run.job_id, retries=run.retries, last_error=error
            )
            await asyncio.sleep(self._backoff(run.retries))
            return
        await self._end(run, outcome, error)

    async def _try_steps(self, run: _Run, budget: float) -> tuple[str, str | None]:
        deadline = asyncio.timeout(budget)
        try:
            async with deadline:
                await self._steps(run)
        except TimeoutError as exc:
            if deadline.expired():
                error = "purge run budget exhausted"
                stuck = run.steps.get(run.at_store or "")
                if stuck is not None and stuck.state != "failed":
                    # Out of time between steps, or in the wait for work in
                    # flight: blame the step that was to run next.
                    run.steps[stuck.store] = PurgeStep(
                        stuck.store, "failed", stuck.deleted, error
                    )
                return "failed", error
            return "retry", repr(exc)[:500]
        except Exception as exc:
            return ("retry" if _retryable(exc) else "failed"), repr(exc)[:500]
        return "succeeded", None

    async def _steps(self, run: _Run) -> None:
        await self._clear(run)
        await afailpoint("purge.before_step7")
        await self._drain(run)
        await afailpoint("purge.before_step8")
        await self._clear(run)

    async def _clear(self, run: _Run) -> None:
        """Steps 1-6, each persisted as it ends.

        The index and SQLite rows are counted before step 2: moving the
        markdown lets the cascade delete some of them before steps 3-4 get
        there, and those were purged all the same.
        """
        before = {
            self._index_store: await count_index(self._index, run.scope),
            "sqlite": sum((await self._scope_repo.count(run.scope)).values()),
        }
        for n, store, step in (
            (1, "inbox", self._step_inbox),
            (2, "markdown", self._step_markdown),
            (3, self._index_store, self._step_index),
            (4, "sqlite", self._step_sqlite),
            (5, "ome", self._step_ome),
            (6, "md_change_state", self._step_change_state),
        ):
            run.at_store = store
            await afailpoint(f"purge.before_step{n}")
            await self._run_step(run, store, step, at_least=before.get(store, 0))
        run.at_store = None

    async def _run_step(
        self,
        run: _Run,
        store: str,
        step: Callable[[_Run], Awaitable[int]],
        *,
        at_least: int = 0,
    ) -> None:
        started = time.monotonic()
        done = run.steps[store]
        try:
            deleted = max(await self._busy_retry(run, step), at_least)
        except BaseException as exc:
            run.steps[store] = PurgeStep(store, "failed", done.deleted, repr(exc)[:500])
            with contextlib.suppress(Exception):
                await self._save(run, started)
            raise
        run.steps[store] = PurgeStep(store, "done", done.deleted + deleted)
        await self._save(run, started)

    async def _busy_retry(
        self, run: _Run, step: Callable[[_Run], Awaitable[int]]
    ) -> int:
        """A contended index table is retried within the step, unbounded but
        inside the attempt's budget; it is no failure of the attempt."""
        n = 0
        while True:
            try:
                return await step(run)
            except VectorStoreBusyError:
                n += 1
                logger.info("purge_step_busy_retry", job_id=run.job_id, retry=n)
                await asyncio.sleep(self._backoff(n))

    async def _save(self, run: _Run, started: float) -> None:
        run.run_seconds += time.monotonic() - started
        await self._jobs.update_progress(
            run.job_id,
            run_seconds=run.run_seconds,
            steps_json=steps_to_json(list(run.steps.values())),
        )

    # ── steps 1-6 ────────────────────────────────────────────────────────

    async def _step_inbox(self, run: _Run) -> int:
        """Rows accepted before ``T`` only: later ones are new writes."""
        discarded = await self._scope_repo.discard_queued_inbox(
            run.scope, job_id=run.job_id, before=run.at
        )
        removed = await self._scope_repo.delete_inbox_history(run.scope, before=run.at)
        # The unprocessed messages go now, not at step 4: a flush meanwhile
        # would cut them into a memcell whose work counts as new input.
        await self._scope_repo.delete_buffer(run.scope)
        return discarded + removed

    async def _step_markdown(self, run: _Run) -> int:
        await self._markdown.restore_counters(run.scope, run.job_id)
        # An earlier attempt that stopped after its move left this job's trash
        # behind; its counters are back in place now, so it can go before the
        # scope is moved there again (a rename onto it would fail).
        await self._markdown.empty(run.scope, run.job_id)
        moved = await self._markdown.move_out(run.scope, run.job_id)
        reading = self._fence.in_flight()  # projections that may have read S's files
        await self._markdown.restore_counters(run.scope, run.job_id)
        await self._markdown.empty(run.scope, run.job_id)
        if run.fence_tokens is None:  # first pass: step 7 waits for them
            run.fence_tokens = reading
        else:  # second pass: they end before the index is cleared
            await afailpoint("purge.pass2.before_index")
            while True:  # the projection's state; nothing to await
                if not reading & self._fence.in_flight():
                    break
                await asyncio.sleep(_DRAIN_POLL_S)
        return moved

    async def _step_index(self, run: _Run) -> int:
        where = scope_filter(run.scope)
        deleted = 0
        for repo in self._index.all():
            with _index_errors():
                count = await (
                    repo.count() if where is None else repo.count_where(where)
                )
                await afailpoint("purge.step3.delete")
                if where is None:
                    await repo.delete_all()
                else:
                    await repo.delete(where)
            deleted += count
        return deleted

    async def _step_sqlite(self, run: _Run) -> int:
        return sum((await self._scope_repo.delete(run.scope)).values())

    async def _step_ome(self, run: _Run) -> int:
        return await self._engine.redact_payloads(
            lambda payload: payload_in_scope(run.scope, payload) is True, run.marker
        )

    async def _step_change_state(self, run: _Run) -> int:
        return await self._scope_repo.delete_change_state(run.scope)

    # ── step 7 ───────────────────────────────────────────────────────────

    async def _drain(self, run: _Run) -> None:
        """Wait until no work on input taken in before ``T`` is in flight.

        Asked again every poll, so work such input starts late (an
        extraction created after the first pass) is waited for too. No limit
        of its own: the attempt's budget bounds it, and the time spent is
        added to ``run_seconds`` (persisted every few seconds) so a restart
        does not reset it.
        """
        run.at_store = next(iter(run.steps))  # the second pass starts there
        saved = time.monotonic()
        while not await self._drained(run):
            await asyncio.sleep(_DRAIN_POLL_S)
            if time.monotonic() - saved >= _DRAIN_SAVE_S:
                await self._save(run, saved)
                saved = time.monotonic()
        await self._save(run, saved)

    async def _drained(self, run: _Run) -> bool:
        if (run.fence_tokens or frozenset()) & self._fence.in_flight():
            return False
        if self._marks.tracked_before(run.at, run.scope):
            return False
        for r in await self._engine.running_runs():
            in_scope = payload_in_scope(run.scope, r.event_payload)
            if in_scope is False:
                continue
            # A tick has no input of its own: it reads what is there when it
            # starts, so its start is what counts.
            began = (
                r.started_at
                if in_scope is None
                else _event_ts(r.event_payload, r.started_at)
            )
            if began < run.at:
                return False
        return True

    # ── step 9 ───────────────────────────────────────────────────────

    async def _end(self, run: _Run, status: str, error: str | None) -> None:
        """Persist the terminal state (retrying the write); the job is done."""
        await afailpoint("purge.before_step9")
        attempt = 0
        while True:
            try:
                await afailpoint("purge.finalize")
                await self._jobs.update_progress(
                    run.job_id, run_seconds=run.run_seconds, retries=run.retries
                )
                await self._jobs.finish(
                    run.job_id,
                    status=status,
                    finished_at=get_utc_now(),
                    steps_json=steps_to_json(list(run.steps.values())),
                    last_error=error,
                )
                break
            except Exception as exc:
                attempt += 1
                logger.warning(
                    "purge_finalize_failed",
                    job_id=run.job_id,
                    attempt=attempt,
                    error=repr(exc)[:200],
                )
                await asyncio.sleep(self._backoff(attempt))
        self._marks.lift_mark()
        logger.info("purge_finished", job_id=run.job_id, status=status, error=error)
