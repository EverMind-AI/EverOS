"""Purge: count a scope, hand out a one-time nonce, start and report a job."""

from __future__ import annotations

__all__ = ["PurgeService"]

import asyncio
import secrets
import uuid
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from everos.core.errors import (
    NotFoundError,
    PurgeInProgressError,
    PurgeNonceInvalidError,
    StorageError,
)
from everos.core.failpoints import afailpoint
from everos.core.observability.logging import get_logger
from everos.core.purge_marks import MARKS, PurgeMarks, PurgeScope, Tombstone
from everos.core.utils.datetime import get_utc_now

from ._runner import PurgeRunner, count_index
from ._views import (
    PurgeJobView,
    PurgeRequestResult,
    PurgeStep,
    PurgeSummary,
    step_stores,
    steps_from_json,
    steps_to_json,
)

if TYPE_CHECKING:
    from everos.config import PurgeSettings
    from everos.infra.cascade import DeleteFence
    from everos.infra.ome import OfflineEngine
    from everos.infra.persistence.index import IndexRepositories
    from everos.infra.persistence.markdown import MarkdownScopePurger
    from everos.infra.persistence.sqlite import (
        PurgeJob,
        PurgeJobRepo,
        ScopePurgeRepo,
    )
    from everos.memory.ports import Scope

logger = get_logger(__name__)

ScopeOf = Callable[[str, str | None, str | None], PurgeScope]


@dataclass
class _Nonce:
    scope: PurgeScope
    expires_at: datetime


@dataclass
class PurgeService:
    """Online clearing of one memory root (the fourth application facade).

    Everything is injected; the bootstrap builds the stores.
    """

    settings: PurgeSettings
    root: Path
    scope_of: ScopeOf
    jobs: PurgeJobRepo
    scope_repo: ScopePurgeRepo
    markdown: MarkdownScopePurger
    index: IndexRepositories
    index_store: str
    engine: OfflineEngine
    fence: DeleteFence
    _nonces: dict[str, _Nonce] = field(default_factory=dict, init=False)
    _reserved: set[str] = field(default_factory=set, init=False)
    _recording: asyncio.Event | None = field(default=None, init=False)
    """Set while a confirm writes its job row (active already, the row not yet)."""
    _runner: PurgeRunner | None = field(default=None, init=False)

    @property
    def marks(self) -> PurgeMarks:
        return MARKS.for_root(self.root)

    # ── request ──────────────────────────────────────────────────────────

    async def request(
        self, kind: str, app_id: str | None, project_id: str | None
    ) -> PurgeRequestResult:
        """Count what ``kind`` / ``app_id`` / ``project_id`` holds; a nonce."""
        scope = self.scope_of(kind, app_id, project_id)
        summary = await self.summarize(scope)
        nonce = secrets.token_urlsafe(24)
        now = get_utc_now()
        expires_at = now + timedelta(seconds=self.settings.nonce_ttl_seconds)
        self._nonces = {n: v for n, v in self._nonces.items() if v.expires_at > now}
        self._nonces[nonce] = _Nonce(scope, expires_at)
        return PurgeRequestResult(
            nonce=nonce,
            expires_at=expires_at,
            scope=scope.kind,
            app_id=scope.app_id,
            project_id=scope.project_id,
            summary=summary,
        )

    async def summarize(self, scope: PurgeScope) -> PurgeSummary:
        markdown = await self._count("markdown", lambda: self.markdown.count(scope))
        sqlite = await self._count("sqlite", lambda: self.scope_repo.count(scope))
        index = await self._count("index", lambda: count_index(self.index, scope))
        errors = [e for e in (markdown[1], sqlite[1], index[1]) if e]
        comment = (
            "Counts are what the scope holds now; a purge deletes what it holds"
            " when it runs. Markdown counts memory entries (not today's empty"
            " counter files or .taxonomy.md)."
        )
        if self.index_store == "milvus":
            comment += " milvus_total counts this root's collections."
        return PurgeSummary(
            markdown=dict(markdown[0] or {}),
            markdown_total=_total(markdown[0]),
            sqlite=dict(sqlite[0] or {}),
            sqlite_total=_total(sqlite[0]),
            index_store=self.index_store,
            index_total=index[0],
            complete=not errors,
            counted_at=get_utc_now(),
            comment=comment,
            error="; ".join(errors) or None,
        )

    async def _count[T](
        self, store: str, count: Callable[[], Awaitable[T]]
    ) -> tuple[T | None, str | None]:
        try:
            async with asyncio.timeout(self.settings.count_timeout_seconds):
                await afailpoint(f"purge.count.{store}")
                return await count(), None
        except TimeoutError:
            return None, f"{store}: count timed out"
        except Exception as exc:  # a summary never fails the request
            logger.warning("purge_count_failed", store=store, error=repr(exc)[:200])
            return None, f"{store}: count failed"

    # ── confirm ──────────────────────────────────────────────────────────

    async def confirm(self, nonce: str) -> PurgeJobView:
        """Start the purge ``nonce`` was issued for (or report the active one).

        Raises:
            PurgeNonceInvalidError: Unknown, expired, used or in use.
            StorageError: The job row could not be written; nothing started
                and the nonce stays valid.
        """
        marks = self.marks
        while True:
            entry = self._nonces.get(nonce)
            if (
                entry is None
                or entry.expires_at <= get_utc_now()
                or nonce in self._reserved
            ):
                raise PurgeNonceInvalidError("unknown, expired or used purge nonce")
            active = marks.active_job_id
            if active is None:
                break
            if self._recording is not None:
                # That job's confirm is still writing its row: wait, then
                # decide again (it may also give up).
                await self._recording.wait()
                continue
            view = await self.get(active)
            # Used only once the running job can be returned.
            self._nonces.pop(nonce, None)
            return view
        # Synchronous from the last check on: two confirms of one nonce, or
        # of two nonces, cannot both get past here.
        self._reserved.add(nonce)
        scope = entry.scope
        job_id = f"pj_{uuid.uuid4().hex}"
        at = get_utc_now()
        tombstone = Tombstone(scope, at)
        marks.raise_mark(job_id, scope)
        marks.add_tombstone(tombstone)
        recording = self._recording = asyncio.Event()
        try:
            try:
                await afailpoint("purge.confirm_insert")
                row = await self.jobs.insert(
                    job_id=job_id,
                    scope=scope.kind,
                    app_id=scope.app_id,
                    project_id=scope.project_id,
                    created_at=at,
                    steps_json=steps_to_json(
                        [PurgeStep(s, "pending") for s in step_stores(self.index_store)]
                    ),
                )
            except Exception as exc:
                # ponytail: a stale event this tombstone dropped in these few ms
                # stays dropped; it was in S, which the caller is clearing anyway.
                self._undo(scope, tombstone, nonce)
                logger.warning("purge_job_insert_failed", error=repr(exc)[:200])
                raise StorageError("could not record the purge job; retry") from exc
            if row is None:  # another process holds the active row
                self._undo(scope, tombstone, nonce)
                other = await self.jobs.active()
                if other is None:
                    raise StorageError("could not record the purge job; retry")
                return self._view(other)
        finally:
            self._recording = None
            recording.set()
        self._nonces.pop(nonce, None)
        self._reserved.discard(nonce)
        logger.info("purge_confirmed", job_id=job_id, scope=scope.kind)
        if not await afailpoint("purge.confirm_wake") and self._runner is not None:
            self._runner.wake()
        return self._view(row)

    def _undo(self, scope: PurgeScope, tombstone: Tombstone, nonce: str) -> None:
        marks = self.marks
        marks.lift_mark()
        marks.remove_tombstone(tombstone)
        self._reserved.discard(nonce)

    # ── reads ────────────────────────────────────────────────────────────

    async def get(self, job_id: str) -> PurgeJobView:
        """The job's current state.

        Raises:
            NotFoundError: No such job.
        """
        row = await self.jobs.get(job_id)
        if row is None:
            raise NotFoundError(f"purge job {job_id!r} not found")
        return self._view(row)

    def track(self, origin: datetime) -> AbstractContextManager[None]:
        """Register a write request taken in at ``origin`` while it runs."""
        return self.marks.track(origin)

    def refuse_writes_to(self, app_id: str, project_id: str) -> None:
        """Refuse a write into the range the running purge is clearing.

        Accepting it would answer success for data the purge's second pass
        may then delete in part.

        Raises:
            PurgeInProgressError: The running purge covers
                ``(app_id, project_id)``.
        """
        marks = self.marks
        active = marks.active_scope
        if active is not None and active.covers(app_id, project_id):
            job_id = marks.active_job_id
            raise PurgeInProgressError(
                f"purge job {job_id} is clearing this app / project; poll "
                f"GET /api/v2/memory/purge/{job_id} and resend after it ends"
            )

    def covers_trash(self, scope: Scope) -> bool:
        """Whether a ``/delete`` trash folder belongs to the running purge."""
        active = self.marks.active_scope
        return active is not None and active.covers(scope.app_id, scope.project_id)

    def _view(self, row: PurgeJob) -> PurgeJobView:
        return PurgeJobView(
            job_id=row.job_id,
            status=row.status,
            scope=row.scope,
            app_id=row.app_id,
            project_id=row.project_id,
            created_at=row.created_at,
            started_at=row.started_at,
            heartbeat_at=row.updated_at,
            finished_at=row.finished_at,
            steps=tuple(steps_from_json(row.steps_json)),
        )

    # ── lifecycle ────────────────────────────────────────────────────────

    async def restore(self) -> None:
        """Rebuild the tombstones from ``system.db`` (startup, before any
        background work replays)."""
        marks = self.marks
        rows = await self.jobs.all_rows()
        marks.set_tombstones(
            Tombstone(self.scope_of(r.scope, r.app_id, r.project_id), r.created_at)
            for r in rows
        )
        active = await self.jobs.active()
        if active is not None:
            scope = self.scope_of(active.scope, active.app_id, active.project_id)
            marks.raise_mark(active.job_id, scope)
            logger.info("purge_resumed", job_id=active.job_id)
        await self.markdown.sweep_inactive(active.job_id if active else None)

    def start(self) -> None:
        self._runner = PurgeRunner(
            settings=self.settings,
            marks=self.marks,
            jobs=self.jobs,
            scope_repo=self.scope_repo,
            markdown=self.markdown,
            index=self.index,
            index_store=self.index_store,
            engine=self.engine,
            fence=self.fence,
            scope_of=self.scope_of,
        )
        self._runner.start()

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.stop()


def _total(counts: dict[str, int] | None) -> int | None:
    return None if counts is None else sum(counts.values())
