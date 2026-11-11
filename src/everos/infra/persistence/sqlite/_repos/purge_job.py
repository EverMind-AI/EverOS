"""The ``purge_job`` rows: insert, progress, terminal state."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from everos.infra.persistence.sqlite._base.repository import RepoBase
from everos.infra.persistence.sqlite._base.session import session_scope

from .._tables import PurgeJob


class PurgeJobRepo(RepoBase[PurgeJob]):
    """Purge jobs of one root; at most one is active (``active = 1``)."""

    model = PurgeJob

    async def insert(self, row: PurgeJob) -> PurgeJob | None:
        """Insert ``row``; ``None`` when another job is already active."""
        async with session_scope(self._factory) as s:
            s.add(row)
            try:
                await s.commit()
            except IntegrityError as exc:
                await s.rollback()
                if "purge_job.active" in str(exc.orig):
                    return None
                raise
            await s.refresh(row)
            return row

    async def get(self, job_id: str) -> PurgeJob | None:
        async with session_scope(self._factory) as s:
            return await s.get(PurgeJob, job_id)

    async def active(self) -> PurgeJob | None:
        async with session_scope(self._factory) as s:
            result = await s.execute(select(PurgeJob).where(PurgeJob.active == 1))
            return result.scalars().first()

    async def all_rows(self) -> list[PurgeJob]:
        """Every job, oldest first (the tombstones)."""
        async with session_scope(self._factory) as s:
            result = await s.execute(select(PurgeJob).order_by(PurgeJob.created_at))
            return list(result.scalars().all())

    async def update_progress(self, job_id: str, **fields: object) -> None:
        async with session_scope(self._factory) as s:
            await s.execute(
                update(PurgeJob).where(PurgeJob.job_id == job_id).values(**fields)
            )
            await s.commit()

    async def finish(
        self,
        job_id: str,
        *,
        status: str,
        finished_at: datetime,
        steps_json: str,
        last_error: str | None,
    ) -> None:
        """Write the terminal state and free the active slot, in one UPDATE."""
        await self.update_progress(
            job_id,
            status=status,
            finished_at=finished_at,
            steps_json=steps_json,
            last_error=last_error,
            active=None,
        )
