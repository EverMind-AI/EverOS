"""``purge_job`` — one row per confirmed purge of a project, an app or the root.

``created_at`` is the confirm time ``T`` (a tombstone: events stamped before
it never write the purged scope back); ``updated_at`` is the job's
heartbeat. ``active`` is 1 while the job runs and NULL once it is terminal:
the unique index on it (SQLite treats NULLs as distinct) allows at most one
running purge per root.
"""

from __future__ import annotations

from sqlmodel import Field

from everos.core.utils.datetime import UtcDatetime
from everos.infra.persistence.sqlite._base.base import BaseTable, UtcDateTimeColumn


class PurgeJob(BaseTable, table=True):
    """One purge job and its progress."""

    __tablename__ = "purge_job"  # type: ignore[assignment]

    job_id: str = Field(primary_key=True)
    scope: str
    """``space`` | ``app`` | ``project``."""
    app_id: str | None = None
    project_id: str | None = None
    status: str = "queued"
    """``queued`` | ``running`` | ``succeeded`` | ``failed``."""
    active: int | None = Field(default=1, unique=True)
    started_at: UtcDatetime | None = Field(default=None, sa_type=UtcDateTimeColumn)
    finished_at: UtcDatetime | None = Field(default=None, sa_type=UtcDateTimeColumn)
    attempts: int = 0
    retries: int = 0
    """Retryable failures so far; past the cap the job fails."""
    run_seconds: float = 0.0
    """Running time across attempts (the budget; downtime not counted)."""
    steps_json: str = "[]"
    last_error: str | None = None
