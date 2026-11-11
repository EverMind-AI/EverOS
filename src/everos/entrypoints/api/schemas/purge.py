"""Purge DTOs (``/api/v2/memory/purge``): shaped as Cloud's, stores renamed.

A purge is a hard, irreversible delete of the whole memory root (``space``),
one app, or one project of one app, run as an async job. ``POST
/memory/purge`` counts what the scope holds and returns a nonce (nothing is
deleted); ``POST /memory/purge/confirm`` starts the job within the nonce's
lifetime. The scope always comes from the nonce, never from the confirm body.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from everos.service.purge import PurgeJobView, PurgeRequestResult

from .ids import AppScopeId, ProjectScopeId

PurgeScopeName = Literal["space", "app", "project"]
PurgeStatus = Literal["queued", "running", "succeeded", "failed"]
PurgeStepState = Literal["pending", "done", "failed"]
PurgeNonce = Annotated[
    str, StringConstraints(min_length=16, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
]
PurgeJobId = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
]


class PurgeRequest(BaseModel):
    """Step 1: ask for a purge. Nothing is deleted."""

    model_config = ConfigDict(extra="forbid")

    scope: PurgeScopeName = Field(
        description=(
            "`space` purges the whole memory root; `app` one app in it (`app_id`); "
            "`project` one project of one app (`app_id` + `project_id`)."
        )
    )
    app_id: AppScopeId | None = Field(
        default=None, description="Required for `app` and `project` scope."
    )
    project_id: ProjectScopeId | None = Field(
        default=None, description="Required for `project` scope."
    )

    @model_validator(mode="after")
    def _scope_fields(self) -> PurgeRequest:
        # Each scope takes exactly its own ids: a stray id on a wider scope would
        # read as a narrower purge than the one that runs.
        if self.scope == "space":
            if self.app_id is not None or self.project_id is not None:
                raise ValueError("scope=space takes neither app_id nor project_id")
        elif self.scope == "app":
            if not self.app_id:
                raise ValueError("scope=app requires app_id")
            if self.project_id is not None:
                raise ValueError("scope=app takes no project_id")
        elif not self.app_id or not self.project_id:
            raise ValueError("scope=project requires app_id and project_id")
        return self


class PurgeSummary(BaseModel):
    """What the purge would delete right now, counted per store.

    A snapshot at request time: data written before the purge runs is deleted
    too. A count that did not finish in time is ``null`` and ``complete`` is
    ``false``. The index count is under the field of the backend this server
    runs (``lancedb_total`` or ``milvus_total``); the other one is ``null``.
    """

    markdown: dict[str, int | None] = Field(
        description="Memory entries per kind in the Markdown files (the truth)."
    )
    markdown_total: int | None = Field(
        description="Sum of `markdown`; null if not counted."
    )
    sqlite: dict[str, int | None] = Field(description="Rows per system.db table.")
    sqlite_total: int | None = Field(
        description="Sum of `sqlite`; null if not counted."
    )
    lancedb_total: int | None = Field(
        default=None, description="LanceDB index rows (derived); null if not counted."
    )
    milvus_total: int | None = Field(
        default=None, description="Milvus index rows (derived); null if not counted."
    )
    complete: bool = Field(description="False when any count timed out or failed.")
    counted_at: datetime
    comment: str = Field(description="What the counts include and exclude.")
    error: str | None = Field(
        default=None, description="Which counts did not finish; null when `complete`."
    )


class PurgeRequestData(BaseModel):
    """The nonce to confirm, and what it will purge."""

    nonce: str = Field(description="Send to /purge/confirm before `expires_at`.")
    expires_at: datetime
    scope: PurgeScopeName
    app_id: str | None = None
    project_id: str | None = None
    summary: PurgeSummary

    @classmethod
    def from_result(cls, result: PurgeRequestResult) -> PurgeRequestData:
        s = result.summary
        index = {f"{s.index_store}_total": s.index_total}
        return cls(
            nonce=result.nonce,
            expires_at=result.expires_at,
            scope=result.scope,  # type: ignore[arg-type]
            app_id=result.app_id,
            project_id=result.project_id,
            summary=PurgeSummary(
                markdown=s.markdown,
                markdown_total=s.markdown_total,
                sqlite=s.sqlite,
                sqlite_total=s.sqlite_total,
                complete=s.complete,
                counted_at=s.counted_at,
                comment=s.comment,
                error=s.error,
                **index,
            ),
        )


class PurgeConfirmRequest(BaseModel):
    """Step 2: confirm. The scope is the one bound to the nonce."""

    model_config = ConfigDict(extra="forbid")

    nonce: PurgeNonce


class PurgeStepData(BaseModel):
    """One store of one purge step, e.g. ``markdown`` or ``lancedb``."""

    store: str
    state: PurgeStepState
    deleted: int = 0
    error: str | None = None


class PurgeJobData(BaseModel):
    """A purge job as the caller sees it."""

    job_id: str
    status: PurgeStatus
    scope: PurgeScopeName
    app_id: str | None = None
    project_id: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    heartbeat_at: datetime | None = None
    finished_at: datetime | None = None
    steps: list[PurgeStepData] = Field(default_factory=list)

    @classmethod
    def from_view(cls, view: PurgeJobView) -> PurgeJobData:
        return cls(
            job_id=view.job_id,
            status=view.status,  # type: ignore[arg-type]
            scope=view.scope,  # type: ignore[arg-type]
            app_id=view.app_id,
            project_id=view.project_id,
            created_at=view.created_at,
            started_at=view.started_at,
            heartbeat_at=view.heartbeat_at,
            finished_at=view.finished_at,
            steps=[
                PurgeStepData(
                    store=s.store,
                    state=s.state,  # type: ignore[arg-type]
                    deleted=s.deleted,
                    error=s.error,
                )
                for s in view.steps
            ],
        )
