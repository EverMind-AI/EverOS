"""What the purge service answers: a request's summary and a job's state.

Shaped as Cloud's ``PurgeRequestData`` / ``PurgeJobData`` (the HTTP layer
maps them field for field); the stores are named as EverOS has them.
"""

from __future__ import annotations

__all__ = [
    "PurgeJobView",
    "PurgeRequestResult",
    "PurgeStep",
    "PurgeSummary",
    "step_stores",
    "steps_from_json",
    "steps_to_json",
]

import json
from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass(frozen=True)
class PurgeSummary:
    """What a scope holds, counted per store; ``None``: the count timed out."""

    markdown: dict[str, int | None]
    markdown_total: int | None
    sqlite: dict[str, int | None]
    sqlite_total: int | None
    index_store: str
    """``lancedb`` or ``milvus``: which ``*_total`` field ``index_total`` is."""
    index_total: int | None
    complete: bool
    counted_at: datetime
    comment: str
    error: str | None = None


@dataclass(frozen=True)
class PurgeRequestResult:
    nonce: str
    expires_at: datetime
    scope: str
    app_id: str | None
    project_id: str | None
    summary: PurgeSummary


@dataclass(frozen=True)
class PurgeStep:
    store: str
    state: str
    """``pending`` | ``done`` | ``failed``."""
    deleted: int = 0
    error: str | None = None


@dataclass(frozen=True)
class PurgeJobView:
    job_id: str
    status: str
    scope: str
    app_id: str | None
    project_id: str | None
    created_at: datetime
    started_at: datetime | None
    heartbeat_at: datetime | None
    finished_at: datetime | None
    steps: tuple[PurgeStep, ...]


def step_stores(index_store: str) -> tuple[str, ...]:
    """The stores of steps 1-6, in order (step 3 names the index backend)."""
    return ("inbox", "markdown", index_store, "sqlite", "ome", "md_change_state")


def steps_to_json(steps: list[PurgeStep]) -> str:
    return json.dumps([asdict(step) for step in steps])


def steps_from_json(raw: str) -> list[PurgeStep]:
    return [PurgeStep(**item) for item in json.loads(raw or "[]")]
