"""What the ``async_mode`` queue exchanges: the delivery call and the task views.

:class:`Deliver` is the memorize call the worker makes per row;
:class:`TaskItem` / :class:`TaskPage` are the Cloud-shaped views of inbox
rows the task readers return, and :class:`QueueHealth` is the queue's block
of ``GET /health``.
"""

from __future__ import annotations

__all__ = [
    "QUEUE_LOGGER",
    "TASK_STATUSES",
    "Deliver",
    "QueueHealth",
    "TaskItem",
    "TaskPage",
    "TaskStatus",
    "error_summary",
    "to_task_item",
]

import datetime as dt
from typing import Any, Literal, Protocol

from pydantic import BaseModel

from everos.memory.models import MemorizeTask
from everos.memory.ports import LOST_STATUS_WRITE, REPLAYS_EXHAUSTED_PREFIX

QUEUE_LOGGER = "everos.service.memorize_queue"
"""The logger the queue logs under, kept from when it was one module."""


class Deliver(Protocol):
    """The memorize call the worker makes for one accepted request."""

    async def __call__(
        self,
        payload: dict[str, Any],
        *,
        is_final: bool,
        defer_extraction: bool,
    ) -> object: ...


_ERROR_SUMMARY_CHARS = 500
# Exceptions from these packages carry SQL text and bound parameters, i.e.
# the request body, in their message; GET /tasks must not echo that back.
_OPAQUE_ERROR_MODULES = ("sqlalchemy", "sqlite3", "aiosqlite")


def error_summary(exc: BaseException) -> str:
    """``last_error`` for a failed batch: the type, plus the message unless
    the exception comes from the storage layer (whose messages embed SQL and
    parameters)."""
    name = type(exc).__name__
    module = type(exc).__module__ or ""
    if module.split(".", 1)[0] in _OPAQUE_ERROR_MODULES:
        return f"{name}: storage error (details in the server log)"
    return f"{name}: {exc}"[:_ERROR_SUMMARY_CHARS]


TaskStatus = Literal["queued", "processing", "success", "failed"]
TASK_STATUSES: tuple[str, ...] = ("queued", "processing", "success", "failed")


class TaskItem(BaseModel):
    """``GET /tasks/{task_id}`` -- one accepted request, in the Cloud's shape.

    ``id`` is the request id of the accepting call. Cloud also emits
    ``pending`` (accepted, boundary not run yet); this build never does, so a
    client written for Cloud sees a subset of the statuses it already handles.
    """

    id: str
    status: TaskStatus
    task_type: Literal["memory_add", "memory_flush"]
    error: str | None = None
    error_code: str | None = None
    created_at: dt.datetime
    finished_at: dt.datetime | None = None


class TaskPage(BaseModel):
    """``GET /tasks`` -- one page of :class:`TaskItem`, newest first."""

    total: int
    page: int
    page_size: int
    items: list[TaskItem]


class QueueHealth(BaseModel):
    """The ``memorize_queue`` block of ``GET /health``.

    ``healthy`` is the delivery loop being alive. The probe only observes:
    a dead loop is reported as such, and it is the loop's own exit callback
    (and the next ``enqueue``) that relaunches it, so ``loop_restarts``
    counts every time that happened. The counts are for alerting: a growing
    ``oldest_pending_age_s`` with a live loop means delivery is slower than
    arrival (or the LLM is down).
    """

    healthy: bool
    reasons: list[str]
    loop_alive: bool
    accepting: bool
    pending: int
    oldest_pending_age_s: float | None
    failed: int
    loop_restarts: int
    last_loop_error_at: dt.datetime | None


def to_task_item(row: MemorizeTask) -> TaskItem:
    """The Cloud-shaped view of one inbox row."""
    terminal = row.status in ("success", "failed")
    error_code: str | None = None
    if row.last_error == LOST_STATUS_WRITE:
        error_code = "status_write_lost"
    elif row.last_error and row.last_error.startswith(REPLAYS_EXHAUSTED_PREFIX):
        error_code = "replays_exhausted"
    elif row.last_error and ": " in row.last_error:
        error_code = row.last_error.split(": ", 1)[0]  # the exception type
    return TaskItem(
        id=row.request_id,
        status=row.status,  # type: ignore[arg-type]
        task_type="memory_flush" if row.is_final else "memory_add",
        error=row.last_error,
        error_code=error_code,
        created_at=row.created_at,
        finished_at=row.updated_at if terminal else None,
    )
