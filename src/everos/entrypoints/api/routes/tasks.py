"""``GET /tasks`` -- progress of asynchronous memorize work, Cloud-shaped.

The EverOS Cloud gateway answers ``GET /api/v2/tasks/{task_id}`` and
``GET /api/v2/tasks`` from its message bus; here the same paths read the
``memorize_inbox`` table, so one client polls both the same way. ``task_id``
is the ``request_id`` a ``202`` carried.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, Request

from everos.core.errors import InvalidInputError, NotFoundError
from everos.entrypoints.api.utils import extract_request_id
from everos.service import TASK_STATUSES, TaskItem, TaskPage, get_task, list_tasks

from .memorize import SuccessEnvelope

router = APIRouter(prefix="/tasks", tags=["tasks"])

_MAX_PAGE_SIZE = 100


@router.get("")
async def task_list(
    request: Request,
    session_id: str | None = None,
    status: str | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1)] = 20,
    app_id: str = "default",
    project_id: str = "default",
) -> SuccessEnvelope[TaskPage]:
    """Accepted requests, newest first; ``session_id`` / ``status`` narrow it.

    Drain a session before exiting by polling with ``status=queued`` and
    ``status=processing`` until both ``total`` are 0. ``page_size`` is capped
    at 100.
    """
    if status is not None and status not in TASK_STATUSES:
        raise InvalidInputError(
            f"status must be one of {', '.join(TASK_STATUSES)}; got {status!r}"
        )
    page_data = await list_tasks(
        session_id=session_id,
        status=status,
        page=page,
        page_size=min(page_size, _MAX_PAGE_SIZE),
        app_id=app_id,
        project_id=project_id,
    )
    return SuccessEnvelope(request_id=extract_request_id(request), data=page_data)


@router.get("/{task_id}")
async def task_status(task_id: str, request: Request) -> SuccessEnvelope[TaskItem]:
    """One accepted request's outcome.

    ``404`` once it has aged out of ``memorize.async_queue.row_retention_hours``
    (or never existed).
    """
    item = await get_task(task_id)
    if item is None:
        raise NotFoundError(f"task {task_id} not found or expired")
    return SuccessEnvelope(request_id=extract_request_id(request), data=item)
