"""The ``async_mode`` task readers: inbox rows as Cloud-shaped tasks."""

from __future__ import annotations

__all__ = ["get_task", "list_tasks"]

from typing import TYPE_CHECKING

from everos.core.errors import InvalidInputError
from everos.memory.ports import SessionKey
from everos.service.memory import TASK_STATUSES, TaskItem, TaskPage, to_task_item

if TYPE_CHECKING:
    from everos.memory.ports import MemorizeInboxStore

_MAX_TASK_PAGE_SIZE = 100


async def get_task(inbox: MemorizeInboxStore, request_id: str) -> TaskItem | None:
    """One accepted request's outcome; ``None`` once it aged out (or never was)."""
    row = await inbox.get_task(request_id)
    return None if row is None else to_task_item(row)


async def list_tasks(
    inbox: MemorizeInboxStore,
    *,
    session_id: str | None,
    status: str | None,
    page: int,
    page_size: int,
    app_id: str,
    project_id: str,
) -> TaskPage:
    """Accepted requests, newest first; ``page_size`` is capped at 100.

    Raises:
        InvalidInputError: ``status`` is not one of :data:`TASK_STATUSES`.
    """
    if status is not None and status not in TASK_STATUSES:
        raise InvalidInputError(
            f"status must be one of {', '.join(TASK_STATUSES)}; got {status!r}"
        )
    page_size = min(page_size, _MAX_TASK_PAGE_SIZE)
    session: SessionKey | None = None
    if session_id is not None:
        session = SessionKey(
            app_id=app_id, project_id=project_id, session_id=session_id
        )
    tasks = await inbox.list_tasks(
        session=session,
        status=status,
        offset=(page - 1) * page_size,
        limit=page_size,
    )
    return TaskPage(
        total=tasks.total,
        page=page,
        page_size=page_size,
        items=[to_task_item(t) for t in tasks.items],
    )
