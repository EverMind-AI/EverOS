"""The memory deletion use case behind ``POST /memory/delete``.

Five scopes, one shape: delete the source first, then the memories made
from it, then their index rows.

* only ``session_id`` -- the conversation everywhere: refused (503, nothing
  deleted) while the session has queued work or an ``add()`` holds it past
  ``delete_wait_seconds``; otherwise, under the session's memorize lock, its
  memcells, buffers and status go in one transaction, then every owner the
  memcells name loses that session's entries.
* ``user_id`` / ``agent_id`` with ``session_id`` -- that owner's entries of
  the session; the conversation stays.
* ``user_id`` / ``agent_id`` alone -- the owner's whole directory, its
  clusters and (for a user) its Reflection reports.

Once anything is deleted, a failure is reported as incomplete (500): the
same request finishes the job. Index rows are removed last and never fail
the request (the markdown is gone; the cascade catches up).
"""

from __future__ import annotations

__all__ = ["DeleteResult", "Deleter"]

from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from everos.core.errors import (
    DeletionIncompleteError,
    PathTraversalError,
    SessionBusyError,
)
from everos.core.failpoints import failpoint
from everos.core.locks import LOCKS
from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import get_partition_lock
from everos.memory.ports import MemoryRef, Scope, SessionKey

if TYPE_CHECKING:
    from everos.config import MemorizeSettings
    from everos.memory.deps import Stores
    from everos.memory.ports import IndexEraser, OwnerType

logger = get_logger("everos.service.memory")

_QUEUED = "queued work for this session; nothing was deleted, retry later"
_LOCKED = "session is busy; nothing was deleted, retry later"

type Removed = Mapping[str, Sequence[MemoryRef]]
"""The entries one owner lost, by index table."""


@dataclass(frozen=True)
class DeleteResult:
    """What a deletion did.

    Attributes:
        filters: The scope fields the request set (``user_id`` /
            ``agent_id`` / ``session_id``).
        count: Memories removed from markdown.
    """

    filters: list[str]
    count: int


class Deleter:
    """Deletes memories by owner, by session, or both.

    Args:
        stores: Write-side stores.
        eraser: Removes the deleted entries' index rows.
        settings: ``[memorize]`` (``delete_wait_seconds``).
    """

    def __init__(
        self, *, stores: Stores, eraser: IndexEraser, settings: MemorizeSettings
    ) -> None:
        self._stores = stores
        self._eraser = eraser
        self._wait = settings.delete_wait_seconds

    async def delete(
        self,
        *,
        app_id: str,
        project_id: str,
        user_id: str | None,
        agent_id: str | None,
        session_id: str | None,
    ) -> DeleteResult:
        """Delete one scope; at least one of the three ids is set."""
        if user_id is None and agent_id is None:
            if session_id is None:
                raise ValueError("a deletion needs user_id, agent_id or session_id")
            session = SessionKey(app_id, project_id, session_id)
            count = await self._erase(await self._session(session))
            filters = ["session_id"]
        else:
            owner_type: OwnerType = "user" if user_id is not None else "agent"
            scope = Scope(
                app_id=app_id,
                project_id=project_id,
                owner_id=user_id or agent_id or "",
                owner_type=owner_type,
            )
            filters = [f"{owner_type}_id"]
            if session_id is None:
                count = await self._owner(scope)
            else:
                removed = await _incomplete(self._remove_session(scope, session_id))
                count = await self._erase([(scope, removed)])
                filters.append("session_id")
        logger.info(
            "memory_deleted",
            app_id=app_id,
            project_id=project_id,
            filters=filters,
            count=count,
        )
        return DeleteResult(filters=filters, count=count)

    async def _erase(self, deleted: Sequence[tuple[Scope, Removed]]) -> int:
        """Remove the deleted entries' index rows; how many entries."""
        failpoint("delete.before_index_erase")
        for scope, removed in deleted:
            await self._eraser.erase(scope, removed)
        return sum(len(refs) for _, removed in deleted for refs in removed.values())

    async def _session(self, session: SessionKey) -> list[tuple[Scope, Removed]]:
        """Delete one conversation everywhere, under its memorize lock."""
        inbox = self._stores.memorize_inbox
        if await inbox.count_pending(session):
            raise SessionBusyError(_QUEUED)
        key = f"{session.app_id}\x1f{session.project_id}\x1f{session.session_id}"
        try:
            async with LOCKS.hold("session", key, max_wait=self._wait):
                # A request queued while this one waited for the lock.
                if await inbox.count_pending(session):
                    raise SessionBusyError(_QUEUED)
                return await _incomplete(self._purge_session(session))
        except TimeoutError:  # only the wait for the lock raises it here
            raise SessionBusyError(_LOCKED) from None

    async def _purge_session(self, session: SessionKey) -> list[tuple[Scope, Removed]]:
        records = await self._stores.session_purge.purge(session)
        failpoint("delete.after_sqlite")
        senders = dict.fromkeys(s for record in records for s in record.sender_ids)
        # A sender's memories live under users/ or agents/ (the assistant side
        # of an agent trajectory); trying both costs a scan of a missing dir.
        owners = [
            Scope(
                app_id=session.app_id,
                project_id=session.project_id,
                owner_id=sender,
                owner_type=owner_type,
            )
            for sender in senders
            for owner_type in ("user", "agent")
        ] or await self._stores.owner_dir.owners_with_session(
            session.app_id, session.project_id, session.session_id
        )
        deleted: list[tuple[Scope, Removed]] = []
        for scope in owners:
            deleted.append(
                (scope, await self._remove_session(scope, session.session_id))
            )
            failpoint("delete.after_first_rewrite")
        return deleted

    async def _remove_session(self, scope: Scope, session_id: str) -> Removed:
        stores = self._stores
        if scope.owner_type == "agent":
            return {
                "agent_case": await stores.agent_case.remove_session(scope, session_id)
            }
        return {
            "episode": await stores.episode.remove_session(scope, session_id),
            "atomic_fact": await stores.atomic_fact.remove_session(scope, session_id),
            "foresight": await stores.foresight.remove_session(scope, session_id),
        }

    async def _owner(self, scope: Scope) -> int:
        """Delete the owner's directory, clusters and reports; the count."""
        stores = self._stores
        try:
            deletion = await stores.owner_dir.delete(scope)
        except PathTraversalError:
            raise  # refused before anything moved
        except Exception as exc:
            raise DeletionIncompleteError(str(exc)) from exc
        if scope.owner_type == "user":
            partition = f"{scope.app_id}:{scope.project_id}:{scope.owner_id}"
            async with get_partition_lock("trigger_profile_clustering", partition):
                await _incomplete(stores.cluster.delete_for_owner(scope, "user_memory"))
                await _incomplete(stores.reflection_report.delete_for_owner(scope))
            tables = ["user_profile"]
        else:
            await _incomplete(stores.cluster.delete_for_owner(scope, "agent_case"))
            tables = ["agent_skill"]
        failpoint("delete.before_index_erase")
        await self._eraser.erase(scope, deletion.refs)
        await self._eraser.erase_owner(scope, tables)
        return deletion.count


async def _incomplete[T](step: Awaitable[T]) -> T:
    """Await a step taken after the deletion began; a failure is incomplete."""
    try:
        return await step
    except (SessionBusyError, DeletionIncompleteError):
        raise
    except Exception as exc:
        raise DeletionIncompleteError(str(exc)) from exc
