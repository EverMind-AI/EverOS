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

``POST /memory/delete_by_ids`` deletes single episodes by memory id, with
what was merged into them, their facts, and their cluster membership (see
:meth:`Deleter.delete_by_ids`).

Once anything is deleted, a failure is reported as incomplete (500): the
same request finishes the job. Index rows are removed last and never fail
the request (the markdown is gone; the cascade catches up).
"""

from __future__ import annotations

__all__ = ["DeleteResult", "Deleter"]

import re
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from everos.core.errors import (
    DeletionIncompleteError,
    PathTraversalError,
    SessionBusyError,
)
from everos.core.failpoints import failpoint
from everos.core.locks import LOCKS, session_key
from everos.core.observability.logging import get_logger
from everos.memory import user_cluster_lock
from everos.memory.models import Episode
from everos.memory.ports import MemoryRef, Scope, SessionKey

if TYPE_CHECKING:
    from everos.config import MemorizeSettings
    from everos.memory.deps import Stores
    from everos.memory.ports import IndexEraser, OwnerType

logger = get_logger("everos.service.memory")

_QUEUED = "queued work for this session; nothing was deleted, retry later"
_LOCKED = "session is busy; nothing was deleted, retry later"
_MEMORY_ID = re.compile(r"(?P<owner>.+)_(?P<entry>ep_\d{8}_\d+)")
_DETACH_ATTEMPTS = 3
_PREVIEW_CAP = 5
"""A cluster's preview length, as everalgo clustering keeps it (``preview_cap``
default, everalgo-clustering 0.2.1 ``clustering/algorithm.py:41``)."""

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
        key = session_key(session.app_id, session.project_id, session.session_id)
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
            # A picture memory names its session like the kinds above. Left out,
            # a deleted session kept its pictures searchable with no memcell
            # behind them, and re-sending the session added a second copy.
            "image_memory": await stores.image_memory.remove_session(scope, session_id),
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
            async with user_cluster_lock(
                scope.app_id, scope.project_id, scope.owner_id
            ):
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

    # ── by id ────────────────────────────────────────────────────────────

    async def delete_by_ids(
        self,
        *,
        app_id: str,
        project_id: str,
        memory_type: str,
        ids: Sequence[str],
        reason_code: str | None,
        note: str | None,
    ) -> None:
        """Delete episodes by memory id (``<owner_id>_<entry_id>``).

        Per owner, in this order (each step is safe to redo, so the same
        request finishes a deletion a crash cut short):

        1. the requested entries, under the owner lock -- the barrier: a
           Reflection finish marks sources only while the merged episode is
           in md, so none can be marked after this;
        2. what was merged into them, transitively (read from the merge
           marks, so it still works on a resend after step 1);
        3. the atomic facts of all of those;
        4. those episodes out of their clusters, under the clustering lock;
        5. the merge marks keyed by them -- last, a resend after 1 needs them;
        6. their index rows.

        ``note`` is never logged.
        """
        if memory_type != "episode":
            raise ValueError(f"unsupported memory_type {memory_type!r}")
        by_owner: dict[str, list[str]] = {}
        for memory_id in dict.fromkeys(ids):
            owner_id, entry_id = _split_memory_id(memory_id)
            by_owner.setdefault(owner_id, []).append(entry_id)
        hit: list[str] = []
        for owner_id, entry_ids in by_owner.items():
            scope = Scope(
                app_id=app_id,
                project_id=project_id,
                owner_id=owner_id,
                owner_type="user",
            )
            removed = await _incomplete(self._delete_episodes(scope, entry_ids))
            hit += [f"{owner_id}_{e}" for e in removed]
        logger.info(
            "memory_delete_by_ids",
            app_id=app_id,
            project_id=project_id,
            reason=reason_code,
            hit=hit,
            skipped=[m for m in dict.fromkeys(ids) if m not in set(hit)],
        )

    async def _delete_episodes(self, scope: Scope, targets: Sequence[str]) -> list[str]:
        """Steps 1-6 for one owner; the requested entries that were in md."""
        episodes, facts = self._stores.episode, self._stores.atomic_fact
        requested = set(targets)
        first = await episodes.remove(scope, requested)
        failpoint("by_ids.after_step1")
        closure = set(requested)
        frontier = set(requested)
        while frontier:
            frontier = await episodes.merged_into(scope, frontier) - closure
            closure |= frontier
        merged = await episodes.remove(scope, closure - requested)
        fact_refs = await facts.remove_by_parents(scope, closure)
        failpoint("by_ids.after_step3")
        await self._detach_from_clusters(scope, closure)
        await episodes.clear_deprecations(scope, closure)
        failpoint("by_ids.after_step5")
        await self._eraser.erase(
            scope, {"episode": [*first, *merged], "atomic_fact": fact_refs}
        )
        return [ref.entry_id for ref in first]

    async def _detach_from_clusters(self, scope: Scope, gone: set[str]) -> None:
        """Take ``gone`` out of their clusters (and members no longer in md).

        Members and their texts are read without the clustering lock; under
        it the member list is read again and, if it moved, the round starts
        over (at most three times, then the md is read under the lock).
        """
        clusters = self._stores.cluster
        cluster_ids: set[str] = set()
        for entry_id in sorted(gone):
            found = await clusters.find_cluster_for_member(scope, "episode", entry_id)
            if found is not None:
                cluster_ids.add(found)
        for cluster_id in sorted(cluster_ids):
            for attempt in range(_DETACH_ATTEMPTS):
                ids = [m for m, _ in await clusters.members(cluster_id)]
                alive = await self._alive(scope, ids)
                async with user_cluster_lock(
                    scope.app_id, scope.project_id, scope.owner_id
                ):
                    now = [m for m, _ in await clusters.members(cluster_id)]
                    if now != ids:
                        if attempt < _DETACH_ATTEMPTS - 1:
                            continue
                        ids, alive = now, await self._alive(scope, now)
                    await self._restamp(cluster_id, ids, alive, gone)
                break

    async def _alive(self, scope: Scope, ids: Sequence[str]) -> dict[str, Episode]:
        stored = await self._stores.episode.read_many(scope, ids)
        return {s.ref.entry_id: s.item for s in stored}

    async def _restamp(
        self,
        cluster_id: str,
        ids: Sequence[str],
        alive: Mapping[str, Episode],
        gone: set[str],
    ) -> None:
        left = [alive[m] for m in ids if m not in gone and m in alive]
        clusters = self._stores.cluster
        if not left:
            await clusters.delete(cluster_id)
            return
        kept = {m for m in ids if m not in gone and m in alive}
        newest = sorted(left, key=lambda e: e.timestamp)[-_PREVIEW_CAP:]
        await clusters.detach(
            cluster_id,
            [m for m in ids if m not in kept],
            count=len(left),
            last_ts=max(e.timestamp for e in left),
            preview=[e.episode for e in newest],
        )


def _split_memory_id(memory_id: str) -> tuple[str, str]:
    """``<owner_id>_<entry_id>`` -> ``(owner_id, entry_id)`` (validated upstream)."""
    match = _MEMORY_ID.fullmatch(memory_id)
    if match is None:
        raise ValueError(f"not an episode memory id: {memory_id!r}")
    return match["owner"], match["entry"]


async def _incomplete[T](step: Awaitable[T]) -> T:
    """Await a step taken after the deletion began; a failure is incomplete."""
    try:
        return await step
    except (SessionBusyError, DeletionIncompleteError):
        raise
    except Exception as exc:
        raise DeletionIncompleteError(str(exc)) from exc
