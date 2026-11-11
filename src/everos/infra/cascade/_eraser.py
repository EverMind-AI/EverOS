"""Delete a deletion's index rows at once, fenced against cascade write-backs."""

from __future__ import annotations

__all__ = ["FencedIndexEraser"]

import asyncio
from collections.abc import Collection, Mapping, Sequence

from everos.core.observability.logging import get_logger
from everos.infra.persistence.index import IndexRepositories
from everos.infra.persistence.sqlite import MdChangeStateRepo
from everos.memory.ports import Filter, MemoryRef, Scope, all_of, eq, one_of

from ._fence import DeleteFence
from .registry import match_kind

logger = get_logger(__name__)


class FencedIndexEraser:
    """:class:`~everos.memory.ports.IndexEraser` over the derived index.

    Each table's delete is bounded by ``timeout_s``. A table that times out
    or fails is logged (``memory_delete_index_erase_failed``) and every md
    file whose rows it held is queued again, so the cascade's next run of
    that file deletes the rows by diff. Queuing the file again is needed even
    though rewriting it already queued it once: a run of the same file that
    started before the rewrite may still be going, and may have read the
    index before an even earlier run wrote the rows back. A file is queued
    under the cascade kind its path matches, which picks its handler: a table
    is not a kind (``group_profile_item`` is projected by ``group_profile``),
    so the table name stands in only for a path no kind matches.

    Args:
        index: The derived-index tables.
        fence: Shared with the cascade projections.
        ledger: The cascade's ``md_change_state`` queue.
        timeout_s: Upper bound of one table's delete.
    """

    def __init__(
        self,
        index: IndexRepositories,
        fence: DeleteFence,
        ledger: MdChangeStateRepo,
        *,
        timeout_s: float = 15.0,
    ) -> None:
        self._index = index
        self._fence = fence
        self._ledger = ledger
        self._timeout_s = timeout_s

    async def erase(
        self, scope: Scope, removed: Mapping[str, Sequence[MemoryRef]]
    ) -> None:
        self._fence.add(
            (scope.app_id, scope.project_id, ref.id)
            for refs in removed.values()
            for ref in refs
        )
        for table, refs in removed.items():
            if not refs:
                continue
            where = all_of(
                eq("app_id", scope.app_id),
                eq("project_id", scope.project_id),
                one_of("id", [ref.id for ref in refs]),
            )
            if not await self._delete(table, where, rows=len(refs)):
                await self._requeue(table, {ref.md_path for ref in refs})

    async def requeue(self, kind: str, md_paths: Collection[str]) -> None:
        """Queue ``md_paths`` for one more projection, logging a failure.

        A failure is not handed back: the next full scan of the memory root
        re-reads the file anyway, because its mtime moved when it was written.
        """
        await self._requeue(kind, md_paths)

    async def erase_owner(self, scope: Scope, kinds: Collection[str]) -> None:
        # ponytail: no requeue on failure; the owner's files are gone, so the
        # cascade's deletion pass over them removes the rows when it runs.
        where = all_of(
            eq("owner_id", scope.owner_id),
            eq("app_id", scope.app_id),
            eq("project_id", scope.project_id),
        )
        for table in kinds:
            await self._delete(table, where, rows=None)

    async def _delete(self, table: str, where: Filter, *, rows: int | None) -> bool:
        try:
            async with asyncio.timeout(self._timeout_s):
                await getattr(self._index, table).delete(where)
        except Exception as exc:
            logger.warning(
                "memory_delete_index_erase_failed",
                table=table,
                rows=rows,
                error=repr(exc)[:200],
            )
            return False
        return True

    async def _requeue(self, kind: str, md_paths: Collection[str]) -> None:
        for md_path in sorted(md_paths):
            spec = match_kind(md_path)
            try:
                await self._ledger.force_enqueue(
                    md_path, spec.name if spec is not None else kind
                )
            except Exception as exc:  # the scanner still finds the rewrite
                logger.warning(
                    "memory_delete_requeue_failed",
                    md_path=md_path,
                    kind=kind,
                    error=repr(exc)[:200],
                )
