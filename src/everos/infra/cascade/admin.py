"""Administration of what the cascade projects into: system DB, index, ledger.

The server's startup and the operator's ``everos cascade`` commands run the
same handful of steps against the same three things -- the SQLite system DB
schema, the derived index (open / verify / ensure / shutdown), and the cascade
ledger that records which markdown has been projected. :class:`CascadeAdmin`
is those steps as one object, so the application assembly hands it to the
code that sequences them instead of that code reaching into each storage
facade itself.
"""

from __future__ import annotations

__all__ = ["CascadeAdmin"]

import asyncio
from typing import TYPE_CHECKING, Any

from everos.core.observability.logging import get_logger
from everos.infra.persistence.index import (
    DerivedIndex,
    EmbeddingProvenance,
    is_null,
)
from everos.infra.persistence.sqlite import (
    MdChangeState,
    MdChangeStateRepo,
    QueueSummary,
    SqliteDatabase,
)

from .registry import match_kind

if TYPE_CHECKING:
    from everos.infra.persistence import StoreVersion

logger = get_logger(__name__)


class CascadeAdmin:
    """System DB schema, derived-index lifecycle and cascade-ledger operations.

    Args:
        database: The system DB.
        ledger: The ``md_change_state`` repository on ``database``.
        index: The derived index.
        provenance: The embedding check every index open runs; it also keeps
            the width its probe measured.
        store_version: The store-version gate; ``None`` opens any root (a
            test over its own storage).
    """

    def __init__(
        self,
        *,
        database: SqliteDatabase,
        ledger: MdChangeStateRepo,
        index: DerivedIndex,
        provenance: EmbeddingProvenance,
        store_version: StoreVersion | None = None,
    ) -> None:
        self._database = database
        self._ledger = ledger
        self._index = index
        self._provenance = provenance
        self._store_version = store_version

    # ── system DB ──────────────────────────────────────────────────────────

    async def create_schema(self) -> int:
        """Pass the store-version gate, then create every registered SQLite
        table that does not exist yet and add the columns an older
        ``system.db`` lacks.

        The gate runs here because this is the first step of every process
        that opens a root -- the server, each ``everos cascade`` command,
        backfill -- so a refused store is refused before anything in it
        changes.

        Returns:
            The number of registered tables.

        Raises:
            StoreVersionError: The root holds a store this EverOS does not open.
        """
        if self._store_version is not None:
            await asyncio.to_thread(self._store_version.check)
        return await self._database.create_schema()

    async def dispose(self) -> None:
        """Dispose the SQLite engine and its connection pool. Idempotent."""
        await self._database.dispose()

    # ── derived index ──────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """The configured index engine (``lancedb`` / ``milvus``)."""
        return self._index.name

    async def start_index(self) -> Any:
        """The server's open: connect, verify against the configuration, ensure."""
        return await self._index.startup(self._provenance)

    async def open_index(
        self, *, verify: bool = True, ensure: bool = True, may_attest: bool = True
    ) -> None:
        """The CLI's open, whose middle steps ``cascade rebuild`` must skip."""
        await self._index.connect()
        if verify:
            await self._index.verify_configuration(
                self._provenance, may_attest=may_attest
            )
        if ensure:
            await self._index.ensure_business_indexes()

    async def close_index(self) -> None:
        await self._index.shutdown()

    async def ensure_index(self) -> None:
        await self._index.ensure_business_indexes()

    async def drop_index_tables(self) -> list[str]:
        return await self._index.drop_business_tables()

    async def count_index_rows(self) -> dict[str, int]:
        return await self._index.count_business_rows()

    async def count_duplicate_keys(self) -> dict[str, int]:
        return await self._index.count_duplicate_keys()

    async def count_unbackfilled_rows(self) -> int:
        """Rows whose primary vector is still NULL, summed over every table.

        A table that cannot answer is logged and skipped; the count is a
        startup hint, never a reason to fail.
        """
        total = 0
        for repo in self._index.repositories.all():
            if "vector" not in repo.schema.vector_columns():
                continue
            try:
                total += await repo.count_where(is_null("vector"))
            except Exception as exc:
                logger.warning(
                    "unbackfilled_check_failed",
                    schema=repo.schema.__name__,
                    error=repr(exc),
                )
        return total

    async def attest_embedding_model(self, model: str) -> None:
        await self._index.write_embedding_model(model)

    def embedding_probed(self) -> bool:
        """Whether the embedding service answered the startup dimension probe."""
        return self._provenance.last_probe is not None

    # ── cascade ledger ─────────────────────────────────────────────────────

    async def queue_summary(self) -> QueueSummary:
        return await self._ledger.queue_summary()

    async def count_live_sources(self) -> int:
        return await self._ledger.count_done_live_sources()

    async def list_failed(self) -> list[MdChangeState]:
        return await self._ledger.list_failed()

    async def list_unreadable(self) -> list[MdChangeState]:
        """Indexed files whose last projection skipped unreadable entries."""
        return await self._ledger.list_unreadable()

    async def reset_retryable(self) -> int:
        return await self._ledger.reset_retryable_to_pending()

    async def reset_ledger(self) -> int:
        return await self._ledger.reset_all()

    async def force_enqueue(self, rel_md_path: str) -> str | None:
        """Queue one markdown file; ``None`` when no cascade kind matches it.

        Returns:
            The matched kind's name.
        """
        spec = match_kind(rel_md_path)
        if spec is None:
            return None
        await self._ledger.force_enqueue(rel_md_path, spec.name)
        return spec.name
