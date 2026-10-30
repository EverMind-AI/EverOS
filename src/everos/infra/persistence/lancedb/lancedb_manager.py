"""The LanceDB connection of one memory root, with its per-name table cache.

:class:`LanceDbConnection` is built by the application bootstrap and handed
to the index backend and its repositories; nothing in the process reaches
for a global connection. Building it opens nothing -- the connection opens,
and the index directory is created, on the first table access.
"""

from __future__ import annotations

__all__ = ["LanceDbConnection"]

import asyncio
from collections.abc import Sequence

import pyarrow as pa
from lancedb import AsyncConnection, AsyncTable

from everos.config import LanceDBSettings
from everos.core.observability.logging import get_logger
from everos.infra.persistence.lancedb._base.connection import open_lancedb_connection
from everos.infra.persistence.memory_root import MemoryRoot

logger = get_logger(__name__)


class LanceDbConnection:
    """The async connection to ``<root>/.index/lancedb`` and its open tables.

    Args:
        root: Memory root whose LanceDB directory this connects to.
        settings: The ``[lancedb]`` tunables.
    """

    def __init__(self, root: MemoryRoot, settings: LanceDBSettings) -> None:
        self._root = root
        self._settings = settings
        self._conn: AsyncConnection | None = None
        self._tables: dict[str, AsyncTable] = {}
        self._lock = asyncio.Lock()

    @property
    def root(self) -> MemoryRoot:
        """The memory root this connection belongs to."""
        return self._root

    async def connect(self) -> AsyncConnection:
        """Open the connection if it is not open yet; return it."""
        async with self._lock:
            return await self._ensure_connection_locked()

    async def table(self, name: str, arrow_schema: pa.Schema) -> AsyncTable:
        """Open the named table (creating it with ``arrow_schema`` if missing).

        Cached by name: ``arrow_schema`` is only read when the table does not
        exist yet.
        """
        async with self._lock:
            if name not in self._tables:
                conn = await self._ensure_connection_locked()
                existing = await conn.list_tables()
                if name in list(existing.tables):
                    self._tables[name] = await conn.open_table(name)
                    logger.info("lancedb_table_opened", name=name)
                else:
                    self._tables[name] = await conn.create_table(
                        name, schema=arrow_schema
                    )
                    logger.info("lancedb_table_created", name=name)
            return self._tables[name]

    async def drop_tables(self, names: Sequence[str]) -> list[str]:
        """Drop the named tables if present; return the names actually dropped.

        Each dropped table is also evicted from the cache so a later
        :meth:`table` recreates it fresh from the current schema. Used by
        ``cascade rebuild`` to reset a corrupt / drifted index — the tables
        are a rebuildable projection of markdown, so dropping is safe.
        """
        async with self._lock:
            conn = await self._ensure_connection_locked()
            existing = set((await conn.list_tables()).tables)
            dropped: list[str] = []
            for name in names:
                if name in existing:
                    await conn.drop_table(name)
                    self._tables.pop(name, None)
                    dropped.append(name)
                    logger.info("lancedb_table_dropped", name=name)
            return dropped

    async def close(self) -> None:
        """Close the connection and clear the table cache. Idempotent."""
        async with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()  # AsyncConnection.close() is sync in lancedb
                except Exception:
                    logger.exception("lancedb_close_failed")
                logger.info("lancedb_connection_closed")
            self._conn = None
            self._tables.clear()

    async def _ensure_connection_locked(self) -> AsyncConnection:
        """Open the connection if not yet open. Caller must hold ``_lock``."""
        if self._conn is None:
            self._root.ensure()
            self._conn = await open_lancedb_connection(
                self._root.lancedb_dir, self._settings
            )
            logger.info("lancedb_connection_opened", path=str(self._root.lancedb_dir))
        return self._conn
