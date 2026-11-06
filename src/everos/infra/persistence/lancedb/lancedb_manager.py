"""The LanceDB connection of one memory root, with its per-name table cache.

:class:`LanceDbConnection` is built by the application bootstrap and handed
to the index backend and its repositories; nothing in the process reaches
for a global connection. Building it opens nothing -- the connection opens,
and the index directory is created, on the first table access.
"""

from __future__ import annotations

__all__ = ["LanceDbConnection"]

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import pyarrow as pa
from lancedb import AsyncConnection, AsyncTable

from everos.config import LanceDBSettings
from everos.core.observability.logging import get_logger
from everos.core.observability.metrics import Gauge, Histogram
from everos.infra.persistence.lancedb._base.connection import open_lancedb_connection
from everos.infra.persistence.memory_root import MemoryRoot

logger = get_logger(__name__)

_query_wait_seconds = Histogram(
    name="lancedb_query_wait_seconds",
    description="Time a search query waited for a LanceDB query slot.",
    namespace="everos",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
_queries_in_flight = Gauge(
    name="lancedb_queries_in_flight",
    description="LanceDB search queries holding a slot now.",
    namespace="everos",
)


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
        self._slots: asyncio.Semaphore | None = None
        self._slots_loop: asyncio.AbstractEventLoop | None = None

    @property
    def root(self) -> MemoryRoot:
        """The memory root this connection belongs to."""
        return self._root

    async def connect(self) -> AsyncConnection:
        """Open the connection if it is not open yet; return it."""
        async with self._lock:
            return await self._ensure_connection_locked()

    @asynccontextmanager
    async def query_slot(self) -> AsyncIterator[None]:
        """Hold one of ``max_concurrent_queries`` search-query slots."""
        slots = self._slots_of_running_loop()
        started = time.monotonic()
        async with slots:
            _query_wait_seconds.observe(time.monotonic() - started)
            _queries_in_flight.inc()
            try:
                yield
            finally:
                _queries_in_flight.dec()

    def _slots_of_running_loop(self) -> asyncio.Semaphore:
        """The slots, made anew when the running loop is not theirs: a
        semaphore that ever had a waiter is bound to that loop (see
        ``everos.core.locks``)."""
        loop = asyncio.get_running_loop()
        if self._slots is None or loop is not self._slots_loop:
            self._slots = asyncio.Semaphore(self._settings.max_concurrent_queries)
            self._slots_loop = loop
        return self._slots

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
