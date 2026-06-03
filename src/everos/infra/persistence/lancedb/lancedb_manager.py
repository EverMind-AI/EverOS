"""LanceDB connection + table singletons (lazy + process-wide, async).

The single place that owns the LanceDB **runtime state**: the async
connection and per-name table cache. Connection opens lazily on first
:func:`get_connection` call; tables are cached after first
:func:`get_table`. The :class:`LanceDBLifespanProvider` calls
:func:`dispose_connection` on shutdown; in scripts you can call it
manually.
"""

from __future__ import annotations

import asyncio

from lancedb import AsyncConnection, AsyncTable

from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.core.persistence import BaseLanceTable, MemoryRoot, open_lancedb_connection

logger = get_logger(__name__)

_conn: AsyncConnection | None = None
_tables: dict[str, AsyncTable] = {}
_lock = asyncio.Lock()


async def get_connection() -> AsyncConnection:
    """Return the process-wide async LanceDB connection.

    Built on first call from ``MemoryRoot.default().lancedb_dir`` and
    ``Settings.lancedb``. Subsequent calls return the same instance.
    """
    async with _lock:
        return await _ensure_connection_locked()


async def get_table(
    name: str,
    schema: type[BaseLanceTable],
) -> AsyncTable:
    """Open the named table (creating from ``schema`` if missing). Cached."""
    async with _lock:
        if name not in _tables:
            conn = await _ensure_connection_locked()
            existing = await conn.list_tables()
            if name in list(existing.tables):
                _tables[name] = await conn.open_table(name)
                logger.info("lancedb_table_opened", name=name)
            else:
                _tables[name] = await conn.create_table(name, schema=schema)
                logger.info("lancedb_table_created", name=name)
        return _tables[name]


async def dispose_connection() -> None:
    """Close the connection + clear table cache. Idempotent."""
    global _conn
    async with _lock:
        if _conn is not None:
            try:
                _conn.close()  # AsyncConnection.close() is sync in lancedb 0.30
            except Exception:
                logger.exception("lancedb_close_failed")
            logger.info("lancedb_connection_closed")
        _conn = None
        _tables.clear()


async def _ensure_connection_locked() -> AsyncConnection:
    """Open the connection if not yet open. Caller must hold ``_lock``."""
    global _conn
    if _conn is None:
        settings = load_settings()
        memory_root = MemoryRoot.default()
        memory_root.ensure()
        _conn = await open_lancedb_connection(memory_root.lancedb_dir, settings.lancedb)
        logger.info(
            "lancedb_connection_opened",
            path=str(memory_root.lancedb_dir),
        )
    return _conn
