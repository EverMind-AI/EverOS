"""SQLite engine + session-factory singletons (lazy + process-wide).

The single place that owns the SQLite **runtime state**: the async
SQLAlchemy engine and the session factory bound to it. Built lazily on
first :func:`get_engine` / :func:`get_session_factory` call from
:func:`everos.config.load_settings` + :meth:`MemoryRoot.default`. The
:class:`SqliteLifespanProvider` calls :func:`dispose_engine` on shutdown
to drain the connection pool; in scripts you can call it manually.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)

logger = get_logger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the process-wide async SQLAlchemy engine.

    Built on first call from ``MemoryRoot.default()`` and ``Settings.sqlite``.
    Subsequent calls return the same instance.
    """
    global _engine
    if _engine is None:
        settings = load_settings()
        memory_root = MemoryRoot.default()
        memory_root.ensure()
        _engine = create_system_engine(memory_root.system_db, settings.sqlite)
        logger.info(
            "sqlite_engine_built",
            db_path=str(memory_root.system_db),
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide async session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = create_session_factory(get_engine())
    return _session_factory


async def dispose_engine() -> None:
    """Dispose the engine + connection pool. Idempotent."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        logger.info("sqlite_engine_disposed")
    _engine = None
    _session_factory = None
