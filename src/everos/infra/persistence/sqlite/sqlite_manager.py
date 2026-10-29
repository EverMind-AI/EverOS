"""The system database: one engine and session factory per application.

:class:`SqliteDatabase` is built by the application bootstrap and handed to
every repository; nothing in the process reaches for a global engine.
Building it opens nothing and writes nothing -- the engine connects, and the
database directory is created, on first use.
"""

from __future__ import annotations

__all__ = ["SqliteDatabase"]

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from everos.config import SqliteSettings
from everos.core.observability.logging import get_logger
from everos.infra.persistence.memory_root import MemoryRoot
from everos.infra.persistence.sqlite.base.engine import create_system_engine
from everos.infra.persistence.sqlite.base.session import create_session_factory

logger = get_logger(__name__)


class SqliteDatabase:
    """The ``system.db`` under one memory root.

    Args:
        root: Memory root whose ``system_db`` this is.
        settings: The ``[sqlite]`` tunables.
    """

    def __init__(self, root: MemoryRoot, settings: SqliteSettings) -> None:
        self._root = root
        self._engine = create_system_engine(root.system_db, settings)
        self._session_factory = create_session_factory(self._engine)

    @property
    def engine(self) -> AsyncEngine:
        """The async engine; it connects lazily, on the first statement."""
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Opens sessions bound to :attr:`engine`."""
        return self._session_factory

    async def dispose(self) -> None:
        """Close every pooled connection. Idempotent; the engine stays usable."""
        await self._engine.dispose()
        logger.info("sqlite_engine_disposed", db_path=str(self._root.system_db))
