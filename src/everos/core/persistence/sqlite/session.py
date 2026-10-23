"""Async session factory + session scope context manager."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from everos.core.errors import DatabaseBusyError

_LOCK_MARKERS = ("database is locked", "database table is locked")
"""SQLite's two lock messages, checked case-insensitively.

``database is locked`` is what a writer gets when another connection holds
the write lock past ``PRAGMA busy_timeout``. ``database table is locked`` is
the same condition raised by ``DROP``/``ALTER`` paths. Both are transient:
the correct response is to retry shortly, not to report an internal fault.
"""


def _is_locked_error(exc: OperationalError) -> bool:
    """Return whether ``exc`` is SQLite lock contention (not another DB error)."""
    # Prefer SQLite's own primary result code. It is locale- and wording-stable,
    # and it also catches extended codes (e.g. SQLITE_BUSY_SNAPSHOT): the low
    # byte is the primary result, the high bits are the extended cause.
    orig = getattr(exc, "orig", None)
    code = getattr(orig, "sqlite_errorcode", None)
    if isinstance(code, int) and (code & 0xFF) in (
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
    ):
        return True

    # Fallback for drivers/wrappers that do not expose ``sqlite_errorcode``
    # (or a wrapped error whose ``orig`` is not a sqlite3 exception).
    # ``str(exc)`` carries SQLAlchemy's wrapper text; ``orig`` carries the
    # sqlite3 message. Check both, case-insensitively.
    message = str(exc).lower()
    if orig is not None:
        message = f"{message} {orig}".lower()
    return any(marker in message for marker in _LOCK_MARKERS)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build an :class:`async_sessionmaker` bound to ``engine``.

    ``expire_on_commit=False`` keeps attribute access on instances valid
    after commit, which is the conventional setup for async SA usage.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@asynccontextmanager
async def session_scope(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Yield an :class:`AsyncSession` inside a try/rollback/close block.

    The session is rolled back on any exception in the ``async with`` body,
    then closed. Callers are responsible for calling ``await session.commit()``
    on success.

    Usage:
        factory = create_session_factory(engine)
        async with session_scope(factory) as session:
            session.add(some_record)
            await session.commit()

    Lock contention is surfaced as :class:`DatabaseBusyError` (a retryable
    ``StorageError``) rather than an unhandled ``OperationalError``. Without
    this translation a caller that merely lost a race for SQLite's write lock
    receives HTTP 500, which tells it retrying is pointless when the opposite
    is true. See issue #498.
    """
    async with session_factory() as session:
        try:
            yield session
        except OperationalError as exc:
            await session.rollback()
            if _is_locked_error(exc):
                raise DatabaseBusyError("SQLite is busy; retry the request") from exc
            raise
        except Exception:
            await session.rollback()
            raise
