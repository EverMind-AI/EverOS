"""Async session factory + session scope context manager."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from everos.core.errors import StorageError
from everos.core.locks import LOCKS
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

_SQLITE_BUSY = 5
_SQLITE_LOCKED = 6


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

    The checkout is registered with ``LOCKS`` as held (kind
    ``"sqlite_session"``), so the lock-order checks can see a task that waits
    for an asyncio lock while holding a session.

    A statement that waited out ``busy_timeout`` on another writer's lock
    (SQLite ``BUSY`` / ``LOCKED``) is re-raised as :class:`StorageError`,
    which the API answers with a retryable 503. Its message is fixed: the
    driver's message carries the SQL and its bound values, i.e. the request
    body; the SQL alone is logged so the losing write can still be named.
    Other ``OperationalError`` (a missing table, a disk error) is not
    transient and passes through.

    Usage:
        factory = create_session_factory(engine)
        async with session_scope(factory) as session:
            session.add(some_record)
            await session.commit()
    """
    with LOCKS.observe("sqlite_session", "checkout"):
        async with session_factory() as session:
            try:
                yield session
            except Exception as exc:
                await session.rollback()
                if _is_lock_wait(exc):
                    logger.warning(
                        "sqlite_lock_wait_timed_out",
                        statement=str(getattr(exc, "statement", ""))[:200],
                    )
                    raise StorageError("database is busy") from exc
                raise


def _is_lock_wait(exc: Exception) -> bool:
    """Whether ``exc`` is SQLite giving up on another connection's lock."""
    if not isinstance(exc, OperationalError):
        return False
    code = getattr(exc.orig, "sqlite_errorcode", None)
    return code is not None and code & 0xFF in (_SQLITE_BUSY, _SQLITE_LOCKED)
