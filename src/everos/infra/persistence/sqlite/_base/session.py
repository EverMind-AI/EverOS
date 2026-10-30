"""Async session factory + session scope context manager."""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from everos.core.errors import DatabaseBusyError


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


def _is_locked_error(exc: OperationalError) -> bool:
    """Whether SQLAlchemy's ``OperationalError`` is SQLite lock contention."""
    orig = getattr(exc, "orig", None)
    code = getattr(orig, "sqlite_errorcode", None)
    if isinstance(code, int) and (code & 0xFF) in (
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
    ):
        return True
    message = str(exc).lower()
    return "database is locked" in message or "database table is locked" in message
