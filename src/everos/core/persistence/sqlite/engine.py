"""Async SQLAlchemy engine factory + per-connection PRAGMA listener.

The engine connects through ``aiosqlite`` (SA URL ``sqlite+aiosqlite://``).
PRAGMAs are *per-connection* — they must be re-applied every time the
SA pool opens a new connection. We attach a ``connect`` event listener on
the engine's underlying sync engine for that purpose.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from everos.config import SqliteSettings


def create_system_engine(
    db_path: Path,
    sqlite_settings: SqliteSettings,
    *,
    echo: bool = False,
) -> AsyncEngine:
    """Create an async SQLAlchemy engine for the everos system DB.

    ``MemoryRoot.system_db`` is the conventional path; the DB holds system
    state, audit log, task queue, LSN watermark, and other metadata.

    Args:
        db_path: Filesystem path to the system DB file. Parent directory is
            created if missing.
        sqlite_settings: Tunables (journal_mode, synchronous, foreign_keys,
            temp_store, busy_timeout, journal_size_limit, cache_size).
        echo: When ``True``, SQLAlchemy logs every statement (development).

    Returns:
        An :class:`AsyncEngine` ready for use with :class:`AsyncSession`.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Three slashes = relative path; four slashes = absolute. ``str(db_path)``
    # of an absolute Path begins with ``/`` so the f-string yields four.
    url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(url, echo=echo, future=True)

    _register_pragma_listener(engine, sqlite_settings)
    return engine


def _register_pragma_listener(
    engine: AsyncEngine,
    sqlite_settings: SqliteSettings,
) -> None:
    """Attach a ``connect`` listener that applies PRAGMAs on every new connection."""

    @event.listens_for(engine.sync_engine, "connect")
    def _apply_pragmas(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f"PRAGMA journal_mode={sqlite_settings.journal_mode}")
            cursor.execute(f"PRAGMA synchronous={sqlite_settings.synchronous}")
            cursor.execute(
                f"PRAGMA foreign_keys={'ON' if sqlite_settings.foreign_keys else 'OFF'}"
            )
            cursor.execute(f"PRAGMA temp_store={sqlite_settings.temp_store}")
            cursor.execute(f"PRAGMA busy_timeout={sqlite_settings.busy_timeout_ms}")
            cursor.execute(
                f"PRAGMA journal_size_limit={sqlite_settings.journal_size_limit_bytes}"
            )
            # cache_size: negative = KB, positive = pages.
            cursor.execute(f"PRAGMA cache_size=-{sqlite_settings.cache_size_kb}")
        finally:
            cursor.close()
