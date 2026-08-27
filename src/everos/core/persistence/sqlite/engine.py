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
from everos.core.observability.logging import get_logger

logger = get_logger(__name__)


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
    # Pool parameters are passed explicitly rather than inherited. They were
    # inherited before, and the failure that exposed it is not one the defaults
    # can survive: a connection checked out and never returned leaves the pool
    # one slot smaller forever, and once every slot is gone each later caller
    # waits on a checkout that no longer completes. Two benchmark servers reached
    # that state -- aiosqlite connection threads at 20 and 58 against a steady
    # 6-10 on their healthy siblings, 7 and 22 of them parked inside aiosqlite's
    # connect path -- and every SQLite file simply stopped being written, for
    # 2h17m and 3h33m, until they were killed by hand.
    #
    # What made it expensive was that nothing looked broken. The process answered
    # HTTP, the event loop was live, every thread was idle, and CPU was flat. The
    # OME queue stopped draining because strategies could not persist their own
    # results, so `run_record` rows froze mid-flight in RUNNING -- which reads as
    # "still working", not "cannot write". Even the run-timeout backstop was mute:
    # it fired, then needed a connection to record the failure.
    #
    # `pool_pre_ping` and `pool_recycle` reclaim such a connection at its next
    # checkout, and `pool_timeout` bounds the wait so exhaustion surfaces as a
    # retryable error instead of a silent stall. None of this fixes whatever
    # leaks the connection; it stops one leak from taking the process with it.
    engine = create_async_engine(
        url,
        echo=echo,
        future=True,
        pool_size=sqlite_settings.pool_size,
        max_overflow=sqlite_settings.max_overflow,
        pool_timeout=sqlite_settings.pool_timeout_seconds,
        pool_recycle=sqlite_settings.pool_recycle_seconds,
        pool_pre_ping=sqlite_settings.pool_pre_ping,
    )

    _register_pragma_listener(engine, sqlite_settings)
    _register_pool_saturation_listener(engine, sqlite_settings)
    return engine


def _register_pool_saturation_listener(
    engine: AsyncEngine,
    sqlite_settings: SqliteSettings,
) -> None:
    """Log once per checkout that finds the pool at or near capacity.

    The point is a signal that exists at all. When the pool drained on two
    benchmark servers there was nothing to see: no error, no log line, no metric
    -- writes simply stopped, and diagnosis came down to counting aiosqlite
    threads in a py-spy dump against a healthy sibling process. A warning at the
    moment of saturation names the condition while the process is still running,
    and its ``checked_out`` count is what distinguishes real concurrency from a
    leak: honest load returns connections, so the number oscillates; a leak only
    climbs.
    """
    capacity = sqlite_settings.pool_size + sqlite_settings.max_overflow

    @event.listens_for(engine.sync_engine, "checkout")
    def _warn_when_saturated(_dbapi_conn, _rec, _proxy) -> None:  # type: ignore[no-untyped-def]
        pool = engine.sync_engine.pool
        checked_out = getattr(pool, "checkedout", lambda: -1)()
        if checked_out >= capacity:
            logger.warning(
                "sqlite.pool.saturated",
                checked_out=checked_out,
                capacity=capacity,
                pool_size=sqlite_settings.pool_size,
                max_overflow=sqlite_settings.max_overflow,
                pool_timeout_seconds=sqlite_settings.pool_timeout_seconds,
            )


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
