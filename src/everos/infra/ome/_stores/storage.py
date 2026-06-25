"""OME SQLite storage — schema initialization + connection factory.

Single file (default ``MemoryRoot.default().ome_db`` ≡
``<memory-root>/.index/sqlite/ome.db``). Holds 3 OME-managed tables
(counter_store / idle_store / run_record); APS jobstore table is created
by APScheduler itself when its SQLAlchemyJobStore connects.

PRAGMA scopes (see https://www.sqlite.org/pragma.html):
  - ``journal_mode=WAL`` is file-level — persisted in the db header,
    applied once in :meth:`OMEStorage.init`.
  - ``synchronous=NORMAL``, ``cache_size=-65536``, ``busy_timeout=5000``
    are connection-level and reset on every new connection, so they are
    re-applied inside :meth:`OMEStorage.connect` (which is why
    ``connect`` is an ``@asynccontextmanager`` rather than a passthrough).
    This mirrors SQLAlchemy's canonical ``@event.listens_for(Engine,
    "connect")`` pattern for SQLite — aiosqlite exposes no equivalent
    hook. ``busy_timeout=5000`` matters because the APS jobstore writes
    its own table in the same db file; without it, WAL writer-vs-writer
    contention surfaces as ``SQLITE_BUSY`` instead of brief backoff.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

_SCHEMA = """
CREATE TABLE IF NOT EXISTS counter_store (
    strategy_name   TEXT NOT NULL,
    bucket_key      TEXT NOT NULL,
    counter         INTEGER NOT NULL DEFAULT 0,
    last_passed_ts  TIMESTAMP,
    PRIMARY KEY (strategy_name, bucket_key)
);

CREATE TABLE IF NOT EXISTS idle_store (
    strategy_name      TEXT NOT NULL,
    bucket_key         TEXT NOT NULL,
    last_activity_ts   TIMESTAMP NOT NULL,
    PRIMARY KEY (strategy_name, bucket_key)
);
CREATE INDEX IF NOT EXISTS idx_idle_scan
    ON idle_store (strategy_name, last_activity_ts);

CREATE TABLE IF NOT EXISTS run_record (
    run_id                          TEXT PRIMARY KEY,
    strategy_name                   TEXT NOT NULL,
    status                          TEXT NOT NULL,
    attempt                         INTEGER NOT NULL DEFAULT 0,
    started_at                      TIMESTAMP NOT NULL,
    finished_at                     TIMESTAMP,
    error                           TEXT,
    event_topic                     TEXT NOT NULL,
    event_payload                   TEXT NOT NULL,
    max_retries_snapshot            INTEGER NOT NULL,
    event_id                        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_strategy_started
    ON run_record (strategy_name, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_run_status_started
    ON run_record (status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_run_event_id
    ON run_record (event_id);
"""

_INIT_PRAGMAS = ("PRAGMA journal_mode=WAL",)
_CONN_PRAGMAS = (
    "PRAGMA synchronous=NORMAL",
    "PRAGMA cache_size=-65536",
    "PRAGMA busy_timeout=5000",
)


class OMEStorage:
    """Connection factory + schema init for the OME SQLite db."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    async def init(self) -> None:
        """Create parent dirs + apply file-level pragmas + create schema.

        Runs forward-only migrations for existing databases (e.g. adding
        the ``event_id`` column introduced in P3).
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as conn:
            for pragma in _INIT_PRAGMAS:
                await conn.execute(pragma)
            await self._migrate(conn)
            await conn.executescript(_SCHEMA)
            await conn.commit()

    @staticmethod
    async def _migrate(conn: aiosqlite.Connection) -> None:
        """Forward-only column migrations for existing databases."""
        cur = await conn.execute("PRAGMA table_info(run_record)")
        columns = {row[1] for row in await cur.fetchall()}
        if not columns:
            return
        if "event_id" not in columns:
            await conn.execute(
                "ALTER TABLE run_record ADD COLUMN event_id TEXT NOT NULL DEFAULT ''"
            )

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield an aiosqlite connection with per-connection pragmas applied."""
        async with aiosqlite.connect(self.db_path) as conn:
            for pragma in _CONN_PRAGMAS:
                await conn.execute(pragma)
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield a connection inside an ``IMMEDIATE`` transaction.

        Commits on success, rolls back on any exception. Mirrors
        SQLAlchemy's ``conn.begin()`` for raw aiosqlite, which exposes
        no built-in transaction context manager. ``BEGIN IMMEDIATE``
        (rather than ``DEFERRED``) acquires the write lock upfront so
        a read-modify-write block cannot lose to a competing writer
        between its SELECT and its UPDATE.
        """
        async with self.connect() as conn:
            try:
                await conn.execute("BEGIN IMMEDIATE")
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
