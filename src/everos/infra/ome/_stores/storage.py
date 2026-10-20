"""OME SQLite storage — schema initialization + connection factory.

Single file (default ``MemoryRoot.resolve().ome_db`` ≡
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
    hook. ``busy_timeout=5000`` is the cross-process fallback; writers of
    this process queue on :meth:`OMEStorage.write` first (see there).
"""

from __future__ import annotations

import asyncio
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
        self._write_gate: asyncio.Lock | None = None
        self._keeper: aiosqlite.Connection | None = None

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
        # Bound to the loop that starts the engine; a restart gets a new one.
        self._write_gate = asyncio.Lock()

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

    async def hold_open(self) -> None:
        """Keep one connection open for the engine's lifetime.

        In WAL mode the last connection to close checkpoints the WAL and
        deletes the -wal / -shm files under an exclusive lock. Every
        store call here opens and closes its own connection, so that path
        ran dozens of times a second, and on Windows it held the lock for
        seconds at a time: a fresh connection's first statement waited out
        the 5 s busy timeout and failed with database is locked (the
        OME run state-machine test on Windows CI). While this connection
        is open no other close is the last one. It holds no transaction,
        so checkpoints still run; it has read the header once so that the
        WAL is attached to it.
        """
        if self._keeper is None:
            keeper = await aiosqlite.connect(self.db_path)
            cur = await keeper.execute("PRAGMA user_version")
            await cur.fetchone()
            self._keeper = keeper

    async def release(self) -> None:
        """Close the connection :meth:`hold_open` keeps; the last close
        checkpoints and removes the WAL files as usual."""
        if self._keeper is not None:
            keeper, self._keeper = self._keeper, None
            await keeper.close()

    def _gate(self) -> asyncio.Lock:
        if self._write_gate is None:
            self._write_gate = asyncio.Lock()
        return self._write_gate

    @asynccontextmanager
    async def write(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield a connection for one write; commits on success.

        Writers queue on an in-process lock, served in the order they
        reach it with nobody overtaking, and only then touch the file.
        SQLite itself admits one writer at a
        time, but its busy handler waits by sleeping and retrying, so under
        several concurrent writers a late arrival can keep taking the lock
        ahead of an early one; on Windows, where a commit costs 30-80 ms,
        a writer starved that way hit the 5 s busy timeout and failed with
        ``database is locked`` (measured: 16 writers, p99 3.8 s). The queue
        bounds the wait to queue length times commit time. It covers this
        process only, which is every writer of this file: the engine holds
        the jobstore lock. The connection is opened before queueing, so the
        open (100-500 ms on Windows) overlaps and only the statement and
        its commit are serialized. Never await anything slow while holding
        the lock.
        """
        async with self.connect() as conn, self._gate():
            yield conn
            await conn.commit()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Yield a connection inside an ``IMMEDIATE`` transaction.

        Commits on success, rolls back on any exception. Mirrors
        SQLAlchemy's ``conn.begin()`` for raw aiosqlite, which exposes
        no built-in transaction context manager. ``BEGIN IMMEDIATE``
        (rather than ``DEFERRED``) acquires the write lock upfront so
        a read-modify-write block cannot lose to a competing writer
        between its SELECT and its UPDATE. Queues on the same in-process
        writer lock as :meth:`write`.
        """
        async with self.connect() as conn, self._gate():
            try:
                await conn.execute("BEGIN IMMEDIATE")
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
