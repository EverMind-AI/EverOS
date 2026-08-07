"""Generic CRUD repository for LanceDB-backed tables.

``LanceRepoBase`` mirrors the SQLite ``RepoBase`` shape: a pure generic
CRUD helper that knows nothing about a storage runtime. Concrete repos
either pass an :class:`AsyncTable` explicitly (typical in tests) or
override :meth:`_table_lookup` to pull the cached table from their
storage manager (typical in
:mod:`everos.infra.persistence.lancedb.repos`).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, ClassVar

from lancedb import AsyncTable

from everos.component.utils.datetime import get_utc_now
from everos.core.errors import VectorStoreBusyError
from everos.core.observability.logging import get_logger

from .base import BaseLanceTable

logger = get_logger(__name__)

# Write-lock deadlines, per operation class. Every critical section on a
# table's write lock runs under one of these (see ``LanceRepoBase._locked``):
# acquisition included, so neither waiting for the lock nor holding it can be
# unbounded. They are hang-catchers, not throughput limits — but they are sized
# from measured durations, not guessed, because the budget is also how long a
# wedged table stays invisible.
_WRITE_TIMEOUT_SECONDS = 15.0
"""Row writes: add / upsert / update / delete.

Measured on a local SSD across table sizes and batch sizes (10k–100k rows,
50–500 rows per call): median 2–25ms, worst observed 63ms — flat in both
dimensions, because these are append-and-commit operations, not scans.
``merge_insert`` (upsert) is the read-modify-write one and still lands at
5–25ms.

15s is ~240x the worst observation, which covers a slow/contended disk and
several operations queued ahead on the same lock (the deadline includes
acquisition, and ``asyncio.Lock`` is FIFO so no waiter starves). Reaching it
means the table is not merely busy — it is stuck, and failing fast into the
worker's retry is better than blocking writers for minutes. Deliberately not
sized in the hundreds of seconds: the budget doubles as the detection latency
for a wedged table."""

_REBUILD_TIMEOUT_SECONDS = 300.0
"""Index rebuild (drop + recreate every index) — the one genuinely slow
critical section, measured at ~0.3s per 50k rows per indexed column, so 5
minutes covers a multi-million-row table with wide headroom."""

# Safety cap on a single prune's ``optimize(cleanup_older_than=…)`` call — a
# pure hang-catcher, not a bound on normal runtime. A real cleanup is
# milliseconds even on a heavily churned table (measured ~40ms at 320k writes /
# 100 versions), so 60s is ~1500× headroom and never fires in normal operation.
# Cleanup only deletes files already unreferenced by the current manifest, so a
# timeout that cancels it mid-scan just reclaims less this beat — it cannot
# corrupt the table — but it releases the per-table write lock instead of
# wedging every writer behind a hung lance cleanup. Kept well below the prune
# cadence (worker ``DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS``) so a hung beat
# leaves a real write window before the next attempt (review P2 / N1).
_PRUNE_TIMEOUT_SECONDS = 60.0

_COMPACT_TIMEOUT_SECONDS = 60.0
"""Deadline on the lock-free compaction beat. It takes no lock, so it cannot
block writers — but it still must not hang: the scheduler runs one maintenance
task per kind and skips a kind whose task is in flight, so a compaction that
never returns parks that kind's maintenance permanently. Measured at ~460ms on
a table with 77 retained versions."""

_HUSK_MIN_AGE_SECONDS = 7 * 24 * 60 * 60.0
"""Minimum age of an empty ``_indices/<uuid>/`` dir before it is removed.

Deliberately **lance's own number**, not one of ours. ``cleanup.rs`` defines
``UNVERIFIED_THRESHOLD_DAYS = 7`` and uses it for exactly this judgement: an
index UUID that no manifest references is only assumed dead once it is at least
7 days old, because before that it is indistinguishable from an index build
still in progress. Matching the threshold means this sweep can never be more
aggressive than lance itself — the earlier 300s value was our invention, and
that is precisely what made it unjustifiable.

**The cost of the gate is accepted, with numbers — and the effective horizon
is up to 14 days, not 7.** The gate reads ``st_mtime``, and POSIX bumps a
directory's mtime whenever an entry is unlinked from it — so the clock
restarts the moment lance's cleanup empties the husk, not when the dir was
created. Under a short retention window the index *files* themselves first
wait out lance's 7-day unverified window (see :meth:`LanceRepoBase.prune`),
so a husk is reclaimed up to file-wait + 7 days after it appeared. Each one
costs an inode plus one 4KB block. Measured on a soak at ceiling load (see
``DEFAULT_OPTIMIZE_MIN_INTERVAL_SECONDS``): ~127k dirs/day, so the ~14-day
steady state is ~1.8M dirs = **~7GB of empty directories and ~28% of a
default 98GB ext4's inodes**. That is the worst case, not the expected one —
it needs
sustained saturation, and a single-user deployment writing a few hundred times
a day sits four orders of magnitude below it (tens of MB). Platform-wise only
ext4 has a fixed inode budget at all; APFS and xfs allocate dynamically, and
Windows is out of scope. The knob to reach for if this ever does bite is the
optimize cooldown, not this threshold — see there.

**When to remove this sweep** — lance-format/lance#8322 (merged 2026-08-06,
not in any release as of lancedb 0.34.0 / lance 8.0.0 — the merge sits ahead of
v11.0.0-beta.2) makes lance's own cleanup remove the directories it empties,
under this same 7-day / ``delete_unverified`` policy. Once a release carrying
it is pinned, delete :func:`_remove_empty_index_dirs` and its call site rather
than keep a second implementation of upstream's rule. That upgrade also halves
the horizon above: lance drops the directory in the pass that empties it, so
our age gate stops stacking on top of theirs and the steady state goes from
~14 days back to ~7.
"""

_HUSK_SWEEP_TIMEOUT_SECONDS = 60.0
"""Deadline on the (lock-free) husk sweep. Same last-resort role as
:data:`_COMPACT_TIMEOUT_SECONDS`: the maintenance scheduler skips a kind whose
task is in flight, so a sweep that never returns would park that kind's
maintenance forever. Measured at ~460ms over 13061 dirs in the soak (~35us
per dir), which puts the ceiling-load steady state (~1.8M dirs, see
:data:`_HUSK_MIN_AGE_SECONDS`) right at this budget. That is tolerated rather
than sized around: a timeout is swallowed inside :meth:`LanceRepoBase.prune`
(the cleanup commit already succeeded, and billing the sweep to the prune
ledger is exactly the signal corruption this module works to avoid), and the
orphaned worker thread — a cancelled ``to_thread`` future does not stop the
thread — finishes the walk anyway, so the reclamation still happens."""

_READ_TIMEOUT_SECONDS = 60.0
"""Deadline on every read. Reads take no lock, so a hung read blocks no writer
— but it does park the caller, and the cascade drain loop reads on every batch
while advancing strictly one batch at a time. A read that never returns
therefore stops the whole md -> LanceDB projection, leaving claimed rows in
``processing`` forever with nothing logged (a hang raises nothing, so the
drain-failure counter stays at zero and ``/health`` keeps reporting healthy).
Same last-resort shape as :data:`_COMPACT_TIMEOUT_SECONDS`, and generous by
design: everos builds no vector ANN index, so reads are flat scans — measured
~62ms over 117k rows, i.e. 60s is ~1000x headroom and never fires normally. On
expiry the caller gets a retryable :class:`VectorStoreBusyError`, so a drain row
is retried and a search request fails with a structured error rather than
hanging the request."""

_SLOW_HOLD_LOG_SECONDS = 1.0
"""Log a completed critical section that held the write lock at least this
long. Normal writes are 2-25ms and a normal prune ~40ms, so anything past a
second means writers were queued behind it — without this, a section that is
slow but under its deadline is invisible (the maintenance beat only logs at
``debug``), and a soak run left no way to tell whether a 16s stall was a slow
prune or a deep write queue."""


def _q(value: str) -> str:
    """Escape single quotes for a LanceDB SQL-like ``where`` predicate.

    LanceDB has no parameterised query API; predicates are strings.
    Doubling the quote (``'`` → ``''``) is the SQL-standard way to keep
    a literal single quote inside a single-quoted string. everos's PK
    convention (``<owner_id>_<entry_id>``) never carries quotes — this
    is defensive.
    """
    return value.replace("'", "''")


def _remove_empty_index_dirs(
    table_uri: str, *, live_uuids: frozenset[str], min_age_seconds: float
) -> int:
    """Remove empty ``_indices/<uuid>/`` husks under a table dir; return count.

    lance's cleanup unlinks the *files* of a superseded index but never the
    directory — ``cleanup.rs`` contains no ``rmdir``/``remove_dir`` at all, and
    that is structural rather than an oversight: lance is written against an
    object store where paths are flat keys and an "empty directory" does not
    exist. Only a local filesystem materialises them, where they accumulate as
    inodes and slow every directory scan (a soak run reached 13061 dirs, 98%
    empty).

    Three independent guarantees, in order of strength:

    1. **``rmdir`` cannot delete data.** The kernel refuses it on a non-empty
       directory (``ENOTEMPTY``). No file can be lost through this function
       whatever the rest of the logic decides — and because the check *is* the
       operation, there is no check-then-act window to race.
    2. **Live indexes are excluded** by UUID, read from ``list_indices()``.
    3. **Anything else waits out lance's own conservatism bound**
       (:data:`_HUSK_MIN_AGE_SECONDS`): a directory a concurrent
       ``create_index`` just made is seconds old, so it can never qualify.

    Best-effort throughout: a directory that becomes non-empty, vanishes, or is
    unreadable between listing and ``rmdir`` is skipped, not an error.
    """
    indices = Path(table_uri) / "_indices"
    if not indices.is_dir():
        return 0
    cutoff = get_utc_now().timestamp() - min_age_seconds
    removed = 0
    for child in indices.iterdir():
        if not child.is_dir() or child.name in live_uuids:
            continue
        try:
            if child.stat().st_mtime > cutoff:
                continue
            child.rmdir()
        except OSError:
            continue
        removed += 1
    return removed


class LanceRepoBase[T: BaseLanceTable]:
    """Generic CRUD repository for one LanceDB table.

    Subclass and bind to a schema. Two ways to provide the table:

    1. **Explicit (tests / DI)** — pass it to ``__init__``::

           repo = EpisodeRepo(table)

    2. **Lazy hook (production singletons)** — override
       :meth:`_table_lookup` so the repo can be instantiated as a
       module-level singleton with no live connection yet::

           class _EpisodeRepo(LanceRepoBase[Episode]):
               schema = Episode

               async def _table_lookup(self):
                   from everos.infra.persistence.lancedb.lancedb_manager import (
                       get_table,
                   )
                   return await get_table(self.schema.TABLE_NAME, self.schema)

           episode_repo = _EpisodeRepo()
           await episode_repo.add([Episode(text=..., vector=[...])])

    The LanceDB table name lives on the schema (``BaseLanceTable.TABLE_NAME``)
    so every LanceDB-side metadatum — column shape, table name,
    vector dim, BM25 index spec — sits in one place. ``table_name``
    here is a thin pass-through; subclasses normally do **not**
    override it.

    Write paths (``add`` / ``upsert`` / ``delete`` / ``delete_by_md_path``)
    are serialised by a per-``table_name`` :class:`asyncio.Lock`. LanceDB's
    ``merge_insert`` is a read-modify-write at the storage layer with no
    application-visible OCC contract — two concurrent calls against the
    same table can race on the version manifest and lose updates even
    when the row sets are disjoint (observed: cascade worker
    ``asyncio.gather`` over a batch of ``user_profile`` rows where one
    write disappears). Serialising on the table name closes that window;
    reads stay unlocked so search QPS is not impacted by writers.

    Locks live in a class-level dict keyed by table name and are never
    evicted (mirrors :mod:`everos.memory._partition_locks`
    on bpo-28427 — a lock with pending waiters must outlive any dict
    entry that points to it).
    """

    schema: type[T]

    _table_locks: ClassVar[dict[str, asyncio.Lock]] = {}
    """Per-table-name write lock pool (process-wide, lazily populated)."""

    @property
    def table_name(self) -> str:
        """LanceDB table name, resolved from :attr:`schema.TABLE_NAME`."""
        return self.schema.TABLE_NAME

    @classmethod
    def _write_lock(cls, table_name: str) -> asyncio.Lock:
        """Return the write lock for ``table_name``; create on first use.

        ``dict.setdefault`` is atomic under single-threaded asyncio (no
        ``await`` between check and insert), so no meta-lock is needed.
        """
        return cls._table_locks.setdefault(table_name, asyncio.Lock())

    @asynccontextmanager
    async def _deadline(self, budget: float, op: str) -> AsyncIterator[None]:
        """Bound an operation that does **not** take the write lock.

        Same last-resort guarantee as :meth:`_locked` minus the lock: the
        maintenance scheduler runs one task per kind and skips a kind whose
        task has not finished, so any await in that path which can hang must
        have a deadline or that kind stops being maintained for good.
        """
        try:
            async with asyncio.timeout(budget):
                yield
        except TimeoutError as exc:
            logger.warning(
                "lancedb_operation_deadline_exceeded",
                table=self.table_name,
                op=op,
                budget_seconds=budget,
            )
            raise VectorStoreBusyError(
                f"{op} on table {self.table_name!r} exceeded its {budget:g}s deadline"
            ) from exc

    @asynccontextmanager
    async def _locked(self, budget: float, op: str) -> AsyncIterator[None]:
        """Hold the table write lock for at most ``budget`` seconds.

        **The deadline covers acquisition *and* the body.** That is the whole
        point: every critical section on this lock is bounded, so no code path
        can wait for it — or hold it — indefinitely. A single stuck operation
        would otherwise wedge the table forever, because a stuck holder blocks
        every writer *and* the maintenance scheduler skips a kind whose task
        never finishes (observed in a soak run: one table stopped reclaiming
        versions permanently, 150 versions retained, disk 11x live size, with
        no error logged anywhere because nothing failed — it simply never
        returned).

        On expiry the body is cancelled, the lock is released, and the timeout
        is re-raised as :class:`VectorStoreBusyError` so the cascade worker
        treats it as transient and retries instead of marking the row
        permanently failed.

        Callers resolve the table handle **inside** this block, not before it.
        Resolving it outside leaves an unbounded await ahead of the deadline,
        and a maintenance task that hangs there never returns — which silently
        parks that kind forever, because the scheduler skips a kind whose task
        is still in flight (observed in a soak run: one table stopped pruning
        for 13 minutes with zero failure logs while its siblings pruned fine).
        """
        started = time.monotonic()
        acquired_at: float | None = None
        try:
            async with asyncio.timeout(budget):
                async with self._write_lock(self.table_name):
                    acquired_at = time.monotonic()
                    yield
        except TimeoutError as exc:
            # ``acquired`` is the load-bearing field: it separates "never got
            # the lock" (a holder is slow or stuck) from "got it and overran"
            # (this operation itself is slow), which is exactly what a soak
            # investigation cannot otherwise tell apart.
            now = time.monotonic()
            logger.warning(
                "lancedb_write_lock_deadline_exceeded",
                table=self.table_name,
                op=op,
                budget_seconds=budget,
                acquired=acquired_at is not None,
                waited_seconds=round((acquired_at or now) - started, 3),
                held_seconds=round(now - acquired_at, 3) if acquired_at else 0.0,
            )
            raise VectorStoreBusyError(
                f"{op} on table {self.table_name!r} exceeded its "
                f"{budget:g}s write-lock deadline"
            ) from exc
        else:
            held = time.monotonic() - (acquired_at or started)
            if held >= _SLOW_HOLD_LOG_SECONDS:
                logger.info(
                    "lancedb_write_lock_slow_hold",
                    table=self.table_name,
                    op=op,
                    held_seconds=round(held, 3),
                    waited_seconds=round((acquired_at or started) - started, 3),
                )

    @classmethod
    def _reset_locks_for_tests(cls) -> None:
        """Test-only: drop the write-lock pool.

        ``asyncio.Lock`` binds to the current event loop on first
        ``acquire()``; pytest-asyncio creates a fresh loop per test, so
        a module-level lock surviving across tests fails with "bound to
        a different event loop". The production cascade worker runs on
        one loop forever and does not need this hook. Mirrors
        :func:`everos.memory._partition_locks._reset_for_tests`.
        """
        cls._table_locks.clear()

    def __init__(self, table: AsyncTable | None = None) -> None:
        """Bind to a table directly; if ``None``, defer to ``_table_lookup``."""
        self._table_override = table

    async def _table_lookup(self) -> AsyncTable:
        """Resolve the table on first use. Override in subclass.

        ``LanceRepoBase`` itself has no idea where the runtime singleton
        lives. The default raises so a missing override is loud rather
        than silently broken.
        """
        raise NotImplementedError(
            f"{type(self).__name__}: pass table= to __init__ "
            "or override _table_lookup() to wire the storage manager."
        )

    async def _table(self) -> AsyncTable:
        if self._table_override is not None:
            return self._table_override
        return await self._table_lookup()

    # ── Create ─────────────────────────────────────────────────────────────

    async def add(self, records: Sequence[T]) -> None:
        """Insert one or more records."""
        async with self._locked(_WRITE_TIMEOUT_SECONDS, "add"):
            table = await self._table()
            await table.add(list(records))

    # ── Upsert ─────────────────────────────────────────────────────────────

    async def upsert(
        self,
        records: Sequence[T],
        *,
        by: str = "id",
    ) -> None:
        """Upsert records keyed by ``by`` (PK column, default ``"id"``).

        Wraps LanceDB's ``merge_insert(on=...)`` fluent builder with the
        equivalent of ``INSERT ... ON CONFLICT(by) DO UPDATE`` — matching
        rows are replaced wholesale, non-matching rows inserted.

        Cascade uses this when reconciling md → LanceDB: an entry seen
        for the first time inserts; an entry that was edited in md
        updates its existing row.
        """
        async with self._locked(_WRITE_TIMEOUT_SECONDS, "upsert"):
            table = await self._table()
            await (
                table.merge_insert(by)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(list(records))
            )

    # ── Maintenance ────────────────────────────────────────────────────────

    async def optimize(self) -> None:
        """Compact fragments + merge new data into the FTS / vector indexes.

        ``optimize()`` is a **performance + storage-hygiene** operation,
        **not** a correctness/visibility one. LanceDB's ``merge_insert``
        writes new data into a fresh fragment that the FTS / vector
        indexes don't cover yet; queries stay correct regardless because
        LanceDB transparently brute-force flat-scans that unindexed tail
        and unions it with the indexed hits. Verified on lancedb 0.30.2:
        after ``create_index`` + ``merge_insert`` (no ``optimize()``), a
        ``nearest_to_text`` for a token present only in the new rows
        returns those rows immediately.

        (Older lancedb — at/below the ``>=0.13.0`` floor this repo once
        pinned — did **not** flat-scan the FTS tail, so post-build rows
        were genuinely invisible to BM25 until ``optimize()``; that is
        the behaviour the historical LoCoMo-conv0 note described. The
        flat-scan fallback closed that gap, so optimize is now purely
        about keeping that tail small.)

        What ``optimize()`` actually buys on the current stack:

        - **Query speed** — the unindexed tail is flat-scanned on every
          query; merging it into the index keeps that scan bounded as
          ingest accumulates.
        - **Storage hygiene** is *not* done here — physical reclamation
          of replaced fragments / stale manifests / dead index files is
          :meth:`prune`, a separate write-locked call.

        Cascade triggers this through a per-kind throttle + trailing
        edge scheduler (``CascadeWorker._schedule_optimize``): at most
        one run per throttle window per kind, decoupled from the drain
        loop, with a 60s heartbeat sweep as a safety net. Cost is
        O(N) data-rewrite per optimized fragment; the throttle is how
        we cap it under sustained write pressure. Because visibility no
        longer depends on it, the throttle window can be generous.

        This is **compaction only** — physical reclamation of superseded
        files is :meth:`prune`, a separate write-locked call. Kept lock-free
        on purpose: a ``Retryable commit conflict`` against a concurrent
        writer is benign here (compaction is not urgent — the next scheduled
        beat retries), so it must not stall writers.
        """
        async with self._deadline(_COMPACT_TIMEOUT_SECONDS, "optimize"):
            table = await self._table()
            await table.optimize()

    async def prune(self, older_than: dt.timedelta) -> None:
        """Physically reclaim files from versions older than ``older_than``.

        LanceDB's ``AsyncTable`` cannot clean up independently of compaction —
        the only handle is ``optimize(cleanup_older_than=..., delete_unverified=...)``,
        which bundles compact + cleanup into one manifest commit. Under
        sustained churn that commit is a Rewrite that concurrent Delete /
        Update writes preempt, so the bundled cleanup loses the race and
        never runs (observed in the storage soak: 16 successes / 547 conflicts
        over 21h → the index dir grew unbounded to the disk guardrail).

        Fix: run it **under the per-table write lock** so no write is in
        flight for its duration. That does two things at once:

        1. **No commit conflict** — the Rewrite has the manifest to itself,
           so cleanup actually completes every beat.
        2. **Cross-process safe** — ``delete_unverified=False`` keeps lance
           from deleting any file it cannot tie to a removed version, i.e. a
           file a writer in *another process* (a CLI ``cascade sync`` /
           ``backfill``) may be mid-commit on. The per-table write lock is
           in-process only, so it cannot fence a second process; the flag is
           what makes concurrent processes safe. Measured to reclaim
           identically to ``delete_unverified=True`` on churned tables (both
           collapse superseded versions ~97%), because ordinary churn
           orphans are all version-referenced and therefore verifiable —
           ``True`` only additionally deletes in-flight / dangling files,
           which is exactly the corruption vector. Reclaiming *during* active
           load comes from running under the write lock so the cleanup commit
           never loses the manifest race, not from the flag.

        ``cleanup_older_than`` deletes the *files* under a
        superseded ``_indices/<uuid>/`` but never the directory (lance's
        ``cleanup.rs`` contains no directory removal at all — it is written
        against an object store where an empty directory does not exist), so
        those husks accumulate on a local filesystem: a soak run reached 13061
        dirs, 98% empty. They are swept here, outside the lock — see
        :func:`_remove_empty_index_dirs` for why that is safe.

        This is a *separate* gap from index files not being reclaimed while
        they are young: lance skips an index UUID that no manifest references
        until it is 7 days old, and a short retention window deletes those
        manifests first, so the files lose their last reference and wait out
        the full 7 days (measured: 260MB retained on a 19k-row soak table,
        reclaimed in full once backdated past the threshold).

        The trade-off is a brief write stall (~seconds on a churned table,
        dominated by the cleanup's file scan/delete — flat, not proportional
        to the backlog). Cascade runs it on a slow beat
        (``CascadeWorker._optimize_prune_interval``, default 300s), so the
        stall is rare. Does *not* shrink **active** index internals (FTS
        ``part_N`` / index UUID count) — that is ``rebuild_indexes``'s job.
        """
        async with self._locked(_PRUNE_TIMEOUT_SECONDS, "prune"):
            table = await self._table()
            await table.optimize(cleanup_older_than=older_than, delete_unverified=False)
            table_uri = await table.uri()
            live_uuids = frozenset(
                i.index_uuid for i in await table.list_indices() if i.index_uuid
            )
        # Lock-free: the sweep can only ``rmdir``, which the kernel refuses on a
        # non-empty directory, so it cannot lose data no matter who else is
        # writing. Keeping it out of the critical section also means a slow
        # filesystem walk cannot overrun the prune budget.
        #
        # Best-effort means best-effort: the cleanup commit above already
        # succeeded, so a sweep timeout must not escape ``prune()``. Letting it
        # escape bills the failure to the wrong account — the optimize
        # scheduler counts a prune "failure" (feeding the fallback-rebuild
        # threshold) and the prune-staleness clock stops advancing, both
        # reporting a cleanup stall that did not happen. Same defect shape as
        # the alert counter the fallback rebuild used to zero: an auxiliary
        # path corrupting the main signal's ledger. The skipped husks are
        # retried on the next heavy beat.
        try:
            async with self._deadline(_HUSK_SWEEP_TIMEOUT_SECONDS, "prune_husk_sweep"):
                removed = await asyncio.to_thread(
                    _remove_empty_index_dirs,
                    table_uri,
                    live_uuids=live_uuids,
                    min_age_seconds=_HUSK_MIN_AGE_SECONDS,
                )
        except VectorStoreBusyError:
            # _deadline already logged lancedb_operation_deadline_exceeded.
            removed = 0
        if removed:
            logger.debug(
                "lancedb_pruned_empty_index_dirs",
                table=self.table_name,
                removed=removed,
            )

    async def rebuild_indexes(self) -> None:
        """Drop and re-create every index on this table.

        **Why this exists** — workaround for an upstream Python API gap:

        Lance's Rust ``OptimizeOptions`` has a ``num_indices_to_merge``
        knob (default 1) that bounds the number of active index UUIDs
        per column. With ``Some(1)``, every ``optimize_indices()`` call
        merges its delta into the base — active UUID count stays at 1.

        Two problems block us from using it from the application layer:

        1. ``lancedb.AsyncTable.optimize()`` does **not expose** this
           parameter (verified on lancedb main 2026-05-28). It forwards
           only ``cleanup_since_ms`` and ``delete_unverified`` to Rust.
        2. Even calling Lance directly via ``pylance``, the merge
           behaviour itself is buggy on ``lance crate 4.0`` (what
           lancedb 0.30.2 embeds) — ``num_indices_to_merge=1`` does
           nothing. Fix landed in ``lance 7.x``, but ``pylance 7.x``
           can not collapse indexes on a ``lance 4.0``-format dataset
           (verified by experiment).

        So in our current stack there is **no application-level path**
        to bound active index UUID growth. ``optimize()`` keeps
        accumulating one new UUID (vector) / one new ``part_N`` (FTS)
        per call.

        This method is the workaround: rebuild every indexed column from the
        schema's ``ensure_fts_indexes`` contract. Measured effect — the live
        index goes from 7 files back to 4 after 25 ``optimize()`` beats, i.e.
        the fragment set does collapse. The rebuild is an **O(N) full retrain**
        but cheap in practice (~0.3s for 50k rows × 2 FTS columns on local SSD).

        It rebuilds **in place** (``create_index(replace=True)``) rather than
        dropping first. An earlier version dropped every index and recreated
        them, on the assumption that LanceDB falls back to a brute-force scan
        meanwhile. That is true for vector search and **false for FTS**: with no
        inverted index a BM25 query raises ``Cannot perform full text search
        unless an INVERTED index has been created`` (measured). Because the
        recall legs are gathered without ``return_exceptions``, one failing leg
        fails the whole search request, so the window was a source of 500s.

        **Cadence** — :class:`CascadeWorker` runs this on a slow loop
        (default 12h per kind). Frequency is bounded by the rebuild
        cost, not by correctness — even daily is fine functionally;
        12h is a conservative pick to keep file/UUID counts well below
        any FD ceiling under steady-state ingest.

        **When to remove** — once lancedb exposes ``num_indices_to_merge``
        on the async Python API **and** the embedded ``lance crate``
        ships the working merge implementation, delete this method and
        switch to ``optimize(num_indices_to_merge=1)`` in the regular
        ``optimize()`` path. Tracking issues / context:

        - https://github.com/lancedb/lancedb/issues/2193
        - https://github.com/lancedb/lancedb/issues/3177
        - https://github.com/lance-format/lance/pull/6711 (partial fix
          in lance v7.0.0)
        - https://docs.rs/lancedb/latest/lancedb/table/struct.OptimizeOptions.html
        """
        async with self._locked(_REBUILD_TIMEOUT_SECONDS, "rebuild_indexes"):
            table = await self._table()
            # Replace in place rather than drop-then-create. The live columns
            # must never be left without an index: FTS does not degrade when
            # its index is missing the way vector search does — it raises
            # ``Cannot perform full text search unless an INVERTED index has
            # been created``, and the recall legs are gathered without
            # ``return_exceptions``, so the whole search request 500s. Only
            # indexes on columns that are no longer indexed at all get dropped;
            # nothing queries those, so their drop opens no window.
            wanted = set(self.schema.BM25_FIELDS or ())
            for idx in await table.list_indices():
                if not wanted.intersection(idx.columns or ()):
                    await table.drop_index(idx.name)
            await self.schema.ensure_fts_indexes(table, replace=True)

    # ── Read ───────────────────────────────────────────────────────────────

    async def count(self) -> int:
        """Total row count."""
        async with self._deadline(_READ_TIMEOUT_SECONDS, "count"):
            table = await self._table()
            return await table.count_rows()

    async def get_by_id(
        self,
        id_value: str,
        *,
        id_field: str = "id",
    ) -> T | None:
        """Fetch one row by scalar PK; ``None`` if missing.

        Uses LanceDB scalar filter ``<id_field> = '<id_value>'``. Single
        quotes in ``id_value`` are doubled to avoid breaking the SQL-like
        predicate; everos's PK convention is ``<owner_id>_<entry_id>``
        which never contains quotes, so the escape is defensive.
        """
        async with self._deadline(_READ_TIMEOUT_SECONDS, "get_by_id"):
            table = await self._table()
            rows = (
                await table.query()
                .where(f"{id_field} = '{_q(id_value)}'")
                .limit(1)
                .to_list()
            )
        if not rows:
            return None
        return self.schema.model_validate(rows[0])

    async def find_where(
        self,
        where: str,
        *,
        limit: int = 100,
    ) -> list[T]:
        """Scalar query returning *typed* schema instances.

        Like :meth:`search` but returns ``list[T]`` rather than raw
        LanceDB row dicts. No vector ANN; pure scalar filter only.
        Use :meth:`search` when you need ``_distance`` or want to mix
        ANN with filters.
        """
        async with self._deadline(_READ_TIMEOUT_SECONDS, "find_where"):
            table = await self._table()
            rows = await table.query().where(where).limit(limit).to_list()
        return [self.schema.model_validate(r) for r in rows]

    async def find_one_where(self, where: str) -> T | None:
        """Single-row variant of :meth:`find_where` (``None`` if no match)."""
        rows = await self.find_where(where, limit=1)
        return rows[0] if rows else None

    async def find_where_paginated(
        self,
        where: str,
        *,
        sort_by: str,
        descending: bool = True,
        page: int = 1,
        page_size: int = 20,
        max_fetch: int = 20000,
    ) -> tuple[list[T], int]:
        """Paginated scalar query with in-memory sort.

        LanceDB has no native ``ORDER BY``. The chassis fetches up to
        ``max_fetch`` rows matching ``where``, sorts the resulting Arrow
        table by ``sort_by``, then slices ``page`` × ``page_size``. The
        *true* row count of the predicate is returned alongside the
        page so callers can render pagination controls without a second
        query.

        Args:
            where: SQL-like scalar predicate. Required (no implicit
                full-table scan from ``find_where_paginated``).
            sort_by: Column name to sort the result set by.
            descending: ``True`` (default) → newest first; ``False`` →
                ascending.
            page: 1-indexed page number.
            page_size: Rows per page.
            max_fetch: Cap on rows pulled before the in-memory sort.
                When the predicate matches more rows than this cap the
                page is sorted over an *arbitrary* prefix and the page
                contents are only approximately correct — the chassis
                emits a warning so the caller learns about the
                truncation.

        Returns:
            ``(rows, total)`` — ``rows`` is the typed page,
            ``total`` is ``count_rows(filter=where)`` (the predicate's
            true match count, regardless of ``max_fetch``).
        """
        async with self._deadline(_READ_TIMEOUT_SECONDS, "find_where_paginated"):
            table = await self._table()
            total = await table.count_rows(filter=where)
            if total > max_fetch:
                logger.warning(
                    "find_where_paginated truncated",
                    extra={
                        "table": self.table_name,
                        "where": where,
                        "total": total,
                        "max_fetch": max_fetch,
                    },
                )
            arrow_tbl = await table.query().where(where).limit(max_fetch).to_arrow()
        order = "descending" if descending else "ascending"
        arrow_tbl = arrow_tbl.sort_by([(sort_by, order)])
        offset = (page - 1) * page_size
        page_rows = arrow_tbl.slice(offset, page_size)
        return (
            [self.schema.model_validate(r) for r in page_rows.to_pylist()],
            total,
        )

    async def find_by_owner(
        self,
        owner_id: str,
        *,
        limit: int = 100,
    ) -> list[T]:
        """Fetch rows by ``owner_id`` (5 business tables share this column)."""
        return await self.find_where(
            f"owner_id = '{_q(owner_id)}'",
            limit=limit,
        )

    async def find_by_md_path(self, md_path: str) -> T | None:
        """Reverse-lookup from md path (cascade maps md edit → row)."""
        return await self.find_one_where(f"md_path = '{_q(md_path)}'")

    async def search(
        self,
        *,
        vector: Sequence[float] | None = None,
        where: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Hybrid search: optional vector ANN + scalar SQL-like predicate.

        Args:
            vector: Embedding to find nearest rows for; ``None`` skips ANN.
            where: SQL-like predicate (e.g. ``"tags = 'meeting'"``).
            limit: Max rows.

        Returns:
            List of row dicts (LanceDB native shape — fields depend on
            ``schema``; ``_distance`` added when ``vector`` is given).
        """
        async with self._deadline(_READ_TIMEOUT_SECONDS, "search"):
            table = await self._table()
            q = table.query()
            if vector is not None:
                q = q.nearest_to(list(vector))
            if where is not None:
                q = q.where(where)
            return await q.limit(limit).to_list()

    # ── Update ─────────────────────────────────────────────────────────────

    async def update(
        self,
        updates: dict[str, Any],
        *,
        where: str,
    ) -> None:
        """Partial column update for rows matching ``where``.

        Wraps ``AsyncTable.update`` — sets specific column values without
        rewriting the full row. Useful for lightweight metadata patches
        (e.g. setting ``deprecated_by``) where a full embed+upsert cycle
        is unnecessary.

        Args:
            updates: Column-name to new-value mapping.
            where: SQL-like predicate scoping the update.
        """
        async with self._locked(_WRITE_TIMEOUT_SECONDS, "update"):
            table = await self._table()
            await table.update(updates, where=where)

    # ── Delete ─────────────────────────────────────────────────────────────

    async def delete(self, predicate: str) -> None:
        """Delete rows matching a SQL-like predicate."""
        async with self._locked(_WRITE_TIMEOUT_SECONDS, "delete"):
            table = await self._table()
            await table.delete(predicate)

    async def delete_by_md_path(self, md_path: str) -> int:
        """Delete every row whose ``md_path`` matches; return rows deleted.

        Cascade handler calls this when an md file is removed on disk
        (or when reverse-reconcile discovers an orphaned LanceDB row).
        Single quotes in ``md_path`` are doubled defensively.
        """
        async with self._locked(_WRITE_TIMEOUT_SECONDS, "delete_by_md_path"):
            table = await self._table()
            result = await table.delete(f"md_path = '{_q(md_path)}'")
        return int(result.num_deleted_rows)


class LanceDailyLogRepoBase[T: BaseLanceTable](LanceRepoBase[T]):
    """LanceRepoBase + queries unique to daily-log tables.

    Daily-log tables (``episode`` / ``atomic_fact`` / ``foresight`` /
    ``agent_case``) share a fixed schema slice: ``entry_id`` (md seq
    id), ``session_id`` (conversation scope), and ``parent_type`` /
    ``parent_id`` (record lineage). The queries below compose those
    columns; ``agent_skill`` is *not* a daily-log (it is a named
    single-file entity) and uses :class:`LanceRepoBase` directly.
    """

    async def find_by_owner_entry(
        self,
        owner_id: str,
        entry_id: str,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> T | None:
        """Single point-query by ``(app, project, owner_id, entry_id)``.

        ``entry_id`` is only unique within a (app, project, owner) scope —
        the same ``ac_<date>_<seq>`` recurs in another space — so the
        scope segments are part of the predicate to avoid a cross-space hit.
        """
        return await self.find_one_where(
            f"owner_id = '{_q(owner_id)}' AND entry_id = '{_q(entry_id)}' "
            f"AND app_id = '{_q(app_id)}' AND project_id = '{_q(project_id)}'"
        )

    async def find_by_owner_entries(
        self,
        owner_id: str,
        entry_ids: Sequence[str],
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> list[T]:
        """Bulk point-query by ``(app, project, owner_id, entry_id IN ...)``.

        Empty ``entry_ids`` short-circuits to ``[]`` rather than emit a
        ``WHERE entry_id IN ()`` predicate (LanceDB rejects empty
        tuples). The query's ``limit`` is bound to ``len(entry_ids)``
        because at most one row per id can exist under one (app, project,
        owner) scope.
        """
        if not entry_ids:
            return []
        quoted = ", ".join(f"'{_q(eid)}'" for eid in entry_ids)
        return await self.find_where(
            f"owner_id = '{_q(owner_id)}' AND entry_id IN ({quoted}) "
            f"AND app_id = '{_q(app_id)}' AND project_id = '{_q(project_id)}'",
            limit=len(entry_ids),
        )

    async def find_by_session(
        self,
        owner_id: str,
        session_id: str,
        *,
        limit: int = 100,
    ) -> list[T]:
        """Every row in one conversation ``session_id`` under ``owner_id``."""
        return await self.find_where(
            f"owner_id = '{_q(owner_id)}' AND session_id = '{_q(session_id)}'",
            limit=limit,
        )

    async def find_by_parent(
        self,
        parent_type: str,
        parent_id: str,
        *,
        limit: int = 100,
    ) -> list[T]:
        """Every row whose parent matches ``(parent_type, parent_id)``."""
        return await self.find_where(
            f"parent_type = '{_q(parent_type)}' AND parent_id = '{_q(parent_id)}'",
            limit=limit,
        )
