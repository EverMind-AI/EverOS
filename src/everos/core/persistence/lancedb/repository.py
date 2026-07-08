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
from collections.abc import Sequence
from typing import Any, ClassVar

from lancedb import AsyncTable

from everos.core.observability.logging import get_logger

from .base import BaseLanceTable

logger = get_logger(__name__)


def _q(value: str) -> str:
    """Escape single quotes for a LanceDB SQL-like ``where`` predicate.

    LanceDB has no parameterised query API; predicates are strings.
    Doubling the quote (``'`` → ``''``) is the SQL-standard way to keep
    a literal single quote inside a single-quoted string. everos's PK
    convention (``<owner_id>_<entry_id>``) never carries quotes — this
    is defensive.
    """
    return value.replace("'", "''")


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
        table = await self._table()
        async with self._write_lock(self.table_name):
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
        table = await self._table()
        async with self._write_lock(self.table_name):
            await (
                table.merge_insert(by)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(list(records))
            )

    # ── Maintenance ────────────────────────────────────────────────────────

    async def optimize(self, *, cleanup_older_than: dt.timedelta | None = None) -> None:
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
        - **Storage hygiene** — with ``cleanup_older_than`` it prunes
          replaced fragments / stale manifests / dead index files,
          bounding the on-disk file count (and FD usage at scan time).

        Cascade triggers this through a per-kind throttle + trailing
        edge scheduler (``CascadeWorker._schedule_optimize``): at most
        one run per throttle window per kind, decoupled from the drain
        loop, with a 60s heartbeat sweep as a safety net. Cost is
        O(N) data-rewrite per optimized fragment; the throttle is how
        we cap it under sustained write pressure. Because visibility no
        longer depends on it, the throttle window can be generous.

        Args:
            cleanup_older_than: When set, also prune (physically delete)
                files belonging to dataset versions older than this
                interval. ``None`` (default) compacts only — historical
                manifests, replaced data fragments, and stale index
                UUID files are kept on disk forever, which inflates the
                file count (and FD usage at scan time) without bound.
                Cascade passes a non-None value on a slower beat
                (``CascadeWorker._optimize_prune_interval``) so the
                hot drain path stays cheap. Note: this does *not*
                shrink **active** index internals (FTS ``part_N`` count
                or vector index UUID count) — those only collapse via
                ``drop_index + create_index``, which is not done here.
        """
        table = await self._table()
        await table.optimize(cleanup_older_than=cleanup_older_than)

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

        This method is the workaround: drop every existing index and
        rebuild from the schema's ``ensure_fts_indexes`` contract. The
        rebuild is **O(N) full retrain** but cheap in practice (~0.3s
        for 50k rows × 2 FTS columns on local SSD), and during the
        window LanceDB transparently falls back to brute-force scan so
        queries and writes stay available.

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
        table = await self._table()
        async with self._write_lock(self.table_name):
            for idx in await table.list_indices():
                await table.drop_index(idx.name)
            await self.schema.ensure_fts_indexes(table)

    # ── Read ───────────────────────────────────────────────────────────────

    async def count(self) -> int:
        """Total row count."""
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
        table = await self._table()
        async with self._write_lock(self.table_name):
            await table.update(updates, where=where)

    # ── Delete ─────────────────────────────────────────────────────────────

    async def delete(self, predicate: str) -> None:
        """Delete rows matching a SQL-like predicate."""
        table = await self._table()
        async with self._write_lock(self.table_name):
            await table.delete(predicate)

    async def delete_by_md_path(self, md_path: str) -> int:
        """Delete every row whose ``md_path`` matches; return rows deleted.

        Cascade handler calls this when an md file is removed on disk
        (or when reverse-reconcile discovers an orphaned LanceDB row).
        Single quotes in ``md_path`` are doubled defensively.
        """
        table = await self._table()
        async with self._write_lock(self.table_name):
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
