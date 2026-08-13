"""LanceDB business persistence layer.

Sits on top of :mod:`everos.core.persistence.lancedb` (connection
factory + ``BaseLanceTable`` + ``LanceRepoBase``) and provides:

    * lazy process-wide connection + per-name table cache
      (:mod:`.lancedb_manager`)
    * concrete schemas under :mod:`.tables`
    * concrete repository singletons under :mod:`.repos`

External usage::

    from everos.infra.persistence.lancedb import (
        get_connection, get_table, dispose_connection,
        Episode, AtomicFact, Foresight, AgentCase, AgentSkill, UserProfile,
        KnowledgeTopic,
        episode_repo, atomic_fact_repo, foresight_repo,
        agent_case_repo, agent_skill_repo, user_profile_repo,
        knowledge_topic_repo,
    )

Three index kinds: scalar / BM25 / vector. Tables are created lazily on
first access; row population is the cascade daemon's job (see
``docs/cascade_runbook.md``).
"""

import contextlib
import datetime as dt

from everos.core.observability.logging import get_logger
from everos.core.persistence import BaseLanceTable, MemoryRoot, memory_root_lock

# Importing ``tables`` registers every business :class:`BaseLanceTable`
# schema so callers can rely on the package alone to surface every schema.
from . import tables as tables
from .lancedb_manager import dispose_connection as dispose_connection
from .lancedb_manager import drop_tables as _drop_tables
from .lancedb_manager import get_connection as get_connection
from .lancedb_manager import get_table as get_table
from .repos import agent_case_repo as agent_case_repo
from .repos import agent_skill_repo as agent_skill_repo
from .repos import atomic_fact_repo as atomic_fact_repo
from .repos import episode_repo as episode_repo
from .repos import foresight_repo as foresight_repo
from .repos import knowledge_topic_repo as knowledge_topic_repo
from .repos import user_profile_repo as user_profile_repo
from .tables import AgentCase as AgentCase
from .tables import AgentSkill as AgentSkill
from .tables import AtomicFact as AtomicFact
from .tables import Episode as Episode
from .tables import Foresight as Foresight
from .tables import KnowledgeTopic as KnowledgeTopic
from .tables import ParentType as ParentType
from .tables import UserProfile as UserProfile

_BUSINESS_SCHEMAS = (
    Episode,
    AtomicFact,
    Foresight,
    AgentCase,
    AgentSkill,
    UserProfile,
    KnowledgeTopic,
)


class LanceDBSchemaMismatchError(RuntimeError):
    """Raised at startup when an on-disk LanceDB table's columns drift
    from the corresponding Pydantic schema.

    Cascade re-builds LanceDB from md (the SoT), so the recovery is
    deterministic: ``everos cascade rebuild`` drops the business tables
    and re-indexes from md, preserving SQLite state that is *not*
    rebuildable from md (notably ``unprocessed_buffer`` — messages not
    yet extracted). The error message surfaces that command; see
    ``docs/cascade_runbook.md`` for the wider context.
    """


class LanceDBMigrationError(RuntimeError):
    """Raised when :func:`migrate_table_schemas` cannot complete and
    startup must abort rather than continue in a half-migrated state."""


_FTS_INDEX_SCHEMA_VERSION = 2
"""Bump when the FTS index build config changes so existing on-disk
indexes get rebuilt at startup. v2 = ``with_position=False`` (see
:meth:`BaseLanceTable.ensure_fts_indexes` + lance-format/lance#7653)."""


async def migrate_fts_indexes() -> None:
    """One-time rebuild of FTS indexes that predate the current config.

    Older indexes were built with ``with_position=True``; that position
    posting List overflows lance's compaction once it grows large
    (``Max offset exceeds length of values``, lance-format/lance#7653),
    which aborts ``optimize()`` — including version cleanup — so the
    index dir grows unbounded until the disk fills.

    Rebuilds every business table's FTS index with the current
    :meth:`BaseLanceTable.ensure_fts_indexes` config (``with_position``
    now off) and reclaims the orphaned index files / data fragments the
    crashed-optimize churn left behind. Guarded by a version marker in
    the LanceDB dir so it runs at most once per bump; the rebuild is
    O(N) but only on the first startup after upgrade.

    Cross-process serialization: wrapped in :func:`memory_root_lock` so a
    server startup racing with a concurrent ``everos cascade`` command
    (both call this via :func:`ensure_business_indexes`) can't attempt
    two concurrent rebuild passes. First process runs the migration and
    writes the marker; second waits for the lock, re-checks the marker
    on entry, and no-ops. Removes the same TOCTOU window between marker
    read and the drop-index/rebuild loop that :issue:`M7` from the
    PR #361 review flagged on the sibling schema migration.
    """
    logger = get_logger(__name__)
    memory_root = MemoryRoot.resolve()
    async with memory_root_lock(memory_root):
        marker = memory_root.lancedb_dir / ".fts_index_version"
        try:
            current = int(marker.read_text().strip()) if marker.exists() else 0
        except (ValueError, OSError):
            current = 0
        if current >= _FTS_INDEX_SCHEMA_VERSION:
            # Another process finished the migration while we waited
            # for the lock. Nothing to do.
            return
        logger.info("fts_index_migration_started", target=_FTS_INDEX_SCHEMA_VERSION)
        for schema in _BUSINESS_SCHEMAS:
            if not schema.BM25_FIELDS:
                continue
            table = await get_table(schema.TABLE_NAME, schema)
            # Drop existing indexes (everos only builds FTS here; mirrors
            # LanceRepoBase.rebuild_indexes) then rebuild with the new config.
            for idx in await table.list_indices():
                await table.drop_index(idx.name)
            await schema.ensure_fts_indexes(table)
            # Reclaim the orphaned index dirs + data fragments the crashed
            # optimize loop piled up. Safe now: the crashing index is gone,
            # so compaction no longer decodes a position List.
            with contextlib.suppress(Exception):
                await table.optimize(cleanup_older_than=dt.timedelta(seconds=0))
        marker.write_text(str(_FTS_INDEX_SCHEMA_VERSION))
        logger.info("fts_index_migration_done", version=_FTS_INDEX_SCHEMA_VERSION)


_TABLE_SCHEMA_VERSION = 2
"""Bump when column nullability or presence changes on business tables.
v2 = ``vector: Vector(_DIM) | None`` on the 6 business tables that carry
a vector column (see :data:`BUSINESS_SCHEMAS_WITH_VECTOR`).

.. note::

   :func:`migrate_table_schemas` currently hard-codes the v2 alter — it
   does not implement per-version dispatch. Bumping this constant to
   ``3`` without first adding a dispatch table will short-circuit the
   v3 migration entirely (the current body only compares
   ``current >= _TABLE_SCHEMA_VERSION`` and then runs the v2-specific
   alter unconditionally). Any future v3 migration MUST:

   1. Add a ``_MIGRATIONS: dict[int, Callable[[], Awaitable[None]]]``
      registry mapping target version → migration coroutine.
   2. Rewrite :func:`migrate_table_schemas` to run every migration from
      ``current + 1`` up to ``_TABLE_SCHEMA_VERSION`` in order, updating
      the marker after each step so a mid-migration crash resumes at
      the next step instead of replaying earlier ones.

   Accepted as a design gap in PR #361 review finding #5 — not blocking
   today because there is no v3.
"""


BUSINESS_SCHEMAS_WITH_VECTOR: tuple[type[BaseLanceTable], ...] = (
    Episode,
    AtomicFact,
    Foresight,
    AgentCase,
    AgentSkill,
    KnowledgeTopic,
)
"""Business schemas whose ``vector`` column needs the v2 nullability
migration. ``Episode.subject_vector`` was already nullable in v1.1.1 and
is intentionally left out of this migration. ``UserProfile`` has no
vector column; ``knowledge_document`` has no LanceDB table."""


async def migrate_table_schemas() -> None:
    """One-time ``alter_columns`` making ``vector`` nullable on business tables.

    Soft-dependency embedding mode lets cascade write ``vector=None`` for
    rows it cannot embed; the on-disk column must allow that. LanceDB's
    ``alter_columns`` is metadata-only (no data movement, no re-embed),
    so this runs in milliseconds. Guarded by ``.table_schema_version``
    so it runs at most once per version bump; per-table nullable check
    is defense-in-depth so an already-migrated (or freshly-created,
    already-nullable) table is a no-op rather than a redundant alter.

    Cross-process serialization: wrapped in :func:`memory_root_lock` so
    a server startup racing with a concurrent ``everos cascade`` command
    (both call this via :func:`ensure_business_indexes`) doesn't attempt
    two concurrent alter passes. First process runs the migration and
    writes the marker; second waits for the lock, re-checks the marker
    on entry, and no-ops. Removes the TOCTOU window between marker
    read and alter that :issue:`M7` from the PR #361 review flagged.

    Fail-loud semantics preserved: if ``alter_columns`` genuinely fails
    for any table (LanceDB version mismatch, corrupted index), raises
    :class:`LanceDBMigrationError`. Startup aborts rather than
    continuing in a half-migrated state — cascade would otherwise write
    ``vector=None`` (soft-dependency embedding) into a still NOT-NULL
    column and every row would silently fail. Recovery escalates from
    a plain restart (transient hiccup) to ``everos cascade rebuild``,
    which re-indexes from md (the SoT) *and* re-enqueues every file —
    unlike deleting the index dir, which leaves the queue ``done`` and
    yields an empty index.
    """
    logger = get_logger(__name__)
    memory_root = MemoryRoot.resolve()
    async with memory_root_lock(memory_root):
        marker = memory_root.lancedb_dir / ".table_schema_version"
        try:
            current = int(marker.read_text().strip()) if marker.exists() else 0
        except (ValueError, OSError):
            current = 0
        if current >= _TABLE_SCHEMA_VERSION:
            # Another process finished the migration while we waited
            # for the lock. Nothing to do.
            return
        logger.info("table_schema_migration_started", target=_TABLE_SCHEMA_VERSION)
        failed_schemas: list[str] = []
        for schema in BUSINESS_SCHEMAS_WITH_VECTOR:
            table = await get_table(schema.TABLE_NAME, schema)
            arrow_schema = await table.schema()
            if arrow_schema.field("vector").nullable:
                continue
            try:
                await table.alter_columns({"path": "vector", "nullable": True})
            except Exception as exc:
                logger.warning(
                    "table_schema_migration_alter_failed",
                    table=schema.TABLE_NAME,
                    error=str(exc),
                )
                failed_schemas.append(schema.TABLE_NAME)

        if failed_schemas:
            logger.error(
                "table_schema_migration_incomplete",
                failed_schemas=failed_schemas,
            )
            raise LanceDBMigrationError(
                f"LanceDB nullable-vector migration failed for tables "
                f"{failed_schemas!r}. Startup aborted to prevent cascade "
                f"from writing NULL vectors into NOT-NULL columns. This "
                f"typically indicates a LanceDB version mismatch or a "
                f"corrupted index. Recovery, in order of least- to "
                f"most-destructive: (1) restart the process; a transient "
                f"filesystem or LanceDB-side hiccup may resolve. (2) If "
                f"the error persists, run `everos cascade rebuild` (with "
                f"the server stopped) — it re-indexes from source markdown "
                f"and preserves un-extracted buffered messages. Do NOT "
                f"just delete `{memory_root.lancedb_dir}`: that leaves the "
                f"cascade queue marked done, so nothing re-indexes and the "
                f"index comes back empty."
            )

        marker.parent.mkdir(parents=True, exist_ok=True)
        try:
            marker.write_text(str(_TABLE_SCHEMA_VERSION))
        except OSError:
            logger.error("table_schema_migration_marker_write_failed", path=str(marker))
            raise
        logger.info("table_schema_migration_done", version=_TABLE_SCHEMA_VERSION)


async def ensure_business_indexes() -> None:
    """Ensure FTS (BM25) indexes for every business table (idempotent).

    Called once at startup by :class:`LanceDBLifespanProvider`. First
    runs :func:`migrate_table_schemas` (schema first, one-time,
    marker-guarded) to make ``vector`` columns nullable, then
    :func:`migrate_fts_indexes` (index second, one-time, marker-guarded)
    to rebuild any pre-fix ``with_position=True`` indexes, then walks
    the business schemas (each owns its ``TABLE_NAME`` + ``BM25_FIELDS``),
    opens each table via :func:`get_table`, and delegates to
    ``schema.ensure_fts_indexes(table)``. Already-indexed columns are
    skipped, so re-runs are no-ops.

    Adding a new business table = adding it to ``_BUSINESS_SCHEMAS``;
    everything else (table name, columns to index) reads off the
    schema's ClassVars.
    """
    await migrate_table_schemas()
    await migrate_fts_indexes()
    for schema in _BUSINESS_SCHEMAS:
        table = await get_table(schema.TABLE_NAME, schema)
        await schema.ensure_fts_indexes(table)


async def verify_business_schemas() -> None:
    """Fail loud at startup if an existing LanceDB table's columns don't
    match its current Pydantic schema — in **name or type**.

    LanceDB doesn't migrate columns automatically; an older index dir
    would fail unpredictably on upsert. Checking the schema up-front
    turns that into a clean startup error pointing the user at the
    recovery path (``everos cascade rebuild`` — re-indexes from md,
    preserving un-extracted buffered messages; see
    ``docs/cascade_runbook.md``). A bare ``rm -rf`` of the index dir is
    *not* the recovery — it leaves the cascade queue marked ``done`` so
    nothing re-indexes and the index comes back empty.

    Both dimensions are checked against ``schema.to_arrow_schema()`` —
    the exact schema ``get_table`` builds the table from, so a healthy
    table never false-positives:

    * **Column set** — a missing / extra column (e.g. a pre-``content_sha256``
      table) is caught by name.
    * **Column type** — a column whose on-disk Arrow type drifted from
      the current schema. This is the class of drift behind EverOS #337:
      an ``episode.subject_vector`` column left as ``string`` (or ``null``)
      by an older build, while the current schema declares a 1024-d
      ``fixed_size_list``. The name matches, so a name-only check waves it
      through and it detonates deep inside ``merge_insert`` as an opaque
      ``LanceError(IO): Spill has sent an error``. Comparing types surfaces
      it here instead.
    """
    for schema in _BUSINESS_SCHEMAS:
        table = await get_table(schema.TABLE_NAME, schema)
        on_disk = await table.schema()
        expected = schema.to_arrow_schema()
        on_disk_names = set(on_disk.names)
        expected_names = set(expected.names)
        missing = expected_names - on_disk_names
        extra = on_disk_names - expected_names
        # Type drift on columns present in both, compared against the
        # authoritative to_arrow_schema() Arrow types.
        type_drift = [
            f"{name}: on-disk {on_disk.field(name).type} "
            f"!= expected {expected.field(name).type}"
            for name in sorted(on_disk_names & expected_names)
            if not on_disk.field(name).type.equals(expected.field(name).type)
        ]
        if missing or extra or type_drift:
            raise LanceDBSchemaMismatchError(
                f"LanceDB table {schema.TABLE_NAME!r} schema drift: "
                f"missing={sorted(missing)}, extra={sorted(extra)}, "
                f"type_drift={type_drift}.\n"
                "Recover with `everos cascade rebuild` (stop the server "
                "first): it drops and re-indexes from md, preserving "
                "un-extracted buffered messages. Restarting will not "
                "clear this — the startup migrations only alter column "
                "nullability, never a column's name or type, so a "
                "name/type drift never resolves on its own."
            )


async def drop_business_tables() -> list[str]:
    """Drop every business LanceDB table; return the names dropped.

    The tables are a rebuildable projection of markdown, so dropping is
    non-destructive to memory content — ``cascade rebuild`` recreates and
    re-populates them from md. Evicts the dropped tables from the manager
    cache so a later :func:`get_table` reopens the fresh table.
    """
    return await _drop_tables([schema.TABLE_NAME for schema in _BUSINESS_SCHEMAS])


__all__ = [
    "BUSINESS_SCHEMAS_WITH_VECTOR",
    "AgentCase",
    "AgentSkill",
    "AtomicFact",
    "Episode",
    "Foresight",
    "KnowledgeTopic",
    "LanceDBMigrationError",
    "LanceDBSchemaMismatchError",
    "ParentType",
    "UserProfile",
    "agent_case_repo",
    "agent_skill_repo",
    "atomic_fact_repo",
    "dispose_connection",
    "drop_business_tables",
    "ensure_business_indexes",
    "episode_repo",
    "foresight_repo",
    "get_connection",
    "get_table",
    "knowledge_topic_repo",
    "migrate_fts_indexes",
    "migrate_table_schemas",
    "user_profile_repo",
    "verify_business_schemas",
]
