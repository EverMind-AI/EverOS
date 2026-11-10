"""The LanceDB derived-index backend over one connection.

:class:`LanceIndexBackend` owns the LanceDB side of the derived index: the
seven business repositories built over the application's
:class:`~.lancedb_manager.LanceDbConnection`, the one-time schema / FTS
migrations, schema verification, and the embedding-model attestation.
"""

from __future__ import annotations

__all__ = [
    "BUSINESS_SCHEMAS_WITH_VECTOR",
    "LanceDBMigrationError",
    "LanceDBSchemaMismatchError",
    "LanceIndexBackend",
]

import contextlib
import datetime as dt
from collections.abc import Awaitable, Callable, Sequence
from typing import ClassVar

import pyarrow as pa
from lancedb import AsyncTable

from everos.core.errors import IndexSchemaMismatchError
from everos.core.observability.logging import get_logger
from everos.infra.persistence.index.repositories import (
    BUSINESS_SCHEMAS,
    IndexRepositories,
)
from everos.infra.persistence.index.schema import IndexSchema, schema_for
from everos.infra.persistence.locking import memory_root_lock

from ..records import (
    EMBEDDING_MODEL_KEY,
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    ImageMemory,
    IndexRecord,
    KnowledgeTopic,
    UserProfile,
)
from ._base import indexes
from ._base.arrow import arrow_schema_for
from ._repos.agent_case import _AgentCaseRepo
from ._repos.agent_skill import _AgentSkillRepo
from ._repos.atomic_fact import _AtomicFactRepo
from ._repos.episode import _EpisodeRepo
from ._repos.foresight import _ForesightRepo
from ._repos.image_memory import _ImageMemoryRepo
from ._repos.knowledge_topic import _KnowledgeTopicRepo
from ._repos.user_profile import _UserProfileRepo
from .lancedb_manager import LanceDbConnection

logger = get_logger(__name__)


class LanceDBSchemaMismatchError(IndexSchemaMismatchError):
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
    """Raised when :meth:`migrate_table_schemas` cannot complete and
    startup must abort rather than continue in a half-migrated state."""


_FTS_INDEX_SCHEMA_VERSION = 2
"""Bump when the FTS index build config changes so existing on-disk
indexes get rebuilt at startup. v2 = ``with_position=False`` (see
:func:`.base.indexes.ensure_fts_indexes` + lance-format/lance#7653)."""


_PROVENANCE_COLUMN = "vector"
"""Column the attestation hangs on.

Six tables carry a vector column and ``Episode`` carries two, so writing all
of them would raise "how many agreeing copies count as agreement" on the next
schema change. ``episode.vector`` is the one column that exists in every
deployment and is written first; ``subject_vector`` comes from the same model
and is not recorded separately.
"""

_TABLE_SCHEMA_VERSION = 5
"""Bump when column nullability or presence changes on business tables.

v2 = ``vector: DenseVector | None`` on the 6 business tables that carry
a vector column (see :data:`BUSINESS_SCHEMAS_WITH_VECTOR`).

v3 = ``image_memory.visual_vector: DenseVector | None`` — the picture's own
embedding, added so a question about what a picture looks like can reach it
through the vision model's space rather than only its description's. Added
as a nullable column at ``[vision_embedding] dim``.

Existing rows keep NULL after this migration, and that is deliberate: a row's
visual vector is written when its picture is projected, and an unchanged md
entry is skipped by the content digest, so the way to fill a store that
already has pictures is ``everos cascade rebuild`` (it re-projects from md
into empty tables) rather than a restart. A deployment that never turns
``[vision_embedding]`` on never needs the column filled at all — the visual
route skips NULL rows.

v4 = the group-memory columns (docs/group-memory.md, section 6.3):
``episode.subject_ids / origin / source_start / source_end`` and
``image_memory.sender_ids / subject_ids / origin / source_start / source_end``.
Added nullable; an existing row holds NULL and reads ``[]`` / ``None`` /
``timestamp``. Only rows a group owns are ever compared on the window columns,
and those are written with them.

v5 = the rest of group memory's columns (docs/group-memory.md, sections 6.3
and 9.1): ``user_profile.subject_id``, the member a ``group`` row is about,
and ``agent_case.source_group_id``, the group a case was learned in. Added
nullable; a personal profile row and a case from no group hold NULL.

:meth:`migrate_table_schemas` dispatches through :data:`_MIGRATIONS`, one
coroutine per version, so a new version is one entry — not an edit to a
hard-coded body that silently replays the previous step.
"""


BUSINESS_SCHEMAS_WITH_VECTOR: tuple[type[IndexRecord], ...] = (
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
vector column; ``knowledge_document`` has no LanceDB table.

An explicit tuple rather than "every table with a vector column": it names the
tables that existed when v2 shipped, so a vector table added later -- created
nullable from the start -- does not silently join a historical migration."""


class LanceIndexBackend:
    """Own LanceDB schema, index and repository lifecycle over one connection.

    The ``IndexBackend`` port the application selects when ``index.backend``
    is ``"lancedb"``.

    Args:
        connection: The application's LanceDB connection.
        dim: ``[embedding] dim`` -- the default width every vector column is
            built at.
        column_dims: Per-column width overrides, applied to every repository
            and to the schema checks (only tables that declare the column
            use them).
        schemas: The business tables the per-table sweeps (migrations,
            verification, drop) walk; every one of them in production.
    """

    name: ClassVar[str] = "lancedb"

    def __init__(
        self,
        connection: LanceDbConnection,
        *,
        dim: int,
        column_dims: tuple[tuple[str, int], ...] = (),
        schemas: Sequence[type[IndexRecord]] = BUSINESS_SCHEMAS,
    ) -> None:
        self._connection = connection
        self._vector_dim = dim
        self._column_dims = column_dims
        self._schemas = tuple(schemas)
        self.repositories = IndexRepositories(
            episode=_EpisodeRepo(source=connection, dim=dim),
            atomic_fact=_AtomicFactRepo(source=connection, dim=dim),
            foresight=_ForesightRepo(source=connection, dim=dim),
            agent_case=_AgentCaseRepo(source=connection, dim=dim),
            agent_skill=_AgentSkillRepo(source=connection, dim=dim),
            image_memory=_ImageMemoryRepo(
                source=connection, dim=dim, column_dims=column_dims
            ),
            user_profile=_UserProfileRepo(source=connection, dim=dim),
            knowledge_topic=_KnowledgeTopicRepo(source=connection, dim=dim),
        )

    async def connect(self) -> object:
        return await self._connection.connect()

    async def shutdown(self) -> None:
        await self._connection.close()

    def _index_schema(self, record: type[IndexRecord]) -> IndexSchema:
        return schema_for(record, dim=self._vector_dim, column_dims=self._column_dims)

    def _arrow_schema(self, record: type[IndexRecord]) -> pa.Schema:
        return arrow_schema_for(self._index_schema(record))

    async def _open(self, record: type[IndexRecord]) -> AsyncTable:
        return await self._connection.table(
            record.TABLE_NAME, self._arrow_schema(record)
        )

    async def migrate_fts_indexes(self) -> None:
        """One-time rebuild of FTS indexes that predate the current config.

        Older indexes were built with ``with_position=True``; that position
        posting List overflows lance's compaction once it grows large
        (``Max offset exceeds length of values``, lance-format/lance#7653),
        which aborts ``optimize()`` — including version cleanup — so the
        index dir grows unbounded until the disk fills.

        Rebuilds every business table's FTS index with the current
        :func:`.base.indexes.ensure_fts_indexes` config (``with_position``
        now off) and reclaims the orphaned index files / data fragments the
        crashed-optimize churn left behind. Guarded by a version marker in
        the LanceDB dir so it runs at most once per bump; the rebuild is
        O(N) but only on the first startup after upgrade.

        Cross-process serialization: wrapped in :func:`memory_root_lock` so a
        server startup racing with a concurrent ``everos cascade`` command
        (both call this via :meth:`ensure_business_indexes`) can't attempt
        two concurrent rebuild passes. First process runs the migration and
        writes the marker; second waits for the lock, re-checks the marker
        on entry, and no-ops. Removes the same TOCTOU window between marker
        read and the drop-index/rebuild loop that :issue:`M7` from the
        PR #361 review flagged on the sibling schema migration.
        """
        memory_root = self._connection.root
        async with memory_root_lock(memory_root):
            marker = memory_root.lancedb_dir / ".fts_index_version"
            try:
                current = (
                    int(marker.read_text(encoding="utf-8").strip())
                    if marker.exists()
                    else 0
                )
            except (ValueError, OSError):
                current = 0
            if current >= _FTS_INDEX_SCHEMA_VERSION:
                # Another process finished the migration while we waited
                # for the lock. Nothing to do.
                return
            logger.info("fts_index_migration_started", target=_FTS_INDEX_SCHEMA_VERSION)
            for schema in self._schemas:
                if not schema.BM25_FIELDS:
                    continue
                table = await self._open(schema)
                # Drop existing indexes (everos only builds FTS here; mirrors
                # LanceRepoBase.rebuild_indexes) then rebuild with the new config.
                for idx in await table.list_indices():
                    await table.drop_index(idx.name)
                await indexes.ensure_fts_indexes(table, self._index_schema(schema))
                # Reclaim the orphaned index dirs + data fragments the crashed
                # optimize loop piled up. Safe now: the crashing index is gone,
                # so compaction no longer decodes a position List.
                with contextlib.suppress(Exception):
                    await table.optimize(cleanup_older_than=dt.timedelta(seconds=0))
            marker.write_text(str(_FTS_INDEX_SCHEMA_VERSION), encoding="utf-8")
            logger.info("fts_index_migration_done", version=_FTS_INDEX_SCHEMA_VERSION)

    async def migrate_table_schemas(self) -> None:
        """Run every pending on-disk schema migration, in version order.

        Guarded by ``.table_schema_version`` so each step runs once; the marker
        is rewritten after every step, so a crash mid-way resumes at the next
        version instead of replaying the ones already done.

        Cross-process serialization: wrapped in :func:`memory_root_lock` so
        a server startup racing with a concurrent ``everos cascade`` command
        (both call this via :meth:`ensure_business_indexes`) doesn't attempt
        two concurrent alter passes. First process runs the migrations and
        writes the marker; second waits for the lock, re-checks the marker
        on entry, and no-ops.

        Fail-loud semantics: a step that cannot complete raises
        :class:`LanceDBMigrationError` and startup aborts, rather than
        continuing against a half-migrated store. Recovery escalates from a
        plain restart (transient hiccup) to ``everos cascade rebuild``, which
        re-indexes from md (the SoT) *and* re-enqueues every file — unlike
        deleting the index dir, which leaves the queue ``done`` and yields an
        empty index.
        """
        memory_root = self._connection.root
        async with memory_root_lock(memory_root):
            marker = memory_root.lancedb_dir / ".table_schema_version"
            try:
                current = (
                    int(marker.read_text(encoding="utf-8").strip())
                    if marker.exists()
                    else 0
                )
            except (ValueError, OSError):
                current = 0
            if current >= _TABLE_SCHEMA_VERSION:
                # Another process finished the migration while we waited
                # for the lock. Nothing to do.
                return
            logger.info(
                "table_schema_migration_started",
                current=current,
                target=_TABLE_SCHEMA_VERSION,
            )
            marker.parent.mkdir(parents=True, exist_ok=True)
            # Only the versions that actually have a step. v1 never existed as
            # a table-schema migration (the columns were created that way), so
            # a store starting from 0 or from the historical marker value 1
            # runs v2, then v3.
            for version in sorted(v for v in _MIGRATIONS if v > current):
                await _MIGRATIONS[version](self)
                marker.write_text(str(version), encoding="utf-8")
                logger.info("table_schema_migration_step_done", version=version)
            logger.info("table_schema_migration_done", version=_TABLE_SCHEMA_VERSION)

    async def _migrate_v2_nullable_vector(self) -> None:
        """v2 — make ``vector`` nullable on the six vector-carrying tables.

        Soft-dependency embedding mode lets cascade write ``vector=None`` for
        rows it cannot embed; the on-disk column must allow that. LanceDB's
        ``alter_columns`` is metadata-only (no data movement, no re-embed), so
        this runs in milliseconds. The per-table nullable check is
        defense-in-depth so a freshly-created (already-nullable) table is a
        no-op rather than a redundant alter.
        """
        failed_schemas: list[str] = []
        for schema in BUSINESS_SCHEMAS_WITH_VECTOR:
            table = await self._open(schema)
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
                f"just delete `{self._connection.root.lancedb_dir}`: that "
                f"leaves the cascade queue marked done, so nothing re-indexes "
                f"and the index comes back empty."
            )

    async def _migrate_v3_visual_vector(self) -> None:
        """v3 — add ``image_memory.visual_vector`` (the picture's own embedding).

        ``add_columns`` is additive and metadata-only: existing rows get NULL,
        which is exactly the state a deployment without ``[vision_embedding]``
        produces anyway, and the visual route treats NULL as "not embedded"
        rather than as an error. The column is built at the width the running
        configuration declares for it, so a store opened with a different
        vision model is caught by :meth:`verify_business_schemas` afterwards
        rather than silently mis-sized here.
        """
        table = await self._open(ImageMemory)
        on_disk = await table.schema()
        if "visual_vector" in on_disk.names:
            return
        expected = self._arrow_schema(ImageMemory).field("visual_vector")
        try:
            await table.add_columns(pa.schema([expected]))
        except Exception as exc:
            logger.error(
                "table_schema_migration_add_column_failed",
                table=ImageMemory.TABLE_NAME,
                column="visual_vector",
                error=str(exc),
            )
            raise LanceDBMigrationError(
                f"LanceDB v3 migration could not add "
                f"{ImageMemory.TABLE_NAME!r}.visual_vector: {exc}. Startup "
                f"aborted rather than running against a store whose picture "
                f"table lacks the column the vision route searches. Retry; if "
                f"it persists, run `everos cascade rebuild` with the server "
                f"stopped (it re-indexes from md)."
            ) from exc

    async def _migrate_v4_group_columns(self) -> None:
        """v4 -- add the group-memory columns to ``episode`` and ``image_memory``.

        Additive and metadata-only, like v3. Each column is added nullable even
        where the record declares a list: existing rows get NULL, which the read
        side takes as the empty default. A column already there is skipped, so
        a store created at v4 (or a retry after a partial step) is a no-op.
        """
        await self._add_nullable_columns(
            {
                Episode: ("subject_ids", "origin", "source_start", "source_end"),
                ImageMemory: (
                    "sender_ids",
                    "subject_ids",
                    "origin",
                    "source_start",
                    "source_end",
                ),
            },
            version=4,
        )

    async def _migrate_v5_group_owned(self) -> None:
        """v5 -- ``user_profile.subject_id`` and ``agent_case.source_group_id``.

        Additive and metadata-only, like v4: a personal profile row and a
        case learned outside any group read NULL.
        """
        await self._add_nullable_columns(
            {UserProfile: ("subject_id",), AgentCase: ("source_group_id",)},
            version=5,
        )

    async def _add_nullable_columns(
        self,
        wanted: dict[type[IndexRecord], tuple[str, ...]],
        *,
        version: int,
    ) -> None:
        """Add the named columns, nullable, to each table that lacks them.

        Raises:
            LanceDBMigrationError: A table refused the new columns.
        """
        for schema, names in wanted.items():
            table = await self._open(schema)
            on_disk = await table.schema()
            expected = self._arrow_schema(schema)
            missing = [
                pa.field(name, expected.field(name).type, nullable=True)
                for name in names
                if name not in on_disk.names
            ]
            if not missing:
                continue
            try:
                await table.add_columns(pa.schema(missing))
            except Exception as exc:
                logger.error(
                    "table_schema_migration_add_column_failed",
                    table=schema.TABLE_NAME,
                    column=",".join(f.name for f in missing),
                    error=str(exc),
                )
                raise LanceDBMigrationError(
                    f"LanceDB v{version} migration could not add the group "
                    f"columns to {schema.TABLE_NAME!r}: {exc}. Startup aborted "
                    f"rather than running against a table whose group columns "
                    f"are missing. Retry; if it persists, run `everos cascade "
                    f"rebuild` with the server stopped (it re-indexes from md)."
                ) from exc

    # TODO(#477): BTree scalar index on `id` for the business tables — a 4x
    # win on keyed lookups at 100k rows, but it must join rebuild_indexes()
    # and `cascade rebuild` first and be soaked for index-directory growth.
    async def ensure_business_indexes(self) -> None:
        """Ensure FTS (BM25) indexes for every business table (idempotent).

        Called once at startup by :meth:`everos.service.Application.start`. First
        runs :meth:`migrate_table_schemas` (schema first, one-time,
        marker-guarded) to make ``vector`` columns nullable, then
        :meth:`migrate_fts_indexes` (index second, one-time, marker-guarded)
        to rebuild any pre-fix ``with_position=True`` indexes, then walks
        the business schemas (each owns its ``TABLE_NAME`` + ``BM25_FIELDS``),
        opens each table via :meth:`LanceDbConnection.table`, and delegates to
        ``ensure_fts_indexes(table, schema)``. Already-indexed columns are
        skipped, so re-runs are no-ops.

        Adding a new business table = adding it to ``BUSINESS_SCHEMAS``;
        everything else (table name, columns to index) reads off the
        schema's ClassVars.

        Vector (ANN) indexes are deliberately **not** built here. This runs in
        every process that opens the root — the server lifespan and the CLI's
        ``_runtime`` — and a CLI command training an index on a live server's
        table races its commits (soak: ``cascade status`` storms failed with
        ``Retryable commit conflict`` the moment a table crossed the row
        threshold). Vector indexes belong to the cascade worker alone: its
        first rebuild sweep at server start builds a missing one and the heavy
        beat maintains it (:func:`.base.indexes.ensure_vector_indexes`). FTS
        stays here because a search on a column without its inverted index
        raises instead of degrading.
        """
        await self.migrate_table_schemas()
        await self.migrate_fts_indexes()
        for schema in self._schemas:
            table = await self._open(schema)
            await indexes.ensure_fts_indexes(table, self._index_schema(schema))

    async def verify_business_schemas(self) -> None:
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

        Both dimensions are checked against the Arrow schema derived from the
        record at the configured width (:func:`.base.arrow.arrow_schema_for`) —
        the exact schema ``LanceDbConnection.table`` builds the table from, so
        a healthy table never false-positives:

        * **Column set** — a missing / extra column (e.g. a pre-``content_sha256``
          table) is caught by name.
        * **Column type** — a column whose on-disk Arrow type drifted from
          the current schema. This is the class of drift behind EverOS #337:
          an ``episode.subject_vector`` column left as ``string`` (or ``null``)
          by an older build, while the current schema declares a 1024-d
          ``fixed_size_list``. The name matches, so a name-only check waves it
          through and it detonates deep inside ``merge_insert`` as an opaque
          ``LanceError(IO): Spill has sent an error``. Comparing types surfaces
          it here instead. A vector column built at another width than
          ``[embedding] dim`` is the same kind of drift and is caught here too.
        """
        for schema in self._schemas:
            table = await self._open(schema)
            on_disk = await table.schema()
            expected = self._arrow_schema(schema)
            on_disk_names = set(on_disk.names)
            expected_names = set(expected.names)
            missing = expected_names - on_disk_names
            extra = on_disk_names - expected_names
            # Type drift on columns present in both, compared against the
            # authoritative derived Arrow types.
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

    async def drop_business_tables(self) -> list[str]:
        """Drop every business LanceDB table; return the names dropped.

        The tables are a rebuildable projection of markdown, so dropping is
        non-destructive to memory content — ``cascade rebuild`` recreates and
        re-populates them from md. Evicts the dropped tables from the manager
        cache so a later :meth:`LanceDbConnection.table` reopens the fresh table.
        """
        return await self._connection.drop_tables(
            [schema.TABLE_NAME for schema in self._schemas]
        )

    async def read_embedding_model(self) -> str | None:
        """Model this index's vectors were built with, or ``None`` if unattested.

        ``None`` is not an error: an index built before this field existed, or
        one just recreated by ``cascade rebuild``, simply has not been
        attested yet. The caller writes the current configuration in that
        case; see :meth:`write_embedding_model`.
        """
        table = await self._open(Episode)
        metadata = (await table.schema()).field(_PROVENANCE_COLUMN).metadata or {}
        raw = metadata.get(EMBEDDING_MODEL_KEY.encode())
        return raw.decode() if raw else None

    async def write_embedding_model(self, model: str) -> None:
        """Record ``model`` on the episode table's vector column, in place.

        Arrow field metadata is updated without rewriting data: the table
        gains a version, its row count and column types are untouched. The
        create path does not carry this (``arrow_schema_for`` builds fields
        with no metadata), so every attestation goes through here.
        """
        table = await self._open(Episode)
        await table.update_field_metadata(
            {
                "path": _PROVENANCE_COLUMN,
                "metadata": {EMBEDDING_MODEL_KEY: model},
            }
        )
        logger.info("lancedb_embedding_model_recorded", model=model)


_MIGRATIONS: dict[int, Callable[[LanceIndexBackend], Awaitable[None]]] = {
    2: LanceIndexBackend._migrate_v2_nullable_vector,
    3: LanceIndexBackend._migrate_v3_visual_vector,
    4: LanceIndexBackend._migrate_v4_group_columns,
    5: LanceIndexBackend._migrate_v5_group_owned,
}
"""One coroutine per target version, run in order by
:meth:`LanceIndexBackend.migrate_table_schemas`.

The entries are unbound methods, so the mapping is defined after the class.
A new schema change adds one entry here and bumps :data:`_TABLE_SCHEMA_VERSION`;
the dispatcher needs no edit, and a crash mid-sequence resumes at the version
the marker records rather than replaying the steps already done.
"""
