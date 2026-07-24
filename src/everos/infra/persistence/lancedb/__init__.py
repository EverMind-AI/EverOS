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
``12_cascade_design.md``).
"""

import contextlib
import datetime as dt

from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot

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
    """
    logger = get_logger(__name__)
    marker = MemoryRoot.default().lancedb_dir / ".fts_index_version"
    try:
        current = int(marker.read_text().strip()) if marker.exists() else 0
    except (ValueError, OSError):
        current = 0
    if current >= _FTS_INDEX_SCHEMA_VERSION:
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


async def ensure_business_indexes() -> None:
    """Ensure FTS (BM25) indexes for every business table (idempotent).

    Called once at startup by :class:`LanceDBLifespanProvider`. First
    runs :func:`migrate_fts_indexes` (one-time, marker-guarded) to
    rebuild any pre-fix ``with_position=True`` indexes, then walks the
    business schemas (each owns its ``TABLE_NAME`` + ``BM25_FIELDS``),
    opens each table via :func:`get_table`, and delegates to
    ``schema.ensure_fts_indexes(table)``. Already-indexed columns are
    skipped, so re-runs are no-ops.

    Adding a new business table = adding it to ``_BUSINESS_SCHEMAS``;
    everything else (table name, columns to index) reads off the
    schema's ClassVars.
    """
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
    recovery path (``rm -rf ~/.everos/.index/lancedb`` — the index is
    rebuildable from md, see ``12_cascade_design.md``).

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
                f"type_drift={type_drift}. "
                "The index is rebuildable from md — recover with "
                "`everos cascade rebuild` (drops + re-indexes from md, "
                "preserving un-extracted buffered messages)."
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
    "AgentCase",
    "AgentSkill",
    "AtomicFact",
    "Episode",
    "Foresight",
    "KnowledgeTopic",
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
    "user_profile_repo",
    "verify_business_schemas",
]
