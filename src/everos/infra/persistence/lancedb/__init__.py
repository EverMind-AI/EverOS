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
    deterministic: delete the index directory and let it reindex.
    The lifespan surfaces the explicit ``rm -rf ~/.everos/.index/
    lancedb`` instruction in the error message; see
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
    match its current Pydantic schema.

    LanceDB doesn't migrate columns automatically; an older index dir
    (e.g. with the pre-``content_sha256`` shape) would fail
    unpredictably on upsert. Checking column names up-front turns that
    into a clean startup error pointing the user at the recovery path
    (``rm -rf ~/.everos/.index/lancedb`` — the index is rebuildable
    from md, see ``12_cascade_design.md``).
    """
    for schema in _BUSINESS_SCHEMAS:
        table = await get_table(schema.TABLE_NAME, schema)
        arrow_schema = await table.schema()
        actual = set(arrow_schema.names)
        expected = set(schema.model_fields.keys())
        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise LanceDBSchemaMismatchError(
                f"LanceDB table {schema.TABLE_NAME!r} schema drift: "
                f"missing={sorted(missing)}, extra={sorted(extra)}. "
                "The index is rebuildable from md — recover with "
                "`rm -rf ~/.everos/.index/lancedb` and restart."
            )


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
