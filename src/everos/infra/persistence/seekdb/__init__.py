"""SeekDB derived-index backend selected through ``Settings.index.backend``.

The package exposes the same seven repository singletons as LanceDB and
Milvus. pyseekdb itself stays lazily imported by the connection manager, so
default installations can import EverOS without the optional dependency.
"""

from __future__ import annotations

from everos.core.observability.logging import get_logger

from .errors import SeekdbConfigurationError as SeekdbConfigurationError
from .errors import SeekdbIntegrityError as SeekdbIntegrityError
from .errors import SeekdbOperationalError as SeekdbOperationalError
from .errors import SeekdbSchemaMismatchError as SeekdbSchemaMismatchError
from .errors import SeekdbValueLimitError as SeekdbValueLimitError
from .repos import ALL_REPOS as ALL_REPOS
from .repos import agent_case_repo as agent_case_repo
from .repos import agent_skill_repo as agent_skill_repo
from .repos import atomic_fact_repo as atomic_fact_repo
from .repos import episode_repo as episode_repo
from .repos import foresight_repo as foresight_repo
from .repos import knowledge_topic_repo as knowledge_topic_repo
from .repos import user_profile_repo as user_profile_repo
from .repository import SeekdbRepoBase as SeekdbRepoBase
from .seekdb_manager import dispose_connection as _dispose_connection
from .seekdb_manager import get_session as get_session
from .seekdb_manager import run as _run
from .seekdb_manager import table_name as table_name
from .sql import quote_identifier as _quote_identifier

logger = get_logger(__name__)


async def ensure_business_indexes() -> None:
    """Create or verify every configured SeekDB business table."""
    for repo in ALL_REPOS:
        await repo.ensure_table()


async def drop_business_tables() -> list[str]:
    """Drop every configured SeekDB business table and return physical names."""
    session = await get_session()
    dropped: list[str] = []
    for repo in ALL_REPOS:
        if not await repo.table_exists():
            continue
        name = repo.physical_table_name
        await _run(
            session.execute,
            f"DROP TABLE IF EXISTS {_quote_identifier(name)}",
            table=name,
        )
        dropped.append(name)
        logger.info("seekdb_table_dropped", table=repo.table_name, physical_table=name)
    SeekdbRepoBase._reset_table_cache()
    return dropped


async def dispose_connection() -> None:
    """Close SeekDB and clear per-process table readiness state."""
    try:
        await _dispose_connection()
    finally:
        SeekdbRepoBase._reset_table_cache()


__all__ = [
    "ALL_REPOS",
    "SeekdbConfigurationError",
    "SeekdbIntegrityError",
    "SeekdbOperationalError",
    "SeekdbRepoBase",
    "SeekdbSchemaMismatchError",
    "SeekdbValueLimitError",
    "agent_case_repo",
    "agent_skill_repo",
    "atomic_fact_repo",
    "dispose_connection",
    "drop_business_tables",
    "ensure_business_indexes",
    "episode_repo",
    "foresight_repo",
    "get_session",
    "knowledge_topic_repo",
    "table_name",
    "user_profile_repo",
]
