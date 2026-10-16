"""Backend-neutral facade for the rebuildable BM25/vector index.

Markdown remains the source of truth and SQLite remains the system-state
store. This boundary owns only the derived business indexes used by cascade,
search, and get. LanceDB and Milvus implement the same typed ports, so callers
do not branch on physical storage.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from everos.config import load_settings
from everos.infra.persistence import lancedb as _lancedb

from .predicate import (
    All,
    AnyOf,
    Comparison,
    Contains,
    In,
    IsNull,
    Predicate,
    Scalar,
    all_of,
    any_of,
    compare,
    contains,
    eq,
    gt,
    gte,
    is_null,
    lt,
    lte,
    ne,
    one_of,
    primary_key,
)
from .protocols import (
    AgentSkillIndexRepository,
    EpisodeIndexRepository,
    IndexBackend,
    IndexRepository,
)
from .router import (
    RoutedAgentSkillRepository,
    RoutedEpisodeRepository,
    RoutedIndexRepository,
)
from .schema import IndexField, IndexFieldKind, IndexSchema, schema_for

_lance_episode_repo = _lancedb.episode_repo
_lance_atomic_fact_repo = _lancedb.atomic_fact_repo
_lance_foresight_repo = _lancedb.foresight_repo
_lance_agent_case_repo = _lancedb.agent_case_repo
_lance_agent_skill_repo = _lancedb.agent_skill_repo
_lance_user_profile_repo = _lancedb.user_profile_repo
_lance_knowledge_topic_repo = _lancedb.knowledge_topic_repo

AgentCase = _lancedb.AgentCase
AgentSkill = _lancedb.AgentSkill
AtomicFact = _lancedb.AtomicFact
Episode = _lancedb.Episode
Foresight = _lancedb.Foresight
KnowledgeTopic = _lancedb.KnowledgeTopic
ParentType = _lancedb.ParentType
UserProfile = _lancedb.UserProfile

episode_repo = RoutedEpisodeRepository(_lance_episode_repo, "episode_repo")
atomic_fact_repo = RoutedIndexRepository(_lance_atomic_fact_repo, "atomic_fact_repo")
foresight_repo = RoutedIndexRepository(_lance_foresight_repo, "foresight_repo")
agent_case_repo = RoutedIndexRepository(_lance_agent_case_repo, "agent_case_repo")
agent_skill_repo = RoutedAgentSkillRepository(
    _lance_agent_skill_repo, "agent_skill_repo"
)
user_profile_repo = RoutedIndexRepository(_lance_user_profile_repo, "user_profile_repo")
knowledge_topic_repo = RoutedIndexRepository(
    _lance_knowledge_topic_repo, "knowledge_topic_repo"
)

ALL_REPOS = (
    episode_repo,
    atomic_fact_repo,
    foresight_repo,
    agent_case_repo,
    agent_skill_repo,
    user_profile_repo,
    knowledge_topic_repo,
)


def active_backend() -> str:
    """Name of the configured derived-index backend."""
    return load_settings().index.backend


def _backend() -> IndexBackend:
    if active_backend() == "milvus":
        # pymilvus is an optional extra: import the engine only when selected.
        from everos.infra.persistence import milvus

        return milvus.index_backend
    return _lancedb.index_backend


async def connect() -> Any:
    return await _backend().connect()


def check_primary_keys(schemas: Iterable[type[Any]]) -> None:
    """Fail at startup, not at first write, when a table has no PRIMARY_KEY.

    For an undeclared table ``schema_for`` raises the same "has no
    PRIMARY_KEY" message the LanceDB write path raises, so it is caught by
    whichever runs first, on either backend; the write-path check stays as
    the last line of defence for repositories built outside this facade.
    """
    for schema in schemas:
        schema_for(schema)


async def startup() -> Any:
    check_primary_keys(repo.schema for repo in ALL_REPOS)
    return await _backend().startup()


async def shutdown() -> None:
    await _backend().shutdown()


async def ensure_business_indexes() -> None:
    check_primary_keys(repo.schema for repo in ALL_REPOS)
    await _backend().ensure_business_indexes()


async def verify_business_schemas() -> None:
    await _backend().verify_business_schemas()


async def drop_business_tables() -> list[str]:
    return await _backend().drop_business_tables()


async def read_embedding_model() -> str | None:
    """Model the active backend says this index was built with."""
    return await _backend().read_embedding_model()


async def write_embedding_model(model: str) -> None:
    """Attest ``model`` as the producer of this index's vectors."""
    await _backend().write_embedding_model(model)


def repo_for_schema(schema: type[Any]) -> IndexRepository[Any]:
    """Resolve a registered repository by its logical record model."""
    for repo in ALL_REPOS:
        if repo.schema is schema:
            return repo
    raise KeyError(f"no derived-index repository for {schema!r}")


__all__ = [
    "ALL_REPOS",
    "AgentCase",
    "AgentSkill",
    "AgentSkillIndexRepository",
    "All",
    "AnyOf",
    "AtomicFact",
    "Comparison",
    "Contains",
    "Episode",
    "EpisodeIndexRepository",
    "Foresight",
    "In",
    "IndexBackend",
    "IndexField",
    "IndexFieldKind",
    "IndexRepository",
    "IndexSchema",
    "IsNull",
    "KnowledgeTopic",
    "ParentType",
    "Predicate",
    "Scalar",
    "UserProfile",
    "active_backend",
    "agent_case_repo",
    "agent_skill_repo",
    "all_of",
    "any_of",
    "atomic_fact_repo",
    "check_primary_keys",
    "compare",
    "connect",
    "contains",
    "drop_business_tables",
    "ensure_business_indexes",
    "episode_repo",
    "eq",
    "foresight_repo",
    "gt",
    "gte",
    "is_null",
    "knowledge_topic_repo",
    "lt",
    "lte",
    "ne",
    "one_of",
    "primary_key",
    "read_embedding_model",
    "repo_for_schema",
    "schema_for",
    "shutdown",
    "startup",
    "user_profile_repo",
    "verify_business_schemas",
    "write_embedding_model",
]
