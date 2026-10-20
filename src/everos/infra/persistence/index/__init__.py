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
from everos.core.errors import EmbeddingProvenanceError
from everos.infra.persistence import lancedb as _lancedb

from ._provenance import (
    last_probe as last_probe,
)
from ._provenance import (
    probe_embedding_dimension as probe_embedding_dimension,
)
from ._provenance import (
    verify_embedding_provenance as verify_embedding_provenance,
)
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


def assert_record_dim_matches_settings() -> None:
    """Fail loud if the record classes and ``Settings`` disagree on the width.

    The record classes resolve ``[embedding] dim`` with a narrow reader of
    their own, because the width has to be known while they are being
    imported (``Vector(dim)`` bakes it into the pydantic core schema) and
    building ``Settings`` on that path would make a malformed config break
    every CLI command. Two readers means they can diverge -- a different
    memory root, a key one of them cannot parse -- and the symptom would
    otherwise be tables quietly built at the wrong width.

    Exported rather than private because it has to run on paths that skip
    the rest of verification: ``cascade rebuild`` passes ``verify=False``,
    and it is the one command that unconditionally creates tables -- exactly
    where a disagreement would bake the wrong width into fresh tables.
    """
    baked = Episode.to_arrow_schema().field("vector").type.list_size
    configured = load_settings().embedding.dim
    if baked != configured:
        raise EmbeddingProvenanceError(
            f"the index record classes were built for dim={baked} but "
            f"configuration reports dim={configured}.\n"
            f"  The two readers disagree -- check EVEROS_ROOT and "
            f"<root>/everos.toml, then start again."
        )


async def verify_configuration(*, may_attest: bool = True) -> None:
    """Every "does this index match the configuration" check, in order.

    Exported as one call so the server and the CLI phases cannot run
    different subsets: ``startup`` uses it, and so does the CLI's own
    connect / verify / ensure sequence, which exists separately because
    ``cascade rebuild`` has to skip the middle step.

    Raises:
        EmbeddingProvenanceError: the record classes and ``Settings``
            disagree on the vector width, or the configured embedding
            cannot have produced the stored vectors.
        IndexSchemaMismatchError: a stored table's columns drifted from the
            current schema.
    """
    assert_record_dim_matches_settings()
    await _backend().verify_business_schemas()
    await verify_embedding_provenance(_backend(), may_attest=may_attest)


async def startup() -> Any:
    """Open the index and refuse to use one the configuration does not match.

    Order is load-bearing. Schema verification goes first because it is the
    most specific complaint (a column of the wrong type names both widths),
    and because it is also what creates missing tables -- so provenance runs
    against tables that exist. Index creation goes last: there is no point
    building indexes on a store that is about to be rejected.
    """
    check_primary_keys(repo.schema for repo in ALL_REPOS)
    connection = await connect()
    await verify_configuration()
    await ensure_business_indexes()
    return connection


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


async def count_business_rows() -> dict[str, int]:
    """Rows currently held per business table, keyed by table name.

    Backend-neutral: ``count`` is on the repository port, so this answers
    for Milvus as well as LanceDB. Cheap on LanceDB (dataset metadata) and
    one round trip per collection on Milvus -- which is why callers take it
    as a snapshot rather than on every request.

    The number it answers is "what the index actually holds", and the only
    thing that can contradict a cascade ledger claiming every file is
    indexed (issue #501).
    """
    return {repo.table_name: await repo.count() for repo in ALL_REPOS}


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
    "assert_record_dim_matches_settings",
    "atomic_fact_repo",
    "check_primary_keys",
    "compare",
    "connect",
    "contains",
    "count_business_rows",
    "drop_business_tables",
    "ensure_business_indexes",
    "episode_repo",
    "eq",
    "foresight_repo",
    "gt",
    "gte",
    "is_null",
    "knowledge_topic_repo",
    "last_probe",
    "lt",
    "lte",
    "ne",
    "one_of",
    "primary_key",
    "probe_embedding_dimension",
    "read_embedding_model",
    "repo_for_schema",
    "schema_for",
    "shutdown",
    "startup",
    "user_profile_repo",
    "verify_business_schemas",
    "verify_configuration",
    "verify_embedding_provenance",
    "write_embedding_model",
]
