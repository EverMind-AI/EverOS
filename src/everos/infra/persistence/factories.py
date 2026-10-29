"""Construction of the storage adapters behind the memory ports.

The application bootstrap builds its connections, stores and indexes here
and injects them into domain objects; nothing in the domain constructs an
adapter.
"""

from __future__ import annotations

__all__ = [
    "build_derived_index",
    "build_indexes",
    "build_stores",
]

from everos.config import Settings
from everos.memory.deps import Indexes, Stores

from .index import DerivedIndex
from .index.adapters import (
    AgentCaseIndexAdapter,
    AgentSkillIndexAdapter,
    AtomicFactIndexAdapter,
    EpisodeIndexAdapter,
    KnowledgeTopicIndexAdapter,
    ProfileIndexAdapter,
)
from .lancedb import LanceDbConnection, LanceIndexBackend
from .markdown import (
    MarkdownAgentCaseStore,
    MarkdownAgentSkillStore,
    MarkdownAtomicFactStore,
    MarkdownEpisodeStore,
    MarkdownForesightStore,
    MarkdownProfileStore,
)
from .memory_root import MemoryRoot
from .sqlite import (
    SqliteClusterStore,
    SqliteMemcellStore,
    SqliteMemorizeInboxStore,
    SqliteReflectionReportStore,
    SqliteRepos,
    SqliteSessionBufferStore,
)


def build_stores(root: MemoryRoot, sqlite: SqliteRepos) -> Stores:
    """Markdown truth under ``root`` plus the SQLite state behind ``sqlite``."""
    return Stores(
        episode=MarkdownEpisodeStore(root),
        atomic_fact=MarkdownAtomicFactStore(root),
        foresight=MarkdownForesightStore(root),
        profile=MarkdownProfileStore(root),
        agent_case=MarkdownAgentCaseStore(root),
        agent_skill=MarkdownAgentSkillStore(root),
        session_buffer=SqliteSessionBufferStore(
            sqlite.unprocessed_buffer, sqlite.conversation_status
        ),
        memcell=SqliteMemcellStore(sqlite.memcell),
        cluster=SqliteClusterStore(sqlite.cluster),
        memorize_inbox=SqliteMemorizeInboxStore(sqlite.memorize_inbox),
        reflection_report=SqliteReflectionReportStore(sqlite.reflection_report),
    )


def build_derived_index(settings: Settings, root: MemoryRoot) -> DerivedIndex:
    """The derived index of the engine ``settings.index.backend`` selects.

    Builds the engine's connection and backend; nothing connects until the
    index is first used. This is the one place the vector width
    (``[embedding] dim``) reaches the derived index: the backend builds every
    vector column at it.
    """
    if settings.index.backend == "milvus":
        # Deferred: pymilvus is an optional extra, needed only when selected.
        from .milvus import MilvusConnection, MilvusIndexBackend

        backend = MilvusIndexBackend(
            MilvusConnection(settings.milvus), dim=settings.embedding.dim
        )
    else:
        backend = LanceIndexBackend(
            LanceDbConnection(root, settings.lancedb), dim=settings.embedding.dim
        )
    return DerivedIndex(backend, embedding_dim=settings.embedding.dim)


def build_indexes(index: DerivedIndex) -> Indexes:
    """The domain's index readers over ``index``'s repositories."""
    repos = index.repositories
    return Indexes(
        episode=EpisodeIndexAdapter(repos.episode),
        atomic_fact=AtomicFactIndexAdapter(repos.atomic_fact),
        agent_case=AgentCaseIndexAdapter(repos.agent_case),
        agent_skill=AgentSkillIndexAdapter(repos.agent_skill),
        profile=ProfileIndexAdapter(repos.user_profile),
        knowledge_topic=KnowledgeTopicIndexAdapter(repos.knowledge_topic),
    )
