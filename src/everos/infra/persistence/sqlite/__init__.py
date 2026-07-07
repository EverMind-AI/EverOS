"""SQLite business persistence layer.

Sits on top of :mod:`everos.core.persistence.sqlite` (engine + sessions +
``BaseTable`` + ``RepoBase``) and provides:

    * lazy process-wide engine + session-factory singletons
      (:mod:`.sqlite_manager`)
    * concrete table schemas under :mod:`.tables`
    * concrete repository singletons under :mod:`.repos`

External usage::

    from everos.infra.persistence.sqlite import (
        get_engine, get_session_factory, dispose_engine,
        # business tables / repos are re-exported here too —
        # callers MUST go through this top-level package because
        # ``infra.persistence.sqlite.**`` (sub-packages) are forbidden
        # to ``service`` / ``memory`` / ``entrypoints`` by import-linter.
        UnprocessedBuffer, Memcell, ConversationStatus,
        KnowledgeDocumentRow, KnowledgeTopicRow,
        unprocessed_buffer_repo, memcell_repo, conversation_status_repo,
        knowledge_document_repo, knowledge_topic_sqlite_repo,
    )

The :class:`SqliteLifespanProvider` runs ``SQLModel.metadata.create_all``
on app startup and ``dispose_engine`` on shutdown, so business code does
not need to manage either.
"""

# Importing ``tables`` registers every business SQLModel in
# ``SQLModel.metadata`` so ``SqliteLifespanProvider.startup`` can
# ``create_all`` without callers having to import each model module.
from . import tables as tables
from .repos import DocumentListPage as DocumentListPage
from .repos import DocumentUpsertPayload as DocumentUpsertPayload
from .repos import QueueSummary as QueueSummary
from .repos import TopicUpsertPayload as TopicUpsertPayload
from .repos import cluster_repo as cluster_repo
from .repos import conversation_status_repo as conversation_status_repo
from .repos import knowledge_document_repo as knowledge_document_repo
from .repos import knowledge_topic_sqlite_repo as knowledge_topic_sqlite_repo
from .repos import md_change_state_repo as md_change_state_repo
from .repos import memcell_repo as memcell_repo
from .repos import mint_cluster_id as mint_cluster_id
from .repos import reflection_report_repo as reflection_report_repo
from .repos import unprocessed_buffer_repo as unprocessed_buffer_repo
from .sqlite_manager import dispose_engine as dispose_engine
from .sqlite_manager import get_engine as get_engine
from .sqlite_manager import get_session_factory as get_session_factory
from .tables import Cluster as Cluster
from .tables import ClusterMember as ClusterMember
from .tables import ConversationStatus as ConversationStatus
from .tables import KnowledgeDocumentRow as KnowledgeDocumentRow
from .tables import KnowledgeTopicRow as KnowledgeTopicRow
from .tables import MdChangeState as MdChangeState
from .tables import Memcell as Memcell
from .tables import ReflectionReport as ReflectionReport
from .tables import UnprocessedBuffer as UnprocessedBuffer

__all__ = [
    "Cluster",
    "ClusterMember",
    "ConversationStatus",
    "DocumentListPage",
    "DocumentUpsertPayload",
    "KnowledgeDocumentRow",
    "KnowledgeTopicRow",
    "MdChangeState",
    "Memcell",
    "QueueSummary",
    "ReflectionReport",
    "TopicUpsertPayload",
    "UnprocessedBuffer",
    "cluster_repo",
    "conversation_status_repo",
    "dispose_engine",
    "get_engine",
    "get_session_factory",
    "knowledge_document_repo",
    "knowledge_topic_sqlite_repo",
    "md_change_state_repo",
    "memcell_repo",
    "mint_cluster_id",
    "reflection_report_repo",
    "unprocessed_buffer_repo",
]
