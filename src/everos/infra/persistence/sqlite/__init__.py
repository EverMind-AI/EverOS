"""SQLite business persistence layer.

Sits on top of :mod:`._base` (engine + sessions +
``BaseTable`` + ``RepoBase``) and provides:

    * :class:`SqliteDatabase` -- the engine and session factory of one
      ``system.db`` (:mod:`.sqlite_manager`)
    * concrete table schemas under :mod:`._tables`
    * concrete repositories under :mod:`._repos`, and :class:`SqliteRepos`
      to build all of them over one database

External usage::

    from everos.infra.persistence.sqlite import SqliteDatabase, SqliteRepos

    database = SqliteDatabase(root, settings.sqlite)
    repos = SqliteRepos.over(database)

Callers MUST go through this top-level package: ``infra.persistence.sqlite.**``
(sub-packages) are forbidden to ``service`` / ``memory`` / ``entrypoints`` by
import-linter. The application start creates the schema and disposes the
database on shutdown.
"""

# Importing ``_tables`` registers every business SQLModel in
# ``SQLModel.metadata`` so the application start can
# ``create_all`` without callers having to import each model module.
from . import _tables as _tables
from ._base import BaseTable as BaseTable
from ._base import Field as Field
from ._base import Relationship as Relationship
from ._base import RepoBase as RepoBase
from ._base import SQLModel as SQLModel
from ._base import create_session_factory as create_session_factory
from ._base import create_system_engine as create_system_engine
from ._base import session_scope as session_scope
from ._repos import MTIME_TOLERANCE_SECONDS as MTIME_TOLERANCE_SECONDS
from ._repos import ClusterRepo as ClusterRepo
from ._repos import ConversationStatusRepo as ConversationStatusRepo
from ._repos import DocumentListPage as DocumentListPage
from ._repos import DocumentUpsertPayload as DocumentUpsertPayload
from ._repos import KnowledgeDocumentRepo as KnowledgeDocumentRepo
from ._repos import KnowledgeTopicRowRepo as KnowledgeTopicRowRepo
from ._repos import MdChangeStateRepo as MdChangeStateRepo
from ._repos import MemcellRepo as MemcellRepo
from ._repos import MemorizeInboxRepo as MemorizeInboxRepo
from ._repos import QueueSummary as QueueSummary
from ._repos import ReflectionReportRepo as ReflectionReportRepo
from ._repos import TopicUpsertPayload as TopicUpsertPayload
from ._repos import UnprocessedBufferRepo as UnprocessedBufferRepo
from ._tables import Cluster as Cluster
from ._tables import ClusterMember as ClusterMember
from ._tables import ConversationStatus as ConversationStatus
from ._tables import KnowledgeDocumentRow as KnowledgeDocumentRow
from ._tables import KnowledgeTopicRow as KnowledgeTopicRow
from ._tables import MdChangeState as MdChangeState
from ._tables import Memcell as Memcell
from ._tables import MemorizeInbox as MemorizeInbox
from ._tables import ReflectionReport as ReflectionReport
from ._tables import UnprocessedBuffer as UnprocessedBuffer
from .knowledge_catalog import SqliteKnowledgeCatalog as SqliteKnowledgeCatalog
from .repositories import SqliteRepos as SqliteRepos
from .sqlite_manager import SqliteDatabase as SqliteDatabase
from .stores import SqliteClusterStore as SqliteClusterStore
from .stores import SqliteMemcellStore as SqliteMemcellStore
from .stores import SqliteMemorizeInboxStore as SqliteMemorizeInboxStore
from .stores import SqliteReflectionReportStore as SqliteReflectionReportStore
from .stores import SqliteSessionBufferStore as SqliteSessionBufferStore

__all__ = [
    "MTIME_TOLERANCE_SECONDS",
    "BaseTable",
    "Cluster",
    "ClusterMember",
    "ClusterRepo",
    "ConversationStatus",
    "ConversationStatusRepo",
    "DocumentListPage",
    "DocumentUpsertPayload",
    "Field",
    "KnowledgeDocumentRepo",
    "KnowledgeDocumentRow",
    "KnowledgeTopicRow",
    "KnowledgeTopicRowRepo",
    "MdChangeState",
    "MdChangeStateRepo",
    "Memcell",
    "MemcellRepo",
    "MemorizeInbox",
    "MemorizeInboxRepo",
    "QueueSummary",
    "ReflectionReport",
    "ReflectionReportRepo",
    "Relationship",
    "RepoBase",
    "SQLModel",
    "SqliteClusterStore",
    "SqliteDatabase",
    "SqliteKnowledgeCatalog",
    "SqliteMemcellStore",
    "SqliteMemorizeInboxStore",
    "SqliteReflectionReportStore",
    "SqliteRepos",
    "SqliteSessionBufferStore",
    "TopicUpsertPayload",
    "UnprocessedBuffer",
    "UnprocessedBufferRepo",
    "create_session_factory",
    "create_system_engine",
    "session_scope",
]
