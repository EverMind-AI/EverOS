"""SQLite business persistence layer.

Sits on top of :mod:`.base` (engine + sessions +
``BaseTable`` + ``RepoBase``) and provides:

    * :class:`SqliteDatabase` -- the engine and session factory of one
      ``system.db`` (:mod:`.sqlite_manager`)
    * concrete table schemas under :mod:`.tables`
    * concrete repositories under :mod:`.repos`, and :class:`SqliteRepos`
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

# Importing ``tables`` registers every business SQLModel in
# ``SQLModel.metadata`` so the application start can
# ``create_all`` without callers having to import each model module.
from . import tables as tables
from .base import BaseTable as BaseTable
from .base import Field as Field
from .base import Relationship as Relationship
from .base import RepoBase as RepoBase
from .base import SQLModel as SQLModel
from .base import create_session_factory as create_session_factory
from .base import create_system_engine as create_system_engine
from .base import session_scope as session_scope
from .repos import MTIME_TOLERANCE_SECONDS as MTIME_TOLERANCE_SECONDS
from .repos import ClusterRepo as ClusterRepo
from .repos import ConversationStatusRepo as ConversationStatusRepo
from .repos import DocumentListPage as DocumentListPage
from .repos import DocumentUpsertPayload as DocumentUpsertPayload
from .repos import KnowledgeDocumentRepo as KnowledgeDocumentRepo
from .repos import KnowledgeTopicRowRepo as KnowledgeTopicRowRepo
from .repos import MdChangeStateRepo as MdChangeStateRepo
from .repos import MemcellRepo as MemcellRepo
from .repos import MemorizeInboxRepo as MemorizeInboxRepo
from .repos import QueueSummary as QueueSummary
from .repos import ReflectionReportRepo as ReflectionReportRepo
from .repos import TopicUpsertPayload as TopicUpsertPayload
from .repos import UnprocessedBufferRepo as UnprocessedBufferRepo
from .repositories import SqliteRepos as SqliteRepos
from .sqlite_manager import SqliteDatabase as SqliteDatabase
from .stores import SqliteClusterStore as SqliteClusterStore
from .stores import SqliteMemcellStore as SqliteMemcellStore
from .stores import SqliteMemorizeInboxStore as SqliteMemorizeInboxStore
from .stores import SqliteReflectionReportStore as SqliteReflectionReportStore
from .stores import SqliteSessionBufferStore as SqliteSessionBufferStore
from .tables import Cluster as Cluster
from .tables import ClusterMember as ClusterMember
from .tables import ConversationStatus as ConversationStatus
from .tables import KnowledgeDocumentRow as KnowledgeDocumentRow
from .tables import KnowledgeTopicRow as KnowledgeTopicRow
from .tables import MdChangeState as MdChangeState
from .tables import Memcell as Memcell
from .tables import MemorizeInbox as MemorizeInbox
from .tables import ReflectionReport as ReflectionReport
from .tables import UnprocessedBuffer as UnprocessedBuffer

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
