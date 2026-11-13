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
from ._repos import ForgetLogRepo as ForgetLogRepo
from ._repos import GroupMessageRepo as GroupMessageRepo
from ._repos import GroupRosterRepo as GroupRosterRepo
from ._repos import ImageOriginalRepo as ImageOriginalRepo
from ._repos import MdChangeStateRepo as MdChangeStateRepo
from ._repos import MemcellRepo as MemcellRepo
from ._repos import MemorizeInboxRepo as MemorizeInboxRepo
from ._repos import MessageForgetRepo as MessageForgetRepo
from ._repos import QueueSummary as QueueSummary
from ._repos import UnprocessedBufferRepo as UnprocessedBufferRepo
from ._tables import Cluster as Cluster
from ._tables import ClusterMember as ClusterMember
from ._tables import ConversationStatus as ConversationStatus
from ._tables import ForgetLog as ForgetLog
from ._tables import GroupMember as GroupMember
from ._tables import GroupMessage as GroupMessage
from ._tables import GroupMeta as GroupMeta
from ._tables import ImageOriginal as ImageOriginal
from ._tables import MdChangeState as MdChangeState
from ._tables import Memcell as Memcell
from ._tables import MemorizeInbox as MemorizeInbox
from ._tables import MessageForget as MessageForget
from ._tables import SessionBinding as SessionBinding
from ._tables import UnprocessedBuffer as UnprocessedBuffer
from .repositories import SqliteRepos as SqliteRepos
from .sqlite_manager import SqliteDatabase as SqliteDatabase
from .stores import SqliteClusterStore as SqliteClusterStore
from .stores import SqliteForgetLogStore as SqliteForgetLogStore
from .stores import SqliteGroupMessageStore as SqliteGroupMessageStore
from .stores import SqliteMemcellStore as SqliteMemcellStore
from .stores import SqliteMemorizeInboxStore as SqliteMemorizeInboxStore
from .stores import SqliteMessageForgetStore as SqliteMessageForgetStore
from .stores import SqliteSessionBufferStore as SqliteSessionBufferStore
from .stores import SqliteSessionPurgeStore as SqliteSessionPurgeStore

__all__ = [
    "MTIME_TOLERANCE_SECONDS",
    "BaseTable",
    "Cluster",
    "ClusterMember",
    "ClusterRepo",
    "ConversationStatus",
    "Field",
    "ForgetLog",
    "ForgetLogRepo",
    "GroupMember",
    "GroupMessage",
    "GroupMessageRepo",
    "GroupMeta",
    "GroupRosterRepo",
    "ImageOriginal",
    "ImageOriginalRepo",
    "MdChangeState",
    "MdChangeStateRepo",
    "Memcell",
    "MemcellRepo",
    "MemorizeInbox",
    "MemorizeInboxRepo",
    "MessageForget",
    "MessageForgetRepo",
    "QueueSummary",
    "Relationship",
    "RepoBase",
    "SQLModel",
    "SessionBinding",
    "SqliteClusterStore",
    "SqliteDatabase",
    "SqliteForgetLogStore",
    "SqliteGroupMessageStore",
    "SqliteMemcellStore",
    "SqliteMemorizeInboxStore",
    "SqliteMessageForgetStore",
    "SqliteRepos",
    "SqliteSessionBufferStore",
    "SqliteSessionPurgeStore",
    "UnprocessedBuffer",
    "UnprocessedBufferRepo",
    "create_session_factory",
    "create_system_engine",
    "session_scope",
]
