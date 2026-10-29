"""Every business repository of one system database, built together."""

from __future__ import annotations

__all__ = ["SqliteRepos"]

from dataclasses import dataclass

from .repos import (
    ClusterRepo,
    ConversationStatusRepo,
    KnowledgeDocumentRepo,
    KnowledgeTopicRowRepo,
    MdChangeStateRepo,
    MemcellRepo,
    MemorizeInboxRepo,
    ReflectionReportRepo,
    UnprocessedBufferRepo,
)
from .sqlite_manager import SqliteDatabase


@dataclass(frozen=True)
class SqliteRepos:
    """The repositories of every business table in one ``system.db``."""

    cluster: ClusterRepo
    conversation_status: ConversationStatusRepo
    knowledge_document: KnowledgeDocumentRepo
    knowledge_topic: KnowledgeTopicRowRepo
    md_change_state: MdChangeStateRepo
    memcell: MemcellRepo
    memorize_inbox: MemorizeInboxRepo
    reflection_report: ReflectionReportRepo
    unprocessed_buffer: UnprocessedBufferRepo

    @classmethod
    def over(cls, database: SqliteDatabase) -> SqliteRepos:
        """Build every repository over ``database``'s session factory."""
        factory = database.session_factory
        return cls(
            cluster=ClusterRepo(factory),
            conversation_status=ConversationStatusRepo(factory),
            knowledge_document=KnowledgeDocumentRepo(factory),
            knowledge_topic=KnowledgeTopicRowRepo(factory),
            md_change_state=MdChangeStateRepo(factory),
            memcell=MemcellRepo(factory),
            memorize_inbox=MemorizeInboxRepo(factory),
            reflection_report=ReflectionReportRepo(factory),
            unprocessed_buffer=UnprocessedBufferRepo(factory),
        )
