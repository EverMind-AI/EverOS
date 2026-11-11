"""Every business repository of one system database, built together."""

from __future__ import annotations

__all__ = ["SqliteRepos"]

from dataclasses import dataclass

from ._repos import (
    ClusterRepo,
    ForgetLogRepo,
    GroupRosterRepo,
    ImageOriginalRepo,
    KnowledgeDocumentRepo,
    KnowledgeTopicRowRepo,
    MdChangeStateRepo,
    MemcellRepo,
    MemorizeInboxRepo,
    MessageForgetRepo,
    ReflectionReportRepo,
    SessionCutRepo,
    SessionPurgeRepo,
    UnprocessedBufferRepo,
)
from .sqlite_manager import SqliteDatabase


@dataclass(frozen=True)
class SqliteRepos:
    """The repositories of every business table in one ``system.db``."""

    cluster: ClusterRepo
    forget_log: ForgetLogRepo
    group_roster: GroupRosterRepo
    image_original: ImageOriginalRepo
    knowledge_document: KnowledgeDocumentRepo
    knowledge_topic: KnowledgeTopicRowRepo
    md_change_state: MdChangeStateRepo
    memcell: MemcellRepo
    memorize_inbox: MemorizeInboxRepo
    message_forget: MessageForgetRepo
    reflection_report: ReflectionReportRepo
    session_cut: SessionCutRepo
    session_purge: SessionPurgeRepo
    unprocessed_buffer: UnprocessedBufferRepo

    @classmethod
    def over(cls, database: SqliteDatabase) -> SqliteRepos:
        """Build every repository over ``database``'s session factory."""
        factory = database.session_factory
        return cls(
            cluster=ClusterRepo(factory),
            forget_log=ForgetLogRepo(factory),
            group_roster=GroupRosterRepo(factory),
            image_original=ImageOriginalRepo(factory),
            knowledge_document=KnowledgeDocumentRepo(factory),
            knowledge_topic=KnowledgeTopicRowRepo(factory),
            md_change_state=MdChangeStateRepo(factory),
            memcell=MemcellRepo(factory),
            memorize_inbox=MemorizeInboxRepo(factory),
            message_forget=MessageForgetRepo(factory),
            reflection_report=ReflectionReportRepo(factory),
            session_cut=SessionCutRepo(factory),
            session_purge=SessionPurgeRepo(factory),
            unprocessed_buffer=UnprocessedBufferRepo(factory),
        )
