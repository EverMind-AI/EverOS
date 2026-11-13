"""Every business repository of one system database, built together."""

from __future__ import annotations

__all__ = ["SqliteRepos"]

from dataclasses import dataclass

from ._repos import (
    ClusterRepo,
    KnowledgeDocumentRepo,
    KnowledgeTopicRowRepo,
    MdChangeStateRepo,
    MemcellRepo,
    MemorizeInboxRepo,
    PurgeJobRepo,
    ReflectionReportRepo,
    ScopePurgeRepo,
    SessionCutRepo,
    SessionPurgeRepo,
    UnprocessedBufferRepo,
)
from .sqlite_manager import SqliteDatabase


@dataclass(frozen=True)
class SqliteRepos:
    """The repositories of every business table in one ``system.db``."""

    cluster: ClusterRepo
    knowledge_document: KnowledgeDocumentRepo
    knowledge_topic: KnowledgeTopicRowRepo
    md_change_state: MdChangeStateRepo
    memcell: MemcellRepo
    memorize_inbox: MemorizeInboxRepo
    purge_job: PurgeJobRepo
    reflection_report: ReflectionReportRepo
    scope_purge: ScopePurgeRepo
    session_cut: SessionCutRepo
    session_purge: SessionPurgeRepo
    unprocessed_buffer: UnprocessedBufferRepo

    @classmethod
    def over(cls, database: SqliteDatabase) -> SqliteRepos:
        """Build every repository over ``database``'s session factory."""
        factory = database.session_factory
        return cls(
            cluster=ClusterRepo(factory),
            knowledge_document=KnowledgeDocumentRepo(factory),
            knowledge_topic=KnowledgeTopicRowRepo(factory),
            md_change_state=MdChangeStateRepo(factory),
            memcell=MemcellRepo(factory),
            memorize_inbox=MemorizeInboxRepo(factory),
            purge_job=PurgeJobRepo(factory),
            reflection_report=ReflectionReportRepo(factory),
            scope_purge=ScopePurgeRepo(factory),
            session_cut=SessionCutRepo(factory),
            session_purge=SessionPurgeRepo(factory),
            unprocessed_buffer=UnprocessedBufferRepo(factory),
        )
