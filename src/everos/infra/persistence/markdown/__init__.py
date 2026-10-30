"""Markdown business persistence layer.

Sits on top of :mod:`._base` (atomic write +
parse + frontmatter chassis) and provides:

    * concrete frontmatter schemas under :mod:`._mds`
    * concrete business writers under :mod:`._writers`
      (``BaseDailyWriter`` + subclasses, ``AgentSkillWriter``,
      ``ProfileWriter``)
    * concrete business readers under :mod:`._readers`
      (``BaseDailyReader`` + subclasses, ``AgentSkillReader``,
      ``ProfileReader``)

External usage::

    from everos.infra.persistence.markdown import (
        BaseDailyWriter, BaseDailyReader,
        EpisodeWriter, EpisodeReader, EpisodeDailyFrontmatter,
        AtomicFactDailyFrontmatter,
        ForesightDailyFrontmatter,
        AgentCaseDailyFrontmatter,
        AgentSkillFrontmatter, AgentSkillWriter, AgentSkillReader,
        ProfileWriter, ProfileReader,
    )

Outer layers MUST go through this top-level package because
``infra.persistence.markdown.**`` (sub-packages) are forbidden to outer
layers by import-linter.
"""

from ._base import AgentScopedFrontmatter as AgentScopedFrontmatter
from ._base import BaseFrontmatter as BaseFrontmatter
from ._base import DailyLogPathMixin as DailyLogPathMixin
from ._base import Entry as Entry
from ._base import EntryId as EntryId
from ._base import KnowledgeDocumentPathMixin as KnowledgeDocumentPathMixin
from ._base import KnowledgeScopedMixin as KnowledgeScopedMixin
from ._base import KnowledgeTopicPathMixin as KnowledgeTopicPathMixin
from ._base import MarkdownReader as MarkdownReader
from ._base import MarkdownWriter as MarkdownWriter
from ._base import ParsedMarkdown as ParsedMarkdown
from ._base import ProfilePathMixin as ProfilePathMixin
from ._base import SkillPathMixin as SkillPathMixin
from ._base import StructuredEntry as StructuredEntry
from ._base import UserScopedFrontmatter as UserScopedFrontmatter
from ._base import dump_frontmatter as dump_frontmatter
from ._base import find_entry as find_entry
from ._base import parse_frontmatter as parse_frontmatter
from ._base import parse_structured_entry as parse_structured_entry
from ._base import render_structured_entry as render_structured_entry
from ._base import sanitize_dirname as sanitize_dirname
from ._base import split_entries as split_entries
from ._mds import AgentCaseDailyFrontmatter as AgentCaseDailyFrontmatter
from ._mds import AgentSkillFrontmatter as AgentSkillFrontmatter
from ._mds import AtomicFactDailyFrontmatter as AtomicFactDailyFrontmatter
from ._mds import EpisodeDailyFrontmatter as EpisodeDailyFrontmatter
from ._mds import ForesightDailyFrontmatter as ForesightDailyFrontmatter
from ._mds import KnowledgeDocumentFrontmatter as KnowledgeDocumentFrontmatter
from ._mds import KnowledgeTopicFrontmatter as KnowledgeTopicFrontmatter
from ._mds import UserProfileFrontmatter as UserProfileFrontmatter
from ._readers import AgentCaseReader as AgentCaseReader
from ._readers import AgentSkillReader as AgentSkillReader
from ._readers import AtomicFactReader as AtomicFactReader
from ._readers import BaseDailyReader as BaseDailyReader
from ._readers import EpisodeReader as EpisodeReader
from ._readers import ForesightReader as ForesightReader
from ._readers import ProfileReader as ProfileReader
from ._readers import ensure_taxonomy as ensure_taxonomy
from ._readers import parse_taxonomy as parse_taxonomy
from ._writers import AgentCaseWriter as AgentCaseWriter
from ._writers import AgentSkillWriter as AgentSkillWriter
from ._writers import AtomicFactWriter as AtomicFactWriter
from ._writers import BaseDailyWriter as BaseDailyWriter
from ._writers import EpisodeWriter as EpisodeWriter
from ._writers import ForesightWriter as ForesightWriter
from ._writers import KnowledgeWriter as KnowledgeWriter
from ._writers import ProfileWriter as ProfileWriter
from .knowledge_store import MarkdownKnowledgeStore as MarkdownKnowledgeStore
from .stores import MarkdownAgentCaseStore as MarkdownAgentCaseStore
from .stores import MarkdownAgentSkillStore as MarkdownAgentSkillStore
from .stores import MarkdownAtomicFactStore as MarkdownAtomicFactStore
from .stores import MarkdownEpisodeStore as MarkdownEpisodeStore
from .stores import MarkdownForesightStore as MarkdownForesightStore
from .stores import MarkdownProfileStore as MarkdownProfileStore

__all__ = [
    "AgentCaseDailyFrontmatter",
    "AgentCaseReader",
    "AgentCaseWriter",
    "AgentScopedFrontmatter",
    "AgentSkillFrontmatter",
    "AgentSkillReader",
    "AgentSkillWriter",
    "AtomicFactDailyFrontmatter",
    "AtomicFactReader",
    "AtomicFactWriter",
    "BaseDailyReader",
    "BaseDailyWriter",
    "BaseFrontmatter",
    "DailyLogPathMixin",
    "Entry",
    "EntryId",
    "EpisodeDailyFrontmatter",
    "EpisodeReader",
    "EpisodeWriter",
    "ForesightDailyFrontmatter",
    "ForesightReader",
    "ForesightWriter",
    "KnowledgeDocumentFrontmatter",
    "KnowledgeDocumentPathMixin",
    "KnowledgeScopedMixin",
    "KnowledgeTopicFrontmatter",
    "KnowledgeTopicPathMixin",
    "KnowledgeWriter",
    "MarkdownAgentCaseStore",
    "MarkdownAgentSkillStore",
    "MarkdownAtomicFactStore",
    "MarkdownEpisodeStore",
    "MarkdownForesightStore",
    "MarkdownKnowledgeStore",
    "MarkdownProfileStore",
    "MarkdownReader",
    "MarkdownWriter",
    "ParsedMarkdown",
    "ProfilePathMixin",
    "ProfileReader",
    "ProfileWriter",
    "SkillPathMixin",
    "StructuredEntry",
    "UserProfileFrontmatter",
    "UserScopedFrontmatter",
    "dump_frontmatter",
    "ensure_taxonomy",
    "find_entry",
    "parse_frontmatter",
    "parse_structured_entry",
    "parse_taxonomy",
    "render_structured_entry",
    "sanitize_dirname",
    "split_entries",
]
