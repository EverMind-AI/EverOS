"""Markdown business persistence layer.

Sits on top of :mod:`.base` (atomic write +
parse + frontmatter chassis) and provides:

    * concrete frontmatter schemas under :mod:`.mds`
    * concrete business writers under :mod:`.writers`
      (``BaseDailyWriter`` + subclasses, ``AgentSkillWriter``,
      ``ProfileWriter``)
    * concrete business readers under :mod:`.readers`
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

from .base import AgentScopedFrontmatter as AgentScopedFrontmatter
from .base import BaseFrontmatter as BaseFrontmatter
from .base import DailyLogPathMixin as DailyLogPathMixin
from .base import Entry as Entry
from .base import EntryId as EntryId
from .base import KnowledgeDocumentPathMixin as KnowledgeDocumentPathMixin
from .base import KnowledgeScopedMixin as KnowledgeScopedMixin
from .base import KnowledgeTopicPathMixin as KnowledgeTopicPathMixin
from .base import MarkdownReader as MarkdownReader
from .base import MarkdownWriter as MarkdownWriter
from .base import ParsedMarkdown as ParsedMarkdown
from .base import ProfilePathMixin as ProfilePathMixin
from .base import SkillPathMixin as SkillPathMixin
from .base import StructuredEntry as StructuredEntry
from .base import UserScopedFrontmatter as UserScopedFrontmatter
from .base import dump_frontmatter as dump_frontmatter
from .base import find_entry as find_entry
from .base import parse_frontmatter as parse_frontmatter
from .base import parse_structured_entry as parse_structured_entry
from .base import render_structured_entry as render_structured_entry
from .base import sanitize_dirname as sanitize_dirname
from .base import split_entries as split_entries
from .knowledge_store import MarkdownKnowledgeStore as MarkdownKnowledgeStore
from .mds import AgentCaseDailyFrontmatter as AgentCaseDailyFrontmatter
from .mds import AgentSkillFrontmatter as AgentSkillFrontmatter
from .mds import AtomicFactDailyFrontmatter as AtomicFactDailyFrontmatter
from .mds import EpisodeDailyFrontmatter as EpisodeDailyFrontmatter
from .mds import ForesightDailyFrontmatter as ForesightDailyFrontmatter
from .mds import KnowledgeDocumentFrontmatter as KnowledgeDocumentFrontmatter
from .mds import KnowledgeTopicFrontmatter as KnowledgeTopicFrontmatter
from .mds import UserProfileFrontmatter as UserProfileFrontmatter
from .readers import AgentCaseReader as AgentCaseReader
from .readers import AgentSkillReader as AgentSkillReader
from .readers import AtomicFactReader as AtomicFactReader
from .readers import BaseDailyReader as BaseDailyReader
from .readers import EpisodeReader as EpisodeReader
from .readers import ForesightReader as ForesightReader
from .readers import ProfileReader as ProfileReader
from .readers import ensure_taxonomy as ensure_taxonomy
from .readers import parse_taxonomy as parse_taxonomy
from .stores import MarkdownAgentCaseStore as MarkdownAgentCaseStore
from .stores import MarkdownAgentSkillStore as MarkdownAgentSkillStore
from .stores import MarkdownAtomicFactStore as MarkdownAtomicFactStore
from .stores import MarkdownEpisodeStore as MarkdownEpisodeStore
from .stores import MarkdownForesightStore as MarkdownForesightStore
from .stores import MarkdownProfileStore as MarkdownProfileStore
from .writers import AgentCaseWriter as AgentCaseWriter
from .writers import AgentSkillWriter as AgentSkillWriter
from .writers import AtomicFactWriter as AtomicFactWriter
from .writers import BaseDailyWriter as BaseDailyWriter
from .writers import EpisodeWriter as EpisodeWriter
from .writers import ForesightWriter as ForesightWriter
from .writers import KnowledgeWriter as KnowledgeWriter
from .writers import ProfileWriter as ProfileWriter

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
