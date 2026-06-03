"""Markdown file IO toolkit.

Atomic write + YAML frontmatter parse/dump + entry marker parse +
audit-form structured-entry parsing. Knows nothing about business
models (no MemCell / Episode); the :class:`Entry` here is a
*marker-delimited* span within a markdown body, not a business record.

External usage (IO + parse):
    from everos.core.persistence.markdown import (
        Entry, EntryId, StructuredEntry,
        MarkdownReader, MarkdownWriter, ParsedMarkdown,
        parse_frontmatter, dump_frontmatter,
        split_entries, find_entry,
        parse_structured_entry, render_structured_entry,
    )

External usage (frontmatter schema chassis):
    from everos.core.persistence.markdown import (
        BaseFrontmatter, UserScopedFrontmatter, AgentScopedFrontmatter,
        DailyLogPathMixin, SkillPathMixin, ProfilePathMixin,
    )
"""

from .entries import Entry as Entry
from .entries import EntryId as EntryId
from .entries import StructuredEntry as StructuredEntry
from .entries import find_entry as find_entry
from .entries import parse_structured_entry as parse_structured_entry
from .entries import render_structured_entry as render_structured_entry
from .entries import split_entries as split_entries
from .frontmatter import AgentScopedFrontmatter as AgentScopedFrontmatter
from .frontmatter import BaseFrontmatter as BaseFrontmatter
from .frontmatter import DailyLogPathMixin as DailyLogPathMixin
from .frontmatter import ProfilePathMixin as ProfilePathMixin
from .frontmatter import SkillPathMixin as SkillPathMixin
from .frontmatter import UserScopedFrontmatter as UserScopedFrontmatter
from .frontmatter import dump_frontmatter as dump_frontmatter
from .frontmatter import parse_frontmatter as parse_frontmatter
from .parsed import ParsedMarkdown as ParsedMarkdown
from .reader import MarkdownReader as MarkdownReader
from .writer import MarkdownWriter as MarkdownWriter

__all__ = [
    "AgentScopedFrontmatter",
    "BaseFrontmatter",
    "DailyLogPathMixin",
    "Entry",
    "EntryId",
    "MarkdownReader",
    "MarkdownWriter",
    "ParsedMarkdown",
    "ProfilePathMixin",
    "SkillPathMixin",
    "StructuredEntry",
    "UserScopedFrontmatter",
    "dump_frontmatter",
    "find_entry",
    "parse_frontmatter",
    "parse_structured_entry",
    "render_structured_entry",
    "split_entries",
]
