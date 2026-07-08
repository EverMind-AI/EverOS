"""Persistence primitives.

Read/write toolkit for markdown files, async wrappers around the SQLite
system DB and LanceDB index, plus a memory-root path manager. Higher
layers (``memory``, ``infra``) layer business semantics on top of these
building blocks; this subpackage knows nothing about Entry / MemCell /
Episode or any other business model.

External usage:
    from everos.core.persistence import (
        # Path manager + lock
        MemoryRoot, memory_root_lock, LockError,
        # Markdown IO toolkit
        MarkdownReader, MarkdownWriter, ParsedMarkdown, Entry,
        parse_frontmatter, dump_frontmatter, split_entries, find_entry,
        # Frontmatter schema chassis
        BaseFrontmatter, UserScopedFrontmatter, AgentScopedFrontmatter,
        DailyLogPathMixin, SkillPathMixin,
        # Async SQLite (SQLModel / SA 2.0)
        create_system_engine, create_session_factory, session_scope,
        SQLModel, Field, Relationship, BaseTable, RepoBase,
        # Async LanceDB
        open_lancedb_connection, LanceModel, Vector, BaseLanceTable, touch,
        LanceRepoBase,
    )
"""

from .lancedb import BaseLanceTable as BaseLanceTable
from .lancedb import LanceModel as LanceModel
from .lancedb import LanceRepoBase as LanceRepoBase
from .lancedb import Vector as Vector
from .lancedb import open_lancedb_connection as open_lancedb_connection
from .lancedb import touch as touch
from .locking import LockError as LockError
from .locking import memory_root_lock as memory_root_lock
from .markdown import AgentScopedFrontmatter as AgentScopedFrontmatter
from .markdown import BaseFrontmatter as BaseFrontmatter
from .markdown import DailyLogPathMixin as DailyLogPathMixin
from .markdown import Entry as Entry
from .markdown import EntryId as EntryId
from .markdown import MarkdownReader as MarkdownReader
from .markdown import MarkdownWriter as MarkdownWriter
from .markdown import ParsedMarkdown as ParsedMarkdown
from .markdown import SkillPathMixin as SkillPathMixin
from .markdown import StructuredEntry as StructuredEntry
from .markdown import UserScopedFrontmatter as UserScopedFrontmatter
from .markdown import dump_frontmatter as dump_frontmatter
from .markdown import find_entry as find_entry
from .markdown import parse_frontmatter as parse_frontmatter
from .markdown import parse_structured_entry as parse_structured_entry
from .markdown import render_structured_entry as render_structured_entry
from .markdown import split_entries as split_entries
from .memory_root import MemoryRoot as MemoryRoot
from .memory_root import app_dir_name as app_dir_name
from .memory_root import app_id_from_dir as app_id_from_dir
from .memory_root import project_dir_name as project_dir_name
from .memory_root import project_id_from_dir as project_id_from_dir
from .sqlite import BaseTable as BaseTable
from .sqlite import Field as Field
from .sqlite import Relationship as Relationship
from .sqlite import RepoBase as RepoBase
from .sqlite import SQLModel as SQLModel
from .sqlite import create_session_factory as create_session_factory
from .sqlite import create_system_engine as create_system_engine
from .sqlite import session_scope as session_scope

__all__ = [
    "AgentScopedFrontmatter",
    "BaseFrontmatter",
    "BaseLanceTable",
    "BaseTable",
    "DailyLogPathMixin",
    "Entry",
    "EntryId",
    "Field",
    "LanceModel",
    "LanceRepoBase",
    "LockError",
    "MarkdownReader",
    "MarkdownWriter",
    "MemoryRoot",
    "ParsedMarkdown",
    "Relationship",
    "RepoBase",
    "SQLModel",
    "SkillPathMixin",
    "StructuredEntry",
    "UserScopedFrontmatter",
    "Vector",
    "app_dir_name",
    "app_id_from_dir",
    "create_session_factory",
    "create_system_engine",
    "dump_frontmatter",
    "find_entry",
    "memory_root_lock",
    "open_lancedb_connection",
    "parse_frontmatter",
    "parse_structured_entry",
    "project_dir_name",
    "project_id_from_dir",
    "render_structured_entry",
    "session_scope",
    "split_entries",
    "touch",
]
