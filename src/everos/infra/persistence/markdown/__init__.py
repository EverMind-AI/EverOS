"""Markdown business persistence layer.

Sits on top of :mod:`._base` (atomic write +
parse + frontmatter chassis) and provides:

    * concrete frontmatter schemas under :mod:`._mds`
    * concrete business writers under :mod:`._writers`
      (``BaseDailyWriter`` + subclasses)
    * concrete business readers under :mod:`._readers`
      (``BaseDailyReader`` + subclasses)

External usage::

    from everos.infra.persistence.markdown import (
        BaseDailyWriter, BaseDailyReader,
        EpisodeWriter, EpisodeReader, EpisodeDailyFrontmatter,
    )

Outer layers MUST go through this top-level package because
``infra.persistence.markdown.**`` (sub-packages) are forbidden to outer
layers by import-linter.
"""

from ._base import BaseFrontmatter as BaseFrontmatter
from ._base import DailyLogPathMixin as DailyLogPathMixin
from ._base import Entry as Entry
from ._base import EntryId as EntryId
from ._base import GroupScopedFrontmatter as GroupScopedFrontmatter
from ._base import MarkdownReader as MarkdownReader
from ._base import MarkdownWriter as MarkdownWriter
from ._base import ParsedMarkdown as ParsedMarkdown
from ._base import ProfilePathMixin as ProfilePathMixin
from ._base import StructuredEntry as StructuredEntry
from ._base import UnreadableEntry as UnreadableEntry
from ._base import UserScopedFrontmatter as UserScopedFrontmatter
from ._base import dump_frontmatter as dump_frontmatter
from ._base import find_entry as find_entry
from ._base import parse_frontmatter as parse_frontmatter
from ._base import parse_structured_entry as parse_structured_entry
from ._base import render_structured_entry as render_structured_entry
from ._base import scan_entries as scan_entries
from ._base import split_entries as split_entries
from ._base import warn_unreadable_entry as warn_unreadable_entry
from ._mds import EpisodeDailyFrontmatter as EpisodeDailyFrontmatter
from ._mds import GroupEpisodeDailyFrontmatter as GroupEpisodeDailyFrontmatter
from ._mds import GroupFrontmatter as GroupFrontmatter
from ._mds import (
    GroupImageMemoryDailyFrontmatter as GroupImageMemoryDailyFrontmatter,
)
from ._mds import (
    GroupMemberProfileFrontmatter as GroupMemberProfileFrontmatter,
)
from ._mds import GroupProfileDailyFrontmatter as GroupProfileDailyFrontmatter
from ._mds import (
    GroupProfilePendingFrontmatter as GroupProfilePendingFrontmatter,
)
from ._mds import GroupSceneDailyFrontmatter as GroupSceneDailyFrontmatter
from ._mds import ImageMemoryDailyFrontmatter as ImageMemoryDailyFrontmatter
from ._mds import SceneDailyFrontmatter as SceneDailyFrontmatter
from ._mds import UserProfileFrontmatter as UserProfileFrontmatter
from ._mds import group_profile_item_from_entry as group_profile_item_from_entry
from ._mds import render_group_profile_entry as render_group_profile_entry
from ._readers import BaseDailyReader as BaseDailyReader
from ._readers import EpisodeReader as EpisodeReader
from ._writers import BaseDailyWriter as BaseDailyWriter
from ._writers import EpisodeWriter as EpisodeWriter
from ._writers import GroupProfileWriter as GroupProfileWriter
from ._writers import GroupWriter as GroupWriter
from ._writers import ImageMemoryWriter as ImageMemoryWriter
from ._writers import SceneWriter as SceneWriter
from .owner_dir import MarkdownOwnerDirStore as MarkdownOwnerDirStore
from .stores import MarkdownEpisodeStore as MarkdownEpisodeStore
from .stores import MarkdownGroupProfileStore as MarkdownGroupProfileStore
from .stores import MarkdownImageMemoryStore as MarkdownImageMemoryStore
from .stores import MarkdownProfileStore as MarkdownProfileStore
from .stores import MarkdownSceneStore as MarkdownSceneStore

__all__ = [
    "BaseDailyReader",
    "BaseDailyWriter",
    "BaseFrontmatter",
    "DailyLogPathMixin",
    "Entry",
    "EntryId",
    "EpisodeDailyFrontmatter",
    "EpisodeReader",
    "EpisodeWriter",
    "GroupEpisodeDailyFrontmatter",
    "GroupFrontmatter",
    "GroupImageMemoryDailyFrontmatter",
    "GroupMemberProfileFrontmatter",
    "GroupProfileDailyFrontmatter",
    "GroupProfilePendingFrontmatter",
    "GroupProfileWriter",
    "GroupSceneDailyFrontmatter",
    "GroupScopedFrontmatter",
    "GroupWriter",
    "ImageMemoryDailyFrontmatter",
    "ImageMemoryWriter",
    "MarkdownEpisodeStore",
    "MarkdownGroupProfileStore",
    "MarkdownImageMemoryStore",
    "MarkdownOwnerDirStore",
    "MarkdownProfileStore",
    "MarkdownReader",
    "MarkdownSceneStore",
    "MarkdownWriter",
    "ParsedMarkdown",
    "ProfilePathMixin",
    "SceneDailyFrontmatter",
    "SceneWriter",
    "StructuredEntry",
    "UnreadableEntry",
    "UserProfileFrontmatter",
    "UserScopedFrontmatter",
    "dump_frontmatter",
    "find_entry",
    "group_profile_item_from_entry",
    "parse_frontmatter",
    "parse_structured_entry",
    "render_group_profile_entry",
    "render_structured_entry",
    "scan_entries",
    "split_entries",
    "warn_unreadable_entry",
]
