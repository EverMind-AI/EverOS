"""Group frontmatter schemas: ``group.md`` and the group daily logs.

Paths (docs/group-memory.md, section 6.1)::

    groups/<group_id>/group.md                      meta + roster log
    groups/<group_id>/episodes/episode-<date>.md    one entry per segment
    groups/<group_id>/images/image-<date>.md        one entry per picture
    groups/<group_id>/scenes/scene-<date>.md        one entry per scene version
    groups/<group_id>/members/<member_id>.md        member-in-group profile

``group.md`` is a single file: its frontmatter is the group's meta, its body
the roster as append-only entries (``gr_<YYYYMMDD>_<seq>``, the date being the
event's own UTC date). The frontmatter also lists the sessions bound to the
group, so a rebuild of the system database from markdown restores the bindings
with the roster; the bindings are what keep a later batch of a group session
from running the one-to-one path.

The group daily logs mint ``gep`` / ``gim`` entry ids rather than the user
kinds' ``ep`` / ``im``: an index row id is ``<owner_id>_<entry_id>``, and a group
whose id equals a member's user id must not share row ids with that user's
memory in the same table. Scene versions (``gsc``) are rows of the episode
table too, so they need an id space of their own as well.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, ClassVar, Literal

from pydantic import Field

from everos.infra.persistence.markdown._base.frontmatter import (
    DailyLogPathMixin,
    GroupScopedFrontmatter,
    ProfilePathMixin,
)

_MEMBERS_DIR = "members"


class GroupFrontmatter(ProfilePathMixin, GroupScopedFrontmatter):
    """Frontmatter of ``groups/<group_id>/group.md``."""

    PROFILE_FILENAME: ClassVar[str] = "group.md"
    ENTRY_ID_PREFIX: ClassVar[str] = "gr"

    type: Literal["group"] = "group"
    name: str = ""
    description: str = ""
    created_at: str
    """ISO-8601, UTC."""
    created_by: str | None = None
    history_visible_from: Literal["join", "creation"] = "join"
    extra: dict[str, Any] = Field(default_factory=dict)
    sessions: list[dict[str, Any]] = Field(default_factory=list)
    """``[{session_id, bound_at}]`` in binding order; ``bound_at`` is ISO-8601."""
    entry_count: int = 0


class GroupEpisodeDailyFrontmatter(DailyLogPathMixin, GroupScopedFrontmatter):
    """Frontmatter for ``groups/<g>/episodes/episode-<YYYY-MM-DD>.md``."""

    ENTRY_ID_PREFIX: ClassVar[str] = "gep"
    DIR_NAME: ClassVar[str] = "episodes"
    FILE_PREFIX: ClassVar[str] = "episode"

    type: Literal["group_episode_daily"] = "group_episode_daily"
    file_type: Literal["group_episode_daily"] = "group_episode_daily"
    date: _dt.date
    entry_count: int = 0
    last_appended_at: _dt.datetime | None = None
    deprecated_entries: dict[str, str] = Field(default_factory=dict)


class GroupImageMemoryDailyFrontmatter(DailyLogPathMixin, GroupScopedFrontmatter):
    """Frontmatter for ``groups/<g>/images/image-<YYYY-MM-DD>.md``."""

    ENTRY_ID_PREFIX: ClassVar[str] = "gim"
    DIR_NAME: ClassVar[str] = "images"
    FILE_PREFIX: ClassVar[str] = "image"

    type: Literal["group_image_memory_daily"] = "group_image_memory_daily"
    file_type: Literal["group_image_memory_daily"] = "group_image_memory_daily"
    date: _dt.date
    entry_count: int = 0
    last_appended_at: _dt.datetime | None = None
    deprecated_entries: dict[str, str] = Field(default_factory=dict)


class GroupSceneDailyFrontmatter(DailyLogPathMixin, GroupScopedFrontmatter):
    """Frontmatter for ``groups/<g>/scenes/scene-<YYYY-MM-DD>.md``.

    One entry per scene version; a superseded version is listed in
    ``deprecated_entries`` against the version that replaced it.
    """

    ENTRY_ID_PREFIX: ClassVar[str] = "gsc"
    DIR_NAME: ClassVar[str] = "scenes"
    FILE_PREFIX: ClassVar[str] = "scene"

    type: Literal["group_scene_daily"] = "group_scene_daily"
    file_type: Literal["group_scene_daily"] = "group_scene_daily"
    date: _dt.date
    entry_count: int = 0
    last_appended_at: _dt.datetime | None = None
    deprecated_entries: dict[str, str] = Field(default_factory=dict)


class GroupMemberProfileFrontmatter(GroupScopedFrontmatter):
    """Frontmatter of ``groups/<g>/members/<member_id>.md``.

    The profile operator's document about one member, read from this group's
    segments only (docs/group-memory.md, section 7.4), in the user profile's
    shape plus the member and the provenance every item carries
    (``item_id`` / ``source_memcell_ids`` / ``source_start`` /
    ``source_end``). ``pending_memcell_ids`` are segments not read yet;
    ``history`` the items an update replaced or dropped. The body is the
    summary, for a human reader.
    """

    MEMBERS_DIR: ClassVar[str] = _MEMBERS_DIR

    type: Literal["group_member_profile"] = "group_member_profile"
    member_id: str
    summary: str = ""
    explicit_info: list[Any] = Field(default_factory=list)
    implicit_traits: list[Any] = Field(default_factory=list)
    profile_timestamp_ms: int = 0
    primary_language: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    pending_memcell_ids: list[str] = Field(default_factory=list)
    history: list[Any] = Field(default_factory=list)

    @classmethod
    def path_glob(cls) -> str:
        # Leading ``*/*/`` matches the <app>/<project> scope prefix.
        return f"*/*/{cls.SCOPE_DIR}/*/{cls.MEMBERS_DIR}/*.md"
