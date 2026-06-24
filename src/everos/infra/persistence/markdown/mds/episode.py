"""Episode frontmatter — daily-log markdown for user-scoped episodes.

Path: ``users/<scope_id>/episodes/episode-<YYYY-MM-DD>.md``.

This milestone uses ``session_id`` as the scope key (since owner inference
is out of scope). When owner inference lands the scope key will switch to
``owner_id`` while the schema stays compatible.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar, Literal

from pydantic import Field

from everos.core.persistence.markdown import (
    DailyLogPathMixin,
    UserScopedFrontmatter,
)


class EpisodeDailyFrontmatter(DailyLogPathMixin, UserScopedFrontmatter):
    """Frontmatter for ``users/<scope>/episodes/episode-<YYYY-MM-DD>.md``."""

    ENTRY_ID_PREFIX: ClassVar[str] = "ep"
    DIR_NAME: ClassVar[str] = "episodes"
    FILE_PREFIX: ClassVar[str] = "episode"

    type: Literal["episode_daily"] = "episode_daily"
    file_type: Literal["episode_daily"] = "episode_daily"
    date: _dt.date
    entry_count: int = 0
    created_at: _dt.datetime | None = None
    last_appended_at: _dt.datetime | None = None
    deprecated_entries: dict[str, str] = Field(default_factory=dict)
