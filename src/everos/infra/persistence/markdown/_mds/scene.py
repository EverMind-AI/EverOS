"""Scene frontmatter -- daily-log markdown for a user's scene versions.

Path: ``users/<scope_id>/scenes/scene-<YYYY-MM-DD>.md`` (docs/scenes.md).

One entry per scene version; a superseded version is listed in
``deprecated_entries`` against the version that replaced it, a forgotten one
against the forget marker. Scene versions are rows of the episode table
(``parent_type = scene``), so they mint ``sc`` entry ids: an index row id is
``<owner_id>_<entry_id>`` and must not meet one of the owner's ``ep`` ids.
A group's scene versions use :class:`.group.GroupSceneDailyFrontmatter`.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar, Literal

from pydantic import Field

from everos.infra.persistence.markdown._base.frontmatter import (
    DailyLogPathMixin,
    UserScopedFrontmatter,
)


class SceneDailyFrontmatter(DailyLogPathMixin, UserScopedFrontmatter):
    """Frontmatter for ``users/<scope>/scenes/scene-<YYYY-MM-DD>.md``."""

    ENTRY_ID_PREFIX: ClassVar[str] = "sc"
    DIR_NAME: ClassVar[str] = "scenes"
    FILE_PREFIX: ClassVar[str] = "scene"

    type: Literal["scene_daily"] = "scene_daily"
    file_type: Literal["scene_daily"] = "scene_daily"
    date: _dt.date
    entry_count: int = 0
    created_at: _dt.datetime | None = None
    last_appended_at: _dt.datetime | None = None
    deprecated_entries: dict[str, str] = Field(default_factory=dict)
