"""Foresight frontmatter — daily-log markdown for user-scoped foresights.

Path: ``users/<scope_id>/.foresights/foresight-<YYYY-MM-DD>.md``.

The directory is dot-prefixed so it is hidden from end users (same
convention as ``.index``); foresights are framework-internal derived md,
not material the user is expected to read by hand.

Each entry carries a forward-looking inference about the user (intent
window, planned action, projected need) with ``start_time`` /
``end_time`` describing the covered time range. ``parent_type`` always
points back to a MemCell.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar, Literal

from everos.core.persistence.markdown import (
    DailyLogPathMixin,
    UserScopedFrontmatter,
)


class ForesightDailyFrontmatter(DailyLogPathMixin, UserScopedFrontmatter):
    """Frontmatter for ``users/<scope>/.foresights/foresight-<YYYY-MM-DD>.md``."""

    ENTRY_ID_PREFIX: ClassVar[str] = "fs"
    DIR_NAME: ClassVar[str] = ".foresights"
    FILE_PREFIX: ClassVar[str] = "foresight"

    type: Literal["foresight_daily"] = "foresight_daily"
    file_type: Literal["foresight_daily"] = "foresight_daily"
    date: _dt.date
    entry_count: int = 0
    created_at: _dt.datetime | None = None
    last_appended_at: _dt.datetime | None = None
