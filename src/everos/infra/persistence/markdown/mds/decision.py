"""Decision frontmatter — daily-log markdown for user-scoped decisions.

Path: ``users/<scope_id>/decisions/decision-<YYYY-MM-DD>.md``.

User-readable (same convention as episodes), not a dot-prefixed internal
directory. ``deprecated_entries`` is present from day one so Decision
reflection can soft-deprecate a superseded entry without rewriting history.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar, Literal

from pydantic import Field

from everos.core.persistence.markdown import (
    DailyLogPathMixin,
    UserScopedFrontmatter,
)


class DecisionDailyFrontmatter(DailyLogPathMixin, UserScopedFrontmatter):
    """Frontmatter for ``users/<scope>/decisions/decision-<YYYY-MM-DD>.md``."""

    ENTRY_ID_PREFIX: ClassVar[str] = "dc"
    DIR_NAME: ClassVar[str] = "decisions"
    FILE_PREFIX: ClassVar[str] = "decision"

    type: Literal["decision_daily"] = "decision_daily"
    file_type: Literal["decision_daily"] = "decision_daily"
    date: _dt.date
    entry_count: int = 0
    created_at: _dt.datetime | None = None
    last_appended_at: _dt.datetime | None = None
    deprecated_entries: dict[str, str] = Field(default_factory=dict)
