"""AtomicFact frontmatter — daily-log markdown for user-scoped atomic facts.

Path: ``users/<scope_id>/.atomic_facts/atomic_fact-<YYYY-MM-DD>.md``.

The directory is dot-prefixed so it is hidden from end users (same
convention as ``.index``); atomic facts are framework-internal derived md,
not material the user is expected to read by hand.

Each entry carries one atomic fact extracted by the algo layer; the fact
always hangs off the source MemCell (see ``parent_type`` in each entry's
inline fields — handled at the StructuredEntry layer, not on the
file-level frontmatter).
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar, Literal

from everos.core.persistence.markdown import (
    DailyLogPathMixin,
    UserScopedFrontmatter,
)


class AtomicFactDailyFrontmatter(DailyLogPathMixin, UserScopedFrontmatter):
    """Frontmatter for ``users/<scope>/.atomic_facts/atomic_fact-<YYYY-MM-DD>.md``."""

    ENTRY_ID_PREFIX: ClassVar[str] = "af"
    DIR_NAME: ClassVar[str] = ".atomic_facts"
    FILE_PREFIX: ClassVar[str] = "atomic_fact"

    type: Literal["atomic_fact_daily"] = "atomic_fact_daily"
    file_type: Literal["atomic_fact_daily"] = "atomic_fact_daily"
    date: _dt.date
    entry_count: int = 0
    created_at: _dt.datetime | None = None
    last_appended_at: _dt.datetime | None = None
