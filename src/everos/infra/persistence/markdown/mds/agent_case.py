"""AgentCase frontmatter — daily-log markdown for agent-scoped cases.

Path: ``agents/<scope_id>/.cases/agent_case-<YYYY-MM-DD>.md``.

The directory is dotfile-hidden (``.cases``) so users only see the
curated ``agent_skills/`` view, not the raw per-task case log — same
convention as ``.atomic_facts`` / ``.foresights``.

Each entry records one task an agent worked on: intent, approach taken,
quality score, and an optional pivotal insight. A MemCell extracted on
the agent's own execution log yields at most one AgentCase.
"""

from __future__ import annotations

import datetime as _dt
from typing import ClassVar, Literal

from everos.core.persistence.markdown import (
    AgentScopedFrontmatter,
    DailyLogPathMixin,
)


class AgentCaseDailyFrontmatter(DailyLogPathMixin, AgentScopedFrontmatter):
    """Frontmatter for ``agents/<scope>/.cases/agent_case-<YYYY-MM-DD>.md``."""

    ENTRY_ID_PREFIX: ClassVar[str] = "ac"
    DIR_NAME: ClassVar[str] = ".cases"
    FILE_PREFIX: ClassVar[str] = "agent_case"

    type: Literal["agent_case_daily"] = "agent_case_daily"
    file_type: Literal["agent_case_daily"] = "agent_case_daily"
    date: _dt.date
    entry_count: int = 0
    created_at: _dt.datetime | None = None
    last_appended_at: _dt.datetime | None = None
