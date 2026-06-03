"""AgentCase daily-log writer — md is the SoT for agent cases.

Lives on the agent track (``agents/<agent_id>/.cases/...``).
Inline carries audit + scoring fields (``owner_id`` / ``session_id`` /
``timestamp`` / ``parent_id`` / ``quality_score``); sections carry
``TaskIntent`` (required, primary BM25/embed), ``Approach`` (verbatim,
not indexed — too long), and optional ``KeyInsight`` (verbatim).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anyio

from everos.component.utils.datetime import (
    get_now_with_timezone,
    to_iso_format,
)
from everos.core.persistence import MarkdownReader

from ..mds import AgentCaseDailyFrontmatter
from .base import BaseDailyWriter


class AgentCaseWriter(BaseDailyWriter):
    """Daily-log writer for the AgentCase schema (md = SoT).

    ``append_entry`` / ``append_entries`` come from
    :class:`BaseDailyWriter`. The scope id parameter is named ``agent_id``
    semantically (this is the agent track), but the base class accepts
    it via the generic ``scope_id`` parameter.
    """

    schema = AgentCaseDailyFrontmatter

    def _frontmatter_updates(
        self,
        scope_id: str,
        date: _dt.date,
        *,
        next_count: int,
    ) -> Mapping[str, Any] | None:
        return {
            "id": f"agent_case_log_{scope_id}_{date.isoformat()}",
            "type": "agent_case_daily",
            "file_type": "agent_case_daily",
            "schema_version": 1,
            "agent_id": scope_id,
            "track": "agent",
            "date": date.isoformat(),
            "entry_count": next_count,
            "last_appended_at": to_iso_format(get_now_with_timezone()),
        }

    async def _current_count(self, path: Path) -> int:
        if not await anyio.Path(path).is_file():
            return 0
        parsed = await MarkdownReader.read(path)
        return parsed.frontmatter.get("entry_count", 0)
