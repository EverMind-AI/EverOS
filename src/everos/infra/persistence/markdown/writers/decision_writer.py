"""Decision daily-log writer — md is the SoT for decisions.

Caller hands pre-built ``inline`` (``owner_id`` / ``session_id`` /
``timestamp`` / ``parent_type`` / ``parent_id`` / ``tags``) plus the
``Title`` / ``Decision`` / ``Reason`` / optional ``Impact`` sections.
The chassis manages the in-file ``entry_id`` sequence
(``dc_<YYYYMMDD>_<NNNN>``). ``append_entry`` / ``append_entries`` come
from :class:`BaseDailyWriter`; this subclass only declares the schema
and the per-schema frontmatter / counter hooks.

Domain → ``(inline, sections)`` shaping lives in the calling strategy
(infra must not import ``memory``).
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

from ..mds import DecisionDailyFrontmatter
from .base import BaseDailyWriter


class DecisionWriter(BaseDailyWriter):
    """Daily-log writer for the Decision schema (md = SoT)."""

    schema = DecisionDailyFrontmatter

    def _frontmatter_updates(
        self,
        scope_id: str,
        date: _dt.date,
        *,
        next_count: int,
    ) -> Mapping[str, Any] | None:
        return {
            "id": f"decision_log_{scope_id}_{date.isoformat()}",
            "type": "decision_daily",
            "file_type": "decision_daily",
            "schema_version": 1,
            "user_id": scope_id,
            "track": "user",
            "date": date.isoformat(),
            "entry_count": next_count,
            "last_appended_at": to_iso_format(get_now_with_timezone()),
        }

    async def _current_count(self, path: Path) -> int:
        if not await anyio.Path(path).is_file():
            return 0
        parsed = await MarkdownReader.read(path)
        return parsed.frontmatter.get("entry_count", 0)
