"""Foresight daily-log writer — md is the SoT for foresights.

Inline carries the audit / scope + time-window fields (``owner_id`` /
``session_id`` / ``timestamp`` / ``parent_id`` / ``sender_ids`` plus
optional ``start_time`` / ``end_time`` / ``duration_days``). Sections
carry the BM25-indexed content: ``Foresight`` (required, primary
field) and optional ``Evidence`` (secondary BM25 field).
``append_entry`` / ``append_entries`` come from :class:`BaseDailyWriter`.
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

from ..mds import ForesightDailyFrontmatter
from .base import BaseDailyWriter


class ForesightWriter(BaseDailyWriter):
    """Daily-log writer for the Foresight schema (md = SoT)."""

    schema = ForesightDailyFrontmatter

    def _frontmatter_updates(
        self,
        scope_id: str,
        date: _dt.date,
        *,
        next_count: int,
    ) -> Mapping[str, Any] | None:
        return {
            "id": f"foresight_log_{scope_id}_{date.isoformat()}",
            "type": "foresight_daily",
            "file_type": "foresight_daily",
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
