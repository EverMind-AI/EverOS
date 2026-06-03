"""Episode daily-log writer — md is the SoT for Episode memories.

Stays in the chassis style: caller hands in pre-built ``inline`` and
``sections`` dicts plus the scope id (``owner_id``). Domain →
structured-entry shaping lives in the calling pipeline (cf. architecture
rule: ``infra`` may not import ``memory``).

This milestone assumes well-behaved callers (no retransmit dedupe needed).
The writer just appends; the chassis manages the in-file ``entry_id``
sequence, which is the single source of identity for an md entry.
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

from ..mds import EpisodeDailyFrontmatter
from .base import BaseDailyWriter


class EpisodeWriter(BaseDailyWriter):
    """Daily-log writer for the Episode schema (md = SoT).

    ``append_entry`` / ``append_entries`` come from
    :class:`BaseDailyWriter`; the ``entry_id`` (``ep_<YYYYMMDD>_<NNNN>``)
    is the in-file identity allocated under the per-path lock. Callers
    can derive a globally-unique id from ``(owner_id, entry_id)``
    without persisting any algo-side uuid.
    """

    schema = EpisodeDailyFrontmatter

    # ── Frontmatter override (entry_count + last_appended_at) ────────────

    def _frontmatter_updates(
        self,
        scope_id: str,
        date: _dt.date,
        *,
        next_count: int,
    ) -> Mapping[str, Any] | None:
        return {
            "id": f"episode_log_{scope_id}_{date.isoformat()}",
            "type": "episode_daily",
            "file_type": "episode_daily",
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
