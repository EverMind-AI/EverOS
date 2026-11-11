"""Scene daily-log writer -- md is the SoT for a user's scene versions.

Same chassis as :class:`EpisodeWriter`: the caller hands in pre-built
``inline`` and ``sections`` dicts plus the owner id, and this class only binds
the schema and the frontmatter updates (docs/scenes.md).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from everos.core.utils.datetime import get_now_with_timezone, to_iso_format

from .._mds import SceneDailyFrontmatter
from .base import BaseDailyWriter


class SceneWriter(BaseDailyWriter):
    """Daily-log writer for a user's scene versions (``sc_<YYYYMMDD>_<seq>``)."""

    schema = SceneDailyFrontmatter

    def _frontmatter_updates(
        self,
        scope_id: str,
        date: _dt.date,
        *,
        next_count: int,
    ) -> Mapping[str, Any] | None:
        return {
            "id": f"scene_log_{scope_id}_{date.isoformat()}",
            "type": "scene_daily",
            "file_type": "scene_daily",
            "schema_version": 1,
            "user_id": scope_id,
            "track": "user",
            "date": date.isoformat(),
            "entry_count": next_count,
            "last_appended_at": to_iso_format(get_now_with_timezone()),
        }

    async def _current_count(self, path: Path) -> int:
        return await self._count_from_frontmatter(path)
