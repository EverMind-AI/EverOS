"""Image-memory daily-log writer — md is the SoT for picture memories.

Same chassis as :class:`EpisodeWriter`: the caller hands in pre-built
``inline`` and ``sections`` dicts plus the owner id, and this class only binds
the schema and the frontmatter updates. Domain → structured-entry shaping
lives in the store (``infra`` may not import ``memory``).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from everos.core.utils.datetime import (
    get_now_with_timezone,
    to_iso_format,
)

from .._mds import ImageMemoryDailyFrontmatter
from .base import BaseDailyWriter


class ImageMemoryWriter(BaseDailyWriter):
    """Daily-log writer for the image-memory schema (md = SoT).

    ``entry_id`` (``im_<YYYYMMDD>_<NNNN>``) is the in-file identity allocated
    under the per-path lock; the globally-unique row key is
    ``<owner_id>_<entry_id>``, derived by the store and the cascade alike.
    """

    schema = ImageMemoryDailyFrontmatter

    def _frontmatter_updates(
        self,
        scope_id: str,
        date: _dt.date,
        *,
        next_count: int,
    ) -> Mapping[str, Any] | None:
        return {
            "id": f"image_log_{scope_id}_{date.isoformat()}",
            "type": "image_memory_daily",
            "file_type": "image_memory_daily",
            "schema_version": 1,
            "user_id": scope_id,
            "track": "user",
            "date": date.isoformat(),
            "entry_count": next_count,
            "last_appended_at": to_iso_format(get_now_with_timezone()),
        }

    async def _current_count(self, path: Path) -> int:
        return await self._count_from_frontmatter(path)
