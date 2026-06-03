"""Episode daily-log reader — symmetric with :class:`EpisodeWriter`.

md is the source of truth for Episode memories; this reader gives
cascade / search / verification scripts a typed locator instead of
raw :class:`MarkdownReader` calls.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from everos.core.persistence import MemoryRoot

from ..mds import EpisodeDailyFrontmatter
from .base import BaseDailyReader


class EpisodeReader(BaseDailyReader):
    """Read episode daily-log files."""

    schema = EpisodeDailyFrontmatter

    def __init__(self, root: MemoryRoot) -> None:
        super().__init__(root)

    def path_for(
        self,
        owner_id: str,
        date: _dt.date | None = None,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> Path:
        """Resolve the daily-log path for ``owner_id`` on ``date`` (today by default).

        Mirrors :meth:`EpisodeWriter`'s path-resolution shape so callers
        can locate the file written for a given owner / day (under the
        ``<app>/<project>`` prefix) without instantiating the writer.
        """
        return super().path_for(owner_id, date, app_id=app_id, project_id=project_id)
