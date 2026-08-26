"""Decision daily-log reader — symmetric with :class:`DecisionWriter`."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from everos.core.persistence import MemoryRoot

from ..mds import DecisionDailyFrontmatter
from .base import BaseDailyReader


class DecisionReader(BaseDailyReader):
    """Read decision daily-log files."""

    schema = DecisionDailyFrontmatter

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
        """Resolve the decision daily-log path under the <app>/<project> prefix."""
        return super().path_for(owner_id, date, app_id=app_id, project_id=project_id)
