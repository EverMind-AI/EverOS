"""AtomicFact daily-log reader — symmetric with :class:`AtomicFactWriter`."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from everos.core.persistence import MemoryRoot

from ..mds import AtomicFactDailyFrontmatter
from .base import BaseDailyReader


class AtomicFactReader(BaseDailyReader):
    """Read atomic-fact daily-log files."""

    schema = AtomicFactDailyFrontmatter

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
        """Resolve the atomic-fact daily-log path under the <app>/<project> prefix."""
        return super().path_for(owner_id, date, app_id=app_id, project_id=project_id)
