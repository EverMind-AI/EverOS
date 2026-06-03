"""AgentCase daily-log reader — symmetric with :class:`AgentCaseWriter`."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from everos.core.persistence import MemoryRoot

from ..mds import AgentCaseDailyFrontmatter
from .base import BaseDailyReader


class AgentCaseReader(BaseDailyReader):
    """Read agent-case daily-log files."""

    schema = AgentCaseDailyFrontmatter

    def __init__(self, root: MemoryRoot) -> None:
        super().__init__(root)

    def path_for(
        self,
        agent_id: str,
        date: _dt.date | None = None,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> Path:
        """Resolve the agent-case daily-log path under the <app>/<project> prefix."""
        return super().path_for(agent_id, date, app_id=app_id, project_id=project_id)
