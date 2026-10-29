"""Agent memory pipeline — 5A in the add_v1 flow.

Consumes pre-cut cells from :mod:`everos.service._boundary` and returns
one :class:`AgentPipelineStarted` per cell so the OME
``extract_agent_case`` strategy picks it up.

No sqlite memcell row is written here — the boundary stage owns that
ledger (one row per cell, shared across user / agent paths via the same
``memcell_id``). No md is written here either: Episode md comes from
:class:`UserMemoryPipeline`, AgentCase md is the OME strategy's job.
"""

from __future__ import annotations

from everos.memory.events import AgentPipelineStarted
from everos.memory.extract.pipeline.base import CellOutcome, PreparedCell
from everos.memory.models import IngestResult


class AgentMemoryPipeline:
    """Announce each cell to the agent-side strategies."""

    async def run_cell(
        self, ingested: IngestResult, prepared: PreparedCell
    ) -> CellOutcome:
        """Return the cell's ``AgentPipelineStarted``."""
        return CellOutcome(
            status="extracted",
            message_count=len(prepared.cell.items),
            events=[
                AgentPipelineStarted(
                    memcell_id=prepared.memcell_id,
                    session_id=ingested.session_id,
                    app_id=ingested.app_id,
                    project_id=ingested.project_id,
                    memcell=prepared.cell,
                )
            ],
        )
