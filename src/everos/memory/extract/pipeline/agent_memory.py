"""Agent memory pipeline — 5A in the add_v1 flow.

Consumes pre-cut cells from :mod:`everos.service._boundary` and emits
:class:`AgentPipelineStarted` per cell so the OME ``extract_agent_cases``
strategy (separate work item) picks it up.

No sqlite memcell row is written here — the boundary stage owns that
ledger (one row per cell, shared across user / agent paths via the same
``memcell_id``). No md is written here either: Episode md comes from
:class:`UserMemoryPipeline`, AgentCase md is the OME strategy's job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everos.core.observability.logging import get_logger
from everos.memory import IngestResult, PipelineOutcome
from everos.memory.events import AgentPipelineStarted

if TYPE_CHECKING:
    from everalgo.types import MemCell as AlgoMemCell

    from everos.infra.ome.engine import OfflineEngine

logger = get_logger(__name__)

_TRACK = "agent_memory"


class AgentMemoryPipeline:
    """Emit ``AgentPipelineStarted`` per cell — fire-and-forget to OME."""

    def __init__(self, engine: OfflineEngine) -> None:
        self._engine = engine

    async def run(
        self,
        ingested: IngestResult,
        cells: list[AlgoMemCell],
        memcell_ids: list[str],
    ) -> PipelineOutcome:
        """Emit ``AgentPipelineStarted`` per cell."""
        if not cells:
            return PipelineOutcome(track=_TRACK, status="accumulated", message_count=0)

        for cell, memcell_id in zip(cells, memcell_ids, strict=True):
            await self._engine.emit(
                AgentPipelineStarted(
                    memcell_id=memcell_id,
                    session_id=ingested.session_id,
                    app_id=ingested.app_id,
                    project_id=ingested.project_id,
                    memcell=cell,
                )
            )

        return PipelineOutcome(
            track=_TRACK,
            status="extracted",
            message_count=sum(len(c.items) for c in cells),
        )
