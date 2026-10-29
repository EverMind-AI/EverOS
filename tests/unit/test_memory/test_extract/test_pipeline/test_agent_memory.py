"""``AgentMemoryPipeline.run_cell`` — one ``AgentPipelineStarted`` per cell."""

from __future__ import annotations

from everalgo.types import ChatMessage, MemCell

from everos.memory import IngestResult
from everos.memory.events import AgentPipelineStarted
from everos.memory.extract.pipeline import (
    AgentMemoryPipeline,
    CellOutcome,
    PipelineOutcome,
    PreparedCell,
)


def _make_cell(n_items: int, ts: int = 1_700_000_000_000) -> MemCell:
    items = [
        ChatMessage(
            id=f"m{i}",
            role="user",
            sender_id="u1",
            sender_name="u",
            content="hi",
            timestamp=ts,
        )
        for i in range(n_items)
    ]
    return MemCell(items=items, timestamp=ts)


async def test_returns_one_event_per_cell() -> None:
    pipeline = AgentMemoryPipeline()
    ingested = IngestResult(session_id="s1", messages=[])
    cells = [
        PreparedCell(cell=_make_cell(n_items=2), memcell_id="mc_a"),
        PreparedCell(cell=_make_cell(n_items=3), memcell_id="mc_b"),
    ]

    outcomes = [await pipeline.run_cell(ingested, c) for c in cells]

    assert [o.status for o in outcomes] == ["extracted", "extracted"]
    assert [o.message_count for o in outcomes] == [2, 3]
    events = [e for o in outcomes for e in o.events]
    assert all(isinstance(e, AgentPipelineStarted) for e in events)
    assert [e.memcell_id for e in events] == ["mc_a", "mc_b"]
    assert all(e.session_id == "s1" for e in events)

    summary = PipelineOutcome.from_cells("agent_memory", outcomes)
    assert (summary.track, summary.status, summary.message_count) == (
        "agent_memory",
        "extracted",
        5,
    )


def test_summary_of_no_cells_is_accumulated() -> None:
    summary = PipelineOutcome.from_cells("agent_memory", [])
    assert (summary.status, summary.message_count) == ("accumulated", 0)


def test_summary_of_only_skipped_cells_is_skipped() -> None:
    summary = PipelineOutcome.from_cells(
        "user_memory", [CellOutcome(status="skipped"), CellOutcome(status="skipped")]
    )
    assert (summary.status, summary.message_count) == ("skipped", 0)
