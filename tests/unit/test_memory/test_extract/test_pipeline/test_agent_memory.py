"""``AgentMemoryPipeline.run`` — empty short-circuit + per-cell event emit."""

from __future__ import annotations

from everalgo.types import ChatMessage, MemCell

from everos.memory import IngestResult
from everos.memory.events import AgentPipelineStarted
from everos.memory.extract.pipeline.agent_memory import AgentMemoryPipeline


class _FakeEngine:
    """Captures emitted events; mirrors ``OfflineEngine.emit`` async signature."""

    def __init__(self) -> None:
        self.events: list[AgentPipelineStarted] = []

    async def emit(self, event: AgentPipelineStarted) -> None:
        self.events.append(event)


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


async def test_empty_cells_short_circuit() -> None:
    engine = _FakeEngine()
    pipeline = AgentMemoryPipeline(engine)  # type: ignore[arg-type]
    ingested = IngestResult(session_id="s1", messages=[])
    out = await pipeline.run(ingested, cells=[], memcell_ids=[])
    assert out.track == "agent_memory"
    assert out.status == "accumulated"
    assert out.message_count == 0
    assert engine.events == []


async def test_emits_one_event_per_cell() -> None:
    engine = _FakeEngine()
    pipeline = AgentMemoryPipeline(engine)  # type: ignore[arg-type]
    ingested = IngestResult(session_id="s1", messages=[])
    cells = [_make_cell(n_items=2), _make_cell(n_items=3)]
    memcell_ids = ["mc_a", "mc_b"]
    out = await pipeline.run(ingested, cells=cells, memcell_ids=memcell_ids)

    assert out.track == "agent_memory"
    assert out.status == "extracted"
    assert out.message_count == 5  # 2 + 3
    assert [e.memcell_id for e in engine.events] == ["mc_a", "mc_b"]
    assert all(e.session_id == "s1" for e in engine.events)
    assert all(isinstance(e, AgentPipelineStarted) for e in engine.events)
