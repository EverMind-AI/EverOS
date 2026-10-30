"""OmeEventBus publishes through the engine, in order."""

from __future__ import annotations

from pathlib import Path

from everos.infra.ome import OmeEventBus
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.engine import OfflineEngine
from everos.memory.deps import MemoryDeps
from everos.memory.ports import Event, Immediate
from tests.helpers.ome import ome_strategy


class _Ping(Event):
    n: int


async def test_publish_dispatches_each_event_in_order(tmp_path: Path) -> None:
    seen: list[int] = []

    @ome_strategy(name="collect", trigger=Immediate(on=[_Ping]))
    async def collect(event: _Ping, deps: MemoryDeps) -> list[Event]:
        seen.append(event.n)
        return []

    engine = OfflineEngine(
        config=OMEConfig(jobstore_path=tmp_path / "ome.db", config_watch=False)
    )
    engine.register(collect)
    await engine.start()
    try:
        await OmeEventBus(engine).publish([_Ping(n=1), _Ping(n=2), _Ping(n=3)])
        assert await engine.wait_idle(timeout=5)
    finally:
        await engine.stop()

    assert sorted(seen) == [1, 2, 3]
