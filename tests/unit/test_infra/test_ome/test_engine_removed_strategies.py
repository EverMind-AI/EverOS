"""A run persisted for a strategy or an event this release removed is dropped.

EverOS 2.0 removed ``extract_atomic_facts`` / ``extract_foresight`` and the
``AtomicFactsExtracted`` / ``UserPipelineStarted`` events. A store an earlier
release ran can still hold a RUNNING run record (crash recovery re-enqueues
it) or a pending scheduler job naming them. ``dispatch_run`` must drop such a
run with a warning -- not raise out of the scheduler on every start -- and
release the in-flight count the enqueue path took.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import structlog.testing

from everos.infra.ome.config import OMEConfig
from everos.infra.ome.engine import OfflineEngine
from everos.memory.deps import MemoryDeps
from everos.memory.ports import Event, Immediate
from tests.helpers.ome import ome_strategy


class _E(Event):
    pass


@pytest.fixture
def cfg(tmp_path: Path) -> OMEConfig:
    return OMEConfig(jobstore_path=tmp_path / "ome.db", config_watch=False)


async def _engine(cfg: OMEConfig) -> tuple[OfflineEngine, list[_E]]:
    seen: list[_E] = []

    @ome_strategy(name="kept", trigger=Immediate(on=[_E]), emits=[])
    async def kept(event: _E, deps: MemoryDeps) -> list[Event]:
        seen.append(event)
        return []

    engine = OfflineEngine(config=cfg)
    engine.register(kept)
    await engine.start()
    return engine, seen


async def test_a_run_of_a_removed_strategy_is_dropped(cfg: OMEConfig) -> None:
    engine, seen = await _engine(cfg)
    try:
        engine._in_flight.enqueued()  # what the enqueue path did before
        with structlog.testing.capture_logs() as logs:
            await engine.dispatch_run(
                strategy_name="extract_atomic_facts",
                run_id="r_old",
                event_topic=_E.topic(),
                event_payload=_E().model_dump_json(),
                max_retries_snapshot=0,
            )
        assert [log["event"] for log in logs] == ["ome_run_dropped_unknown_strategy"]
        assert engine._in_flight.active == 0
        assert await engine.list_runs("extract_atomic_facts") == []
        assert seen == []
    finally:
        await engine.stop()


@pytest.mark.parametrize(
    "topic",
    [
        "everos.memory.events:AtomicFactsExtracted",
        "everos.memory.events:UserPipelineStarted",
        "everos.no_such_module:Event",
    ],
)
async def test_a_run_of_a_removed_event_is_dropped(cfg: OMEConfig, topic: str) -> None:
    engine, seen = await _engine(cfg)
    try:
        engine._in_flight.enqueued()
        with structlog.testing.capture_logs() as logs:
            await engine.dispatch_run(
                strategy_name="kept",
                run_id="r_old",
                event_topic=topic,
                event_payload="{}",
                max_retries_snapshot=0,
            )
        assert [log["event"] for log in logs] == ["ome_run_dropped_unknown_event"]
        assert engine._in_flight.active == 0
        assert await engine.list_runs("kept") == []
        assert seen == []
    finally:
        await engine.stop()


async def test_a_known_run_still_runs(cfg: OMEConfig) -> None:
    engine, seen = await _engine(cfg)
    try:
        engine._in_flight.enqueued()
        await engine.dispatch_run(
            strategy_name="kept",
            run_id="r_new",
            event_topic=_E.topic(),
            event_payload=_E().model_dump_json(),
            max_retries_snapshot=0,
        )
        assert len(seen) == 1
        assert engine._in_flight.active == 0
    finally:
        await engine.stop()
