"""The Runner consults its gate after recording the attempt, before the body,
and runs the body under the gate's binding (the event's time as the origin)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from everos.core.purge_marks import MARKS, PurgeScope, Tombstone
from everos.infra.ome._dispatch.runner import Runner
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.records import RunStatus
from everos.memory.deps import MemoryDeps
from everos.memory.ports import Event, Immediate
from everos.service.purge import PurgeRunGate
from tests.helpers.ome import ome_strategy

T = datetime(2026, 11, 11, 10, 0, tzinfo=UTC)
S = PurgeScope("project", "a1", "p1", "a1", "p1")


class _Scoped(Event):
    app_id: str = "a1"
    project_id: str = "p1"
    tag: str = ""


async def _no_emit(_event: Event) -> None:
    return None


@pytest.fixture
async def store(tmp_path: Path) -> RunRecordStore:
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    return RunRecordStore(storage=storage, max_records_per_strategy=1000)


@pytest.fixture(autouse=True)
def _marks() -> Iterator[None]:
    MARKS.reset()
    yield
    MARKS.reset()


def _runner(
    store: RunRecordStore, gate: object, *, emit_hook: object = _no_emit
) -> Runner:
    return Runner(
        run_record_store=store,
        engine_sem=asyncio.Semaphore(20),
        emit_hook=emit_hook,  # type: ignore[arg-type]
        config=OMEConfig(jobstore_path=Path("unused")),
        gate=gate,  # type: ignore[arg-type]
    )


def _body(calls: list[str]):  # type: ignore[no-untyped-def]
    @ome_strategy(name="s", trigger=Immediate(on=[_Scoped]), emits=[])
    async def s(event: _Scoped, deps: MemoryDeps) -> list[Event]:
        calls.append(event.tag)
        return []

    return s


async def test_discard_marks_success_and_clears_payload(
    store: RunRecordStore, tmp_path: Path
) -> None:
    marks = MARKS.for_root(tmp_path)
    marks.add_tombstone(Tombstone(S, T))
    calls: list[str] = []
    runner = _runner(store, PurgeRunGate(marks))
    stale = _Scoped(ts=T - timedelta(minutes=1), tag="stale")
    await runner.run(_body(calls), stale, run_id="r1", max_retries_snapshot=0)
    fresh = _Scoped(ts=T + timedelta(minutes=1), tag="fresh")
    await runner.run(_body(calls), fresh, run_id="r2", max_retries_snapshot=0)
    assert calls == ["fresh"]
    record = await store.get("r1")
    assert record is not None
    assert (record.status, record.event_payload) == (RunStatus.SUCCESS, "")


async def test_events_a_run_makes_carry_the_trigger_time(
    store: RunRecordStore, tmp_path: Path
) -> None:
    """A follow-up event inherits its trigger's time, so a purge confirmed
    after the trigger arrived drops the whole chain, not just its head."""
    emitted: list[Event] = []

    async def capture(event: Event) -> None:
        emitted.append(event)

    @ome_strategy(name="parent", trigger=Immediate(on=[_Scoped]), emits=[_Scoped])
    async def parent(event: _Scoped, deps: MemoryDeps) -> list[Event]:
        return [_Scoped(tag="child")]

    runner = _runner(store, PurgeRunGate(MARKS.for_root(tmp_path)), emit_hook=capture)
    trigger = _Scoped(ts=T, tag="parent")
    await runner.run(parent, trigger, run_id="r1", max_retries_snapshot=0)
    assert [(e.tag, e.ts) for e in emitted] == [("child", T)]  # type: ignore[attr-defined]


@pytest.mark.parametrize("entry", ["cron", "idle"])
async def test_a_tick_is_new_input_whatever_context_fires_it(entry: str) -> None:
    """APScheduler may fire a tick in a strategy task's context: the tick
    must not inherit that run's origin (its work on new data would be
    dropped by a purge confirmed in between)."""
    from everos.core.purge_marks import ORIGIN
    from everos.infra.ome._dispatch import scheduling

    made: list[Event] = []

    class _Engine:
        async def emit(self, event: Event) -> None:
            made.append(event)

        async def run_idle_scan(self, strategy_name: str) -> None:
            made.append(Event())

    scheduling.ENGINES["e1"] = _Engine()  # type: ignore[assignment]
    token = ORIGIN.set(T)
    try:
        fire = scheduling.cron_entry if entry == "cron" else scheduling.idle_entry
        await fire("e1", "s")
    finally:
        ORIGIN.reset(token)
        scheduling.ENGINES.pop("e1", None)
    assert len(made) == 1 and made[0].ts > T
