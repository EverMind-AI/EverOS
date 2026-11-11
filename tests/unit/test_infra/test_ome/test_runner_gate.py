"""The Runner consults its gate after recording the attempt, before the body."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
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


class _Gate:
    """Waits the events tagged ``hold`` until :meth:`open`; runs the rest."""

    def __init__(self) -> None:
        self.opened = asyncio.Event()

    def open(self) -> None:
        self.opened.set()

    def verdict(self, event: object) -> str:
        held = getattr(event, "tag", "") == "hold" and not self.opened.is_set()
        return "wait" if held else "run"

    @asynccontextmanager
    async def lifted(self, event: object) -> AsyncIterator[None]:
        await self.opened.wait()
        yield

    @contextmanager
    def bind(self, event: object) -> Iterator[None]:
        yield


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
    store: RunRecordStore, gate: object, *, slots: int = 20, timeout: float = 1800
) -> Runner:
    return Runner(
        run_record_store=store,
        engine_sem=asyncio.Semaphore(slots),
        emit_hook=_no_emit,
        config=OMEConfig(jobstore_path=Path("unused"), run_timeout_seconds=timeout),
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


async def test_wait_releases_slot(store: RunRecordStore) -> None:
    gate = _Gate()
    calls: list[str] = []
    runner = _runner(store, gate, slots=1)
    held = asyncio.create_task(
        runner.run(
            _body(calls), _Scoped(tag="hold"), run_id="r1", max_retries_snapshot=0
        )
    )
    await asyncio.sleep(0.05)
    await asyncio.wait_for(
        runner.run(
            _body(calls), _Scoped(tag="free"), run_id="r2", max_retries_snapshot=0
        ),
        timeout=2,
    )
    assert calls == ["free"]
    gate.open()
    await asyncio.wait_for(held, timeout=2)
    assert calls == ["free", "hold"]


async def test_wait_not_counted_in_run_timeout(store: RunRecordStore) -> None:
    gate = _Gate()
    calls: list[str] = []
    runner = _runner(store, gate, timeout=0.2)
    held = asyncio.create_task(
        runner.run(
            _body(calls), _Scoped(tag="hold"), run_id="r1", max_retries_snapshot=0
        )
    )
    await asyncio.sleep(0.6)
    gate.open()
    await asyncio.wait_for(held, timeout=2)
    record = await store.get("r1")
    assert record is not None and record.status is RunStatus.SUCCESS
    assert calls == ["hold"]


async def test_cancel_while_waiting_keeps_running_row(store: RunRecordStore) -> None:
    runner = _runner(store, _Gate())
    held = asyncio.create_task(
        runner.run(_body([]), _Scoped(tag="hold"), run_id="r1", max_retries_snapshot=0)
    )
    await asyncio.sleep(0.05)
    held.cancel()
    with pytest.raises(asyncio.CancelledError):
        await held
    record = await store.get("r1")
    assert record is not None and record.status is RunStatus.RUNNING


async def test_purge_gate_holds_until_the_mark_lifts(
    store: RunRecordStore, tmp_path: Path
) -> None:
    marks = MARKS.for_root(tmp_path)
    marks.raise_mark("j1", S)
    calls: list[str] = []
    runner = _runner(store, PurgeRunGate(marks))
    event = _Scoped(app_id="a2", tag="other")  # another scope waits too
    held = asyncio.create_task(
        runner.run(_body(calls), event, run_id="r1", max_retries_snapshot=0)
    )
    await asyncio.sleep(0.1)
    assert calls == [] and marks.waiting_events == {event.event_id}
    marks.lift_mark()
    await asyncio.wait_for(held, timeout=2)
    assert calls == ["other"] and marks.waiting_events == set()
