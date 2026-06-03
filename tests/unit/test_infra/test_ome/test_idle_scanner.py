from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from everos.infra.ome._background.idle_scanner import IdleScanner
from everos.infra.ome._stores.idle import IdleStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.events import BaseEvent, IdleTick
from everos.infra.ome.triggers import Idle


class _M(BaseEvent):
    user_id: str = "u1"


@pytest.mark.asyncio
async def test_scan_once_emits_idle_ticks(tmp_path: Path) -> None:
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    idle_store = IdleStore(storage=storage)
    now = datetime.now(UTC)
    await idle_store.touch("s", "u_old", at=now - timedelta(seconds=2000))
    await idle_store.touch("s", "u_fresh", at=now)

    emitted: list[IdleTick] = []

    async def emit(e: BaseEvent) -> None:
        if isinstance(e, IdleTick):
            emitted.append(e)

    trigger = Idle(on=[_M], event_field="user_id", idle_seconds=900)
    scanner = IdleScanner(
        strategy_name="s",
        trigger=trigger,
        idle_store=idle_store,
        emit=emit,
    )
    await scanner.scan_once(now=now)
    assert {e.bucket_key for e in emitted} == {"u_old"}
    assert all(e.strategy_name == "s" for e in emitted)


@pytest.mark.asyncio
async def test_scan_once_with_now_none_uses_current_time(tmp_path: Path) -> None:
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    idle_store = IdleStore(storage=storage)
    now = datetime.now(UTC)
    # Insert bucket with activity timestamp older than the threshold
    await idle_store.touch("s", "u_overdue", at=now - timedelta(seconds=2000))

    emitted: list[IdleTick] = []

    async def emit(e: BaseEvent) -> None:
        if isinstance(e, IdleTick):
            emitted.append(e)

    trigger = Idle(on=[_M], event_field="user_id", idle_seconds=900)
    scanner = IdleScanner(
        strategy_name="s",
        trigger=trigger,
        idle_store=idle_store,
        emit=emit,
    )
    # Call scan_once with no now= argument; should use current time internally
    await scanner.scan_once()
    # Should emit idle tick for overdue bucket
    assert len(emitted) >= 1
    assert any(e.bucket_key == "u_overdue" for e in emitted)
    assert all(e.strategy_name == "s" for e in emitted)


@pytest.mark.asyncio
async def test_scan_once_isolates_failing_emit(tmp_path: Path) -> None:
    """A single bucket's emit failure must not abort the rest of the
    scan. Mirrors dispatcher's _safe_applies isolation: one transient
    downstream error shouldn't drop sibling IdleTicks for this round.
    """
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    idle_store = IdleStore(storage=storage)
    now = datetime.now(UTC)
    # Three overdue buckets — middle one's emit will raise.
    for bucket in ("u_a", "u_boom", "u_c"):
        await idle_store.touch("s", bucket, at=now - timedelta(seconds=2000))

    emitted: list[str] = []

    async def emit(e: BaseEvent) -> None:
        if isinstance(e, IdleTick):
            if e.bucket_key == "u_boom":
                raise RuntimeError("downstream dispatch transient error")
            emitted.append(e.bucket_key)

    trigger = Idle(on=[_M], event_field="user_id", idle_seconds=900)
    scanner = IdleScanner(
        strategy_name="s",
        trigger=trigger,
        idle_store=idle_store,
        emit=emit,
    )
    # Must NOT raise; emit failure for u_boom is swallowed + logged.
    await scanner.scan_once(now=now)

    # Both sibling buckets still received their IdleTick.
    assert sorted(emitted) == ["u_a", "u_c"]
