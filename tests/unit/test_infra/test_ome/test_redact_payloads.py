"""Replacing a purged scope's stored events in finished, unsuccessful runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from everos.core.utils.datetime import get_now_with_timezone
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.records import RunStatus

MARKER = '{"purged_by":"j1"}'


def _in_a1(payload: str) -> bool:
    try:
        data = json.loads(payload)
    except ValueError:
        return False
    return isinstance(data, dict) and data.get("app_id") == "a1"


@pytest.fixture
async def store(tmp_path: Path) -> RunRecordStore:
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    return RunRecordStore(storage=storage, max_records_per_strategy=100)


async def _run(store: RunRecordStore, run_id: str, payload: str, end: str) -> None:
    await store.mark_running(
        run_id=run_id,
        strategy_name="s",
        attempt=0,
        event_topic="x:Y",
        event_payload=payload,
        max_retries_snapshot=1,
        event_id=f"evt_{run_id}",
    )
    now = get_now_with_timezone()
    if end == "success":
        await store.mark_success(run_id=run_id, finished_at=now)
    elif end == "failed":
        await store.mark_failed(run_id=run_id, finished_at=now, error="e")
    elif end == "dead_letter":
        await store.mark_dead_letter(run_id=run_id, finished_at=now, error="e")
    elif end == "crashed":
        await store.mark_crashed(run_id=run_id, finished_at=now, error="e")


async def test_redact_rewrites_failed_rows_only(store: RunRecordStore) -> None:
    a1 = json.dumps({"app_id": "a1", "project_id": "p1", "text": "secret"})
    a2 = json.dumps({"app_id": "a2", "project_id": "p1", "text": "keep"})
    for run_id, payload, end in [
        ("running", a1, "running"),
        ("success", a1, "success"),
        ("failed", a1, "failed"),
        ("dead", a1, "dead_letter"),
        ("crashed", a1, "crashed"),
        ("other", a2, "failed"),
    ]:
        await _run(store, run_id, payload, end)
    assert await store.redact_payloads(_in_a1, MARKER) == 3
    payloads = {
        r.run_id: r.event_payload for r in await store.list_runs(strategy_name="s")
    }
    assert payloads == {
        "running": a1,
        "success": "",
        "failed": MARKER,
        "dead": MARKER,
        "crashed": MARKER,
        "other": a2,
    }
    assert (await store.get("failed")).status is RunStatus.FAILED  # type: ignore[union-attr]
    assert await store.redact_payloads(_in_a1, MARKER) == 0  # idempotent


async def test_redact_skips_rows_without_scope(store: RunRecordStore) -> None:
    tick = json.dumps({"fired_at": "2026-11-11T00:00:00Z"})
    await _run(store, "tick", tick, "failed")
    assert await store.redact_payloads(_in_a1, MARKER) == 0
    assert (await store.get("tick")).event_payload == tick  # type: ignore[union-attr]
