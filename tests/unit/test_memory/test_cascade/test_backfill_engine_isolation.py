"""Backfill OME engines opt out of crash recovery.

Pins PR #361 review finding M10: ``_build_cluster_engine`` and
``_build_skill_engine`` share the server's ``ome.db`` jobstore path.
If ``engine.start()`` ran ``_run_crash_recovery`` on them, a stale
RUNNING row for a strategy the backfill engine does NOT register
(e.g. ``extract_atomic_facts`` left over from a crashed server) would
be re-enqueued into the backfill engine's own APS scheduler, hit
``StrategyRegistry.get`` with an unknown name at dispatch, and be
permanently lost.

Fix: both builders set ``OMEConfig.crash_recovery_enabled=False``.
The stale row stays untouched and is resumed on the next server
restart, by an engine that actually knows those strategy names.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from everos.component.utils.datetime import get_now_with_timezone, to_iso_format
from everos.core.persistence.memory_root import MemoryRoot
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.records import RunStatus
from everos.memory.cascade import _backfill


@pytest.fixture
def _isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``MemoryRoot.resolve()`` at a per-test temp directory so the
    builders don't touch ``~/.everos`` and don't collide across tests.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    return tmp_path


def test_cluster_engine_has_crash_recovery_disabled(_isolated_root: Path) -> None:
    """Phase-2 engine builder wires ``crash_recovery_enabled=False``."""
    engine = _backfill._build_cluster_engine()
    assert engine._config.crash_recovery_enabled is False


def test_skill_engine_has_crash_recovery_disabled(_isolated_root: Path) -> None:
    """Phase-3 engine builder wires ``crash_recovery_enabled=False``."""
    engine = _backfill._build_skill_engine()
    assert engine._config.crash_recovery_enabled is False


@pytest.mark.asyncio
async def test_backfill_engine_start_does_not_reenqueue_stale_running_rows(
    _isolated_root: Path,
) -> None:
    """End-to-end M10 pin: seed a stale RUNNING row for a strategy the
    backfill engine does NOT register, then start the backfill engine.

    Expected: the stale row remains RUNNING (no CRASHED transition),
    and no APS job was scheduled for it — because backfill's engine
    opts out of crash recovery entirely.
    """
    # Seed a stale RUNNING row directly on the shared ome.db, mimicking
    # a prior server session that crashed mid-flight on a strategy the
    # backfill engine does not know.
    root = MemoryRoot.resolve()
    root.ome_db.parent.mkdir(parents=True, exist_ok=True)
    storage = OMEStorage(db_path=root.ome_db)
    await storage.init()
    rec_store = RunRecordStore(storage=storage, max_records_per_strategy=1000)
    await rec_store.mark_running(
        run_id="r_server_stale",
        strategy_name="extract_atomic_facts",
        attempt=0,
        event_topic="everos.memory.events:MemCellCommitted",
        event_payload="{}",
        max_retries_snapshot=1,
        event_id="evt_stale",
    )
    # Rewind ``started_at`` past ``crash_recovery_timeout_seconds`` so
    # that IF crash recovery had run, this row would be swept — making
    # the "row untouched" assertion meaningful.
    async with storage.connect() as conn:
        rewind = to_iso_format(get_now_with_timezone() - timedelta(hours=2))
        await conn.execute(
            "UPDATE run_record SET started_at = ? WHERE run_id = ?",
            (rewind, "r_server_stale"),
        )
        await conn.commit()

    engine = _backfill._build_cluster_engine()
    await engine.start()
    try:
        rec = await rec_store.get("r_server_stale")
        assert rec is not None
        assert rec.status == RunStatus.RUNNING, (
            "backfill engine must not sweep the server's stale RUNNING rows"
        )
        # And no APS job was scheduled for the stale run_id — the
        # backfill scheduler stayed empty (only Cron/Idle jobs would
        # exist, and neither cluster strategy uses those triggers).
        assert engine._scheduler.get_job("r_server_stale") is None
    finally:
        await engine.stop()
