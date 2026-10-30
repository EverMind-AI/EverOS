"""Backfill's replay engines opt out of crash recovery.

Pins PR #361 review finding M10: the throw-away engines
:func:`~everos.service.bootstrap.build_replay_engine` builds for backfill's
clusters and skills phases share the server's ``ome.db`` jobstore path.
If ``engine.start()`` ran ``_run_crash_recovery`` on them, a stale
RUNNING row for a strategy the backfill engine does NOT register
(e.g. ``extract_atomic_facts`` left over from a crashed server) would
be re-enqueued into the backfill engine's own APS scheduler, hit
``StrategyRegistry.get`` with an unknown name at dispatch, and be
permanently lost.

Fix: the builder sets ``OMEConfig.crash_recovery_enabled=False``.
The stale row stays untouched and is resumed on the next server
restart, by an engine that actually knows those strategy names.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from everos.config import OmeSettings
from everos.core.utils.datetime import get_now_with_timezone, to_iso_format
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.records import RunStatus
from everos.infra.persistence.memory_root import MemoryRoot
from everos.memory.strategies import (
    EXTRACT_AGENT_SKILL_SPEC,
    TRIGGER_PROFILE_CLUSTERING_SPEC,
    TRIGGER_SKILL_CLUSTERING_SPEC,
)
from everos.service.bootstrap import build_replay_engine
from tests.fakes import make_deps

_CLUSTER_SPECS = (TRIGGER_PROFILE_CLUSTERING_SPEC, TRIGGER_SKILL_CLUSTERING_SPEC)


@pytest.mark.parametrize(
    "specs", [_CLUSTER_SPECS, (EXTRACT_AGENT_SKILL_SPEC,)], ids=["clusters", "skills"]
)
def test_replay_engines_have_crash_recovery_disabled(
    tmp_path: Path, specs: tuple[Any, ...]
) -> None:
    engine = build_replay_engine(
        MemoryRoot(tmp_path), make_deps(), OmeSettings(), specs
    )
    assert engine._config.crash_recovery_enabled is False


@pytest.mark.asyncio
async def test_backfill_engine_start_does_not_reenqueue_stale_running_rows(
    tmp_path: Path,
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
    root = MemoryRoot(tmp_path)
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
    # Age the row anyway: crash recovery sweeps every RUNNING row, so the
    # "row untouched" assertion below holds only because the replay engine
    # opts out of recovery, not because the row is too young.
    async with storage.connect() as conn:
        rewind = to_iso_format(get_now_with_timezone() - timedelta(hours=2))
        await conn.execute(
            "UPDATE run_record SET started_at = ? WHERE run_id = ?",
            (rewind, "r_server_stale"),
        )
        await conn.commit()

    engine = build_replay_engine(root, make_deps(), OmeSettings(), _CLUSTER_SPECS)
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
