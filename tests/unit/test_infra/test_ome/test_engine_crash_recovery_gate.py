"""Engine crash-recovery gate — pins ``OMEConfig.crash_recovery_enabled``.

``engine.start()`` unconditionally invoked ``_run_crash_recovery`` prior to
PR #361 review finding M10. The finding: a backfill engine registering only
Phase 2/3 strategies shares ``ome.db`` with the live server; if the server
had died leaving stale RUNNING rows for other strategies, the backfill's
``scan_and_resume`` would re-enqueue them into the backfill engine's own
scheduler and permanently lose them on dispatch (``KeyError``).

Fix: add ``OMEConfig.crash_recovery_enabled`` (default ``True`` — server
semantic unchanged). Backfill engines pass ``False``. These tests pin both
sides of the gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from everos.infra.ome import engine as engine_mod
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.engine import OfflineEngine


@pytest.mark.asyncio
async def test_start_runs_crash_recovery_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default config (``crash_recovery_enabled=True``) — ``scan_and_resume``
    must be called during ``engine.start()``. This is the server's semantic
    and must not regress.
    """
    calls: list[tuple[Any, ...]] = []

    async def _spy(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(engine_mod, "scan_and_resume", _spy)

    cfg = OMEConfig(jobstore_path=tmp_path / "ome.db", config_watch=False)
    assert cfg.crash_recovery_enabled is True
    engine = OfflineEngine(config=cfg)
    await engine.start()
    try:
        assert len(calls) == 1, "scan_and_resume must run on start() by default"
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_start_skips_crash_recovery_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opt-out (``crash_recovery_enabled=False``) — ``scan_and_resume``
    must NOT be called. Backfill's cluster + skill engines rely on this
    to avoid stealing the server's stale RUNNING rows.
    """
    calls: list[tuple[Any, ...]] = []

    async def _spy(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(engine_mod, "scan_and_resume", _spy)

    cfg = OMEConfig(
        jobstore_path=tmp_path / "ome.db",
        config_watch=False,
        crash_recovery_enabled=False,
    )
    engine = OfflineEngine(config=cfg)
    await engine.start()
    try:
        assert calls == [], "scan_and_resume must be skipped when opted out"
    finally:
        await engine.stop()
