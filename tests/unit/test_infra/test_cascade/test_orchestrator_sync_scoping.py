"""``cascade sync``: one full pass over the root.

``sync_once`` used to take ``kinds`` so the backfill's skills phase could
sync its skill files alone; EverOS 2.0 removed that phase with agent memory.
Every pass is now a full one: the one-time index reconcile, then the scan of
every registered kind, then the drain of every kind.
"""

from __future__ import annotations

from pathlib import Path

from tests.fakes.storage import Storage


async def test_orchestrator_sync_once_default_scans_all(
    tmp_path: Path, monkeypatch, storage: Storage
) -> None:
    """The CLI ``cascade sync`` path: reconcile, then scan, then drain."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))

    from everos.infra.cascade._worker import CascadeWorker
    from everos.infra.cascade.scanner import CascadeScanner

    calls: list[str] = []

    async def _spy_scan_once(self):  # type: ignore[no-untyped-def]
        calls.append("scan")
        return []

    async def _spy_drain(self, *, max_passes: int = 100) -> int:  # type: ignore[no-untyped-def]
        calls.append("drain")
        return 0

    monkeypatch.setattr(CascadeScanner, "scan_once", _spy_scan_once)
    monkeypatch.setattr(CascadeWorker, "drain_until_empty", _spy_drain)

    async def _spy_reconcile(memory_root, ledger) -> int:  # type: ignore[no-untyped-def]
        calls.append("reconcile")
        return 0

    import everos.infra.cascade.orchestrator as orchestrator_mod

    monkeypatch.setattr(orchestrator_mod, "reconcile_index_once", _spy_reconcile)

    from everos.component.tokenizer import build_tokenizer
    from everos.infra.cascade.orchestrator import CascadeOrchestrator
    from everos.infra.persistence import MemoryRoot

    orch = CascadeOrchestrator(
        memory_root=MemoryRoot.resolve(),
        tokenizer=build_tokenizer(),
        embedding=None,
        index=storage.index,
        sqlite=storage.sqlite,
    )
    await orch.sync_once()

    # A full pass is where the one-time index reconcile runs, before the scan.
    assert calls == ["reconcile", "scan", "drain"]
