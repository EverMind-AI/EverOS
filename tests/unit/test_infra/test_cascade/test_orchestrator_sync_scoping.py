"""Kind-scoped cascade sync (finding #6 round-1 + #3 round-2).

Backfill's skills phase used to call ``orchestrator.sync_once()``
unscoped, which walked every registered kind including knowledge_topic
/ knowledge_document. If the current process had embedding but not
rerank, cascade's knowledge handlers were gated off — the worker
would then see the knowledge md, find no handler, and mark each row
as permanently failed. A *backfill* run must not corrupt the queue
state of unrelated kinds.

Round-1 wired ``kinds={"agent_skill"}`` through the scanner but left
``worker.drain_until_empty`` kind-agnostic — the drain would still
claim any queued knowledge row and flip it to permanently failed.
Round-2 (finding #3) closes the drain end so the scoping intent
holds end-to-end. That the skills phase asks for ``{"agent_skill"}`` is
pinned in ``tests/unit/test_service/test_ops``.
"""

from __future__ import annotations

from pathlib import Path

from tests.fakes.storage import Storage


async def test_orchestrator_sync_once_default_scans_all(
    tmp_path: Path, monkeypatch, storage: Storage
) -> None:
    """Regression guard: the CLI ``cascade sync`` path (no ``kinds``
    argument) must keep its full-registry behaviour — the plumbing
    only opts in to scoping when the caller asks for it. Both the
    scanner and the worker's drain see ``kinds=None``."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))

    from everos.infra.cascade.scanner import CascadeScanner
    from everos.infra.cascade.worker import CascadeWorker

    captured: dict[str, object] = {}

    async def _spy_scan_once(self, *, kinds=None):  # type: ignore[no-untyped-def]
        captured["scan_kinds"] = kinds
        return []

    async def _spy_drain(self, *, kinds=None) -> int:  # type: ignore[no-untyped-def]
        captured["drain_kinds"] = kinds
        return 0

    monkeypatch.setattr(CascadeScanner, "scan_once", _spy_scan_once)
    monkeypatch.setattr(CascadeWorker, "drain_until_empty", _spy_drain)

    async def _spy_reconcile(memory_root, ledger) -> int:  # type: ignore[no-untyped-def]
        captured["reconciled"] = True
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

    assert captured["scan_kinds"] is None
    assert captured["drain_kinds"] is None
    # A full pass is where the one-time index reconcile runs.
    assert captured["reconciled"] is True


async def test_orchestrator_sync_once_forwards_kinds_to_worker_drain(
    tmp_path: Path, monkeypatch, storage: Storage
) -> None:
    """Round-2 finding #3: ``sync_once(kinds=...)`` MUST forward the
    filter to the worker's drain, not just the scanner. Without this,
    a Phase-3 sync scoped to ``{"agent_skill"}`` still drains any
    queued knowledge row and flips it to ``failed(retryable=False)``.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))

    from everos.infra.cascade.scanner import CascadeScanner
    from everos.infra.cascade.worker import CascadeWorker

    captured: dict[str, object] = {}

    async def _spy_scan_once(self, *, kinds=None):  # type: ignore[no-untyped-def]
        captured["scan_kinds"] = kinds
        return []

    async def _spy_drain(self, *, kinds=None) -> int:  # type: ignore[no-untyped-def]
        captured["drain_kinds"] = kinds
        return 0

    monkeypatch.setattr(CascadeScanner, "scan_once", _spy_scan_once)
    monkeypatch.setattr(CascadeWorker, "drain_until_empty", _spy_drain)

    async def _spy_reconcile(memory_root, ledger) -> int:  # type: ignore[no-untyped-def]
        captured["reconciled"] = True
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
    await orch.sync_once(kinds={"agent_skill"})

    assert captured["scan_kinds"] == {"agent_skill"}
    assert captured["drain_kinds"] == {"agent_skill"}
    # A scoped pass drains only its kinds, so it must not queue the rest.
    assert "reconciled" not in captured
