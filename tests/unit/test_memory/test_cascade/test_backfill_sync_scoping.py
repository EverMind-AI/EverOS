"""Kind-scoped sync in Phase 3 (finding #6 round-1 + #3 round-2).

``_sync_new_skill_files`` used to call ``orchestrator.sync_once()``
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
holds end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from everos.memory.cascade import _backfill


async def test_sync_new_skill_files_only_syncs_agent_skill_kind(
    tmp_path: Path, monkeypatch
) -> None:
    """``_sync_new_skill_files`` forwards ``kinds={"agent_skill"}`` to
    the orchestrator so an unscoped sweep never fires.

    We spy on :meth:`CascadeOrchestrator.sync_once` at the class level:
    the helper builds its own orchestrator instance, so a per-instance
    patch would miss it. The spy records the ``kinds`` kwarg and
    short-circuits (returns 0) so no real cascade machinery runs.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))

    from everos.memory.cascade.orchestrator import CascadeOrchestrator

    captured: dict[str, object] = {}

    async def _spy_sync_once(self, *, kinds=None) -> int:  # type: ignore[no-untyped-def]
        captured["kinds"] = kinds
        return 0

    monkeypatch.setattr(CascadeOrchestrator, "sync_once", _spy_sync_once)

    await _backfill._sync_new_skill_files()

    assert captured["kinds"] == {"agent_skill"}


async def test_orchestrator_sync_once_default_scans_all(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression guard: the CLI ``cascade sync`` path (no ``kinds``
    argument) must keep its full-registry behaviour — the plumbing
    only opts in to scoping when the caller asks for it. Both the
    scanner and the worker's drain see ``kinds=None``."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))

    from everos.memory.cascade.scanner import CascadeScanner
    from everos.memory.cascade.worker import CascadeWorker

    captured: dict[str, object] = {}

    async def _spy_scan_once(self, *, kinds=None):  # type: ignore[no-untyped-def]
        captured["scan_kinds"] = kinds
        return []

    async def _spy_drain(self, *, kinds=None) -> int:  # type: ignore[no-untyped-def]
        captured["drain_kinds"] = kinds
        return 0

    monkeypatch.setattr(CascadeScanner, "scan_once", _spy_scan_once)
    monkeypatch.setattr(CascadeWorker, "drain_until_empty", _spy_drain)

    from everos.component.tokenizer import build_tokenizer
    from everos.core.persistence import MemoryRoot
    from everos.memory.cascade.orchestrator import CascadeOrchestrator

    orch = CascadeOrchestrator(
        memory_root=MemoryRoot.resolve(), tokenizer=build_tokenizer()
    )
    await orch.sync_once()

    assert captured["scan_kinds"] is None
    assert captured["drain_kinds"] is None


async def test_orchestrator_sync_once_forwards_kinds_to_worker_drain(
    tmp_path: Path, monkeypatch
) -> None:
    """Round-2 finding #3: ``sync_once(kinds=...)`` MUST forward the
    filter to the worker's drain, not just the scanner. Without this,
    a Phase-3 sync scoped to ``{"agent_skill"}`` still drains any
    queued knowledge row and flips it to ``failed(retryable=False)``.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))

    from everos.memory.cascade.scanner import CascadeScanner
    from everos.memory.cascade.worker import CascadeWorker

    captured: dict[str, object] = {}

    async def _spy_scan_once(self, *, kinds=None):  # type: ignore[no-untyped-def]
        captured["scan_kinds"] = kinds
        return []

    async def _spy_drain(self, *, kinds=None) -> int:  # type: ignore[no-untyped-def]
        captured["drain_kinds"] = kinds
        return 0

    monkeypatch.setattr(CascadeScanner, "scan_once", _spy_scan_once)
    monkeypatch.setattr(CascadeWorker, "drain_until_empty", _spy_drain)

    from everos.component.tokenizer import build_tokenizer
    from everos.core.persistence import MemoryRoot
    from everos.memory.cascade.orchestrator import CascadeOrchestrator

    orch = CascadeOrchestrator(
        memory_root=MemoryRoot.resolve(), tokenizer=build_tokenizer()
    )
    await orch.sync_once(kinds={"agent_skill"})

    assert captured["scan_kinds"] == {"agent_skill"}
    assert captured["drain_kinds"] == {"agent_skill"}
