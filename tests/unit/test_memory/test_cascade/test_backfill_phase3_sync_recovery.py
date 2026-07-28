"""Phase 3 rerun sync recovery (round-2 finding M1).

Group F (commit 9eaa73c) added an on-disk idempotency probe
(``_skill_md_exists_for_cluster``) that lets ``_scan_skill_source``
skip clusters whose ``SKILL.md`` already exists on disk. That fixed
the mid-cluster duplicate-extract race, but introduced a new failure
mode when combined with the Phase-3 shape at the time: the trailing
``_sync_new_skill_files`` sat inside the "have work" branch, so a
scan that legitimately returned empty (because the probe skipped
every cluster) short-circuited before sync could run — orphan
``SKILL.md`` files written by a prior Ctrl-C'd Phase 3 stayed
unindexed until the user manually ran ``everos cascade sync``.

Fix: move ``_sync_new_skill_files`` to before ``_scan_skill_source``
so the recovery is unconditional. This module pins that placement so
the regression can't slip back.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingCapability
from everos.memory.cascade import _backfill
from everos.memory.cascade._backfill import NullBackfillPresenter


async def test_phase3_rerun_syncs_orphan_skill_md_before_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_scan_skill_source`` returning empty must NOT skip sync.

    Simulates a rerun after a Ctrl-C mid-Phase-3: the on-disk probe
    has skipped every cluster (so the scan yields no work), yet the
    orphan ``SKILL.md`` files from the interrupted run still need to
    reach LanceDB. ``_sync_new_skill_files`` must run exactly once,
    before the scan-empty short-circuit.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))

    import everos.component.embedding.accessor as acc

    class _StubProvider:
        dim = 1
        available = True

        async def embed(self, text: str) -> list[float]:
            return [0.0]

        async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
            return [[0.0] for _ in texts]

    monkeypatch.setattr(
        acc, "_capability", EmbeddingCapability(provider=_StubProvider())
    )
    monkeypatch.setattr(
        _backfill,
        "_probe_ome_lock_available",
        lambda: True,
    )
    # Scan returns empty — simulating "every cluster already has SKILL.md".
    monkeypatch.setattr(_backfill, "_scan_skill_source", lambda: _empty_scan())

    sync_calls = 0

    async def _spy_sync() -> None:
        nonlocal sync_calls
        sync_calls += 1

    monkeypatch.setattr(_backfill, "_sync_new_skill_files", _spy_sync)

    result = await _backfill._run_phase_skills(
        auto_yes=True, presenter=NullBackfillPresenter()
    )

    # Sync ran despite the scan short-circuit — the recovery path is
    # exactly what M1 restores. The "nothing to backfill" branch also
    # signals `skills_before/after == 0` (never assigned).
    assert sync_calls == 1
    assert result.aborted is False
    assert result.blocked_by_capability is None


async def _empty_scan() -> list[_backfill._SkillSourceRow]:
    return []
