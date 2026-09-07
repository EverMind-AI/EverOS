"""Integration test for ``everos cascade backfill --phase skills``.

Exercises the real Phase 3 scan + synthetic-event-emission + cascade-sync
path (see ``memory.cascade._backfill._run_phase_skills``): Phase 2's
throwaway engine (``_build_cluster_engine``, Task 22) registers
``trigger_skill_clustering`` but never ``extract_agent_skill``, so every
``SkillClusterUpdated`` it emits during clustering has no listener and is
dropped. Phase 3 walks every agent-case cluster
(``cluster_repo.list_distinct_owners`` + ``list_for_owner``) and replays
one ``SkillClusterUpdated`` per clustered case through a second throwaway
engine that DOES register ``extract_agent_skill``, then runs a one-shot
cascade sync (``CascadeOrchestrator.sync_once``) so the freshly-written
``SKILL.md`` lands in LanceDB before the phase reports its skill count.

Only ``AgentSkillExtractor`` (the LLM call) and ``get_llm_client`` are
mocked — everything else (cluster lookup, agent-case lookup, the
markdown writer, and the cascade scan/handler/embed round trip) is real,
so ``test_skills_phase_grows_agent_skill_count`` is a genuine end-to-end
proof that Phase 3 leaves ``agent_skill_repo.count()`` already grown,
not just eventually-consistent.

Covers: empty DB (nothing to backfill, no engine touched), declined
confirmation (exit 1, no skill written), a small DB with one clustered
agent case (skill extracted + indexed, exit 0), and a missing embedding
capability (exit 2, mirroring Phase 1 / Phase 2's own guard).
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.types import AgentSkill as AlgoAgentSkill

from everos.component.embedding import EmbeddingCapability, EmbeddingProvider
from everos.component.utils.datetime import get_utc_now
from everos.config import load_settings
from everos.entrypoints.cli.commands._backfill_cmd import run_backfill
from everos.infra.persistence.lancedb import (
    AgentCase,
    agent_case_repo,
    agent_skill_repo,
    dispose_connection,
)
from everos.infra.persistence.sqlite import (
    cluster_repo,
    mint_cluster_id,
    sqlite_manager,
)
from everos.memory.cascade import _backfill as backfill_mod

_DIM = 1024


class _StubEmbedder(EmbeddingProvider):
    dim = _DIM

    async def embed(self, text: str) -> list[float]:
        return [float(len(text) % 7)] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [[float(len(t) % 7)] * self.dim for t in texts]


@pytest.fixture
async def backfill_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Path]:
    """Tmp memory root + stub embedder + isolated sqlite engine.

    Phase 3 touches LanceDB (agent_case / agent_skill), sqlite (cluster
    tables + md_change_state), and the real filesystem (SKILL.md write +
    cascade scan) — mirrors ``backfill_runtime`` in
    ``test_backfill_phase2.py``, plus resetting the strategy module's
    ``_writer`` singleton so it re-resolves against ``tmp_path``.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    await dispose_connection()
    if sqlite_manager._engine is not None:
        await sqlite_manager.dispose_engine()
    monkeypatch.setattr(sqlite_manager, "_engine", None, raising=False)
    monkeypatch.setattr(sqlite_manager, "_session_factory", None, raising=False)

    import everos.component.embedding.accessor as acc

    monkeypatch.setattr(
        acc, "_capability", EmbeddingCapability(provider=_StubEmbedder())
    )

    skill_mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")
    monkeypatch.setattr(skill_mod, "_writer", None, raising=False)

    yield tmp_path

    await dispose_connection()
    if sqlite_manager._engine is not None:
        await sqlite_manager.dispose_engine()


@pytest.fixture
def stub_llm_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[AlgoAgentSkill]]:
    """Mock the one non-deterministic seam: the LLM-backed extractor.

    Everything downstream of ``AgentSkillExtractor.aextract`` (writer,
    cascade scan, embed, LanceDB upsert) stays real.
    """
    emitted = [
        AlgoAgentSkill(
            id="dummy",
            cluster_id="",
            name="summarise_doc",
            description="how to summarise a document",
            content="step 1: read; step 2: condense",
            confidence=0.7,
            maturity_score=0.5,
            source_case_ids=["ac1"],
        )
    ]
    # ``strategies/__init__.py`` re-exports the decorated ``Strategy``
    # object under this same dotted name, shadowing the submodule as a
    # package attribute — ``importlib.import_module`` (not
    # ``import ... as``) is required to reach the actual module and
    # patch its module-level references (mirrors
    # ``test_ome_strategies_integration.test_real_offline_engine_grows_
    # cluster_count``).
    skill_mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")
    monkeypatch.setattr(skill_mod, "get_llm_client", lambda: object())
    with patch.object(skill_mod, "AgentSkillExtractor") as mock_extractor_cls:
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=emitted)
        yield emitted


def _agent_case(entry_id: str, *, agent_id: str = "agent1") -> AgentCase:
    return AgentCase(
        id=f"{agent_id}_{entry_id}",
        entry_id=entry_id,
        owner_id=agent_id,
        owner_type="agent",
        session_id="s1",
        timestamp=get_utc_now(),
        parent_id=f"mc_{entry_id}",
        quality_score=0.8,
        task_intent=f"task intent {entry_id}",
        task_intent_tokens=f"task intent {entry_id}",
        approach=f"approach {entry_id}",
        approach_tokens=f"approach {entry_id}",
        md_path=f"agents/default/default/{agent_id}/.cases/agent_case-2026-01-01.md",
        content_sha256=hashlib.sha256(entry_id.encode()).hexdigest(),
        vector=[0.1] * _DIM,
    )


async def _seed_cluster(*, agent_id: str = "agent1", case_entry_id: str = "ac1") -> str:
    """Seed the sqlite ``cluster`` + ``cluster_member`` pair Phase 3 scans."""
    await backfill_mod._ensure_cluster_schema()
    cluster_id = mint_cluster_id()
    algo_cluster = AlgoCluster(
        id=cluster_id,
        centroid=np.zeros(_DIM, dtype=np.float32),
        count=1,
        last_ts=1_700_000_000_000,
        preview=[f"task intent {case_entry_id}"],
        members=[case_entry_id],
    )
    await cluster_repo.upsert_with_members(
        algo_cluster,
        owner_id=agent_id,
        owner_type="agent",
        kind="agent_case",
        member_type="case",
    )
    return cluster_id


async def test_empty_db_reports_nothing_to_backfill(
    backfill_runtime: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await backfill_mod._ensure_cluster_schema()

    code = await run_backfill(phase="skills", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 0
    assert "Nothing to backfill" in out
    assert await agent_skill_repo.count() == 0


async def test_declined_confirmation_writes_nothing_and_exits_one(
    backfill_runtime: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await agent_case_repo.add([_agent_case("ac1")])
    await _seed_cluster()
    monkeypatch.setattr("typer.confirm", lambda *a, **k: False)

    code = await run_backfill(phase="skills", auto_yes=False)

    assert code == 1
    assert await agent_skill_repo.count() == 0


async def test_missing_embedding_capability_returns_exit_two(
    backfill_runtime: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors Phase 1 / Phase 2's own guard: no embedding provider means
    ``extract_agent_skill``'s own top-K cosine ranking (and the cascade
    handler's re-embed on write-back) can't run, so the phase fails
    clean with exit 2 rather than emitting events into a broken pipeline."""
    import everos.component.embedding.accessor as acc

    await agent_case_repo.add([_agent_case("ac1")])
    await _seed_cluster()
    monkeypatch.setattr(acc, "_capability", EmbeddingCapability(provider=None))

    code = await run_backfill(phase="skills", auto_yes=True)

    assert code == 2
    assert await agent_skill_repo.count() == 0


async def test_skills_phase_grows_agent_skill_count(
    backfill_runtime: Path,
    stub_llm_extraction: list[AlgoAgentSkill],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: cluster with no skill yet → Phase 3 replays the event
    the clustering pass never got a listener for → real ``SKILL.md``
    write → real cascade sync → ``agent_skill_repo.count()`` grows."""
    await agent_case_repo.add([_agent_case("ac1")])
    await _seed_cluster()

    skills_before = await agent_skill_repo.count()
    assert skills_before == 0

    code = await run_backfill(phase="skills", auto_yes=True)
    out = capsys.readouterr().out

    skills_after = await agent_skill_repo.count()
    assert code == 0
    assert skills_after == skills_before + 1
    assert "phase 3 complete" in out

    rows = await agent_skill_repo.find_where("owner_id = 'agent1'", limit=10)
    assert len(rows) == 1
    assert rows[0].name == "summarise_doc"
    assert rows[0].cluster_id is not None
