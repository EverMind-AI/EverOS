"""Integration test for ``everos cascade backfill --phase skills``.

Exercises the real Phase 3 scan + synthetic-event-emission + cascade-sync
path (see ``service.backfill._run_phase_skills``): Phase 2's
throwaway engine (``_build_cluster_engine``, Task 22) registers
``trigger_skill_clustering`` but never ``extract_agent_skill``, so every
``SkillClusterUpdated`` it emits during clustering has no listener and is
dropped. Phase 3 walks every agent-case cluster
(``storage_view.sqlite.cluster.list_distinct_owners`` + ``list_for_owner``) and replays
one ``SkillClusterUpdated`` per clustered case through a second throwaway
engine that DOES register ``extract_agent_skill``, then runs a one-shot
cascade sync (``CascadeOrchestrator.sync_once``) so the freshly-written
``SKILL.md`` lands in LanceDB before the phase reports its skill count.

Only ``AgentSkillExtractor`` (the LLM call) and the LLM client are
mocked — everything else (cluster lookup, agent-case lookup, the
markdown writer, and the cascade scan/handler/embed round trip) is real,
so ``test_skills_phase_grows_agent_skill_count`` is a genuine end-to-end
proof that Phase 3 leaves ``storage_view.repos.agent_skill.count()`` already grown,
not just eventually-consistent.

Covers: empty DB (nothing to backfill, no engine touched), declined
confirmation (exit 1, no skill written), a small DB with one clustered
agent case (skill extracted + indexed, exit 0), and a missing embedding
capability (exit 2, mirroring Phase 1 / Phase 2's own guard).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.types import AgentSkill as AlgoAgentSkill

from everos.component.embedding import EmbeddingProvider
from everos.component.utils.datetime import get_utc_now
from everos.config import load_settings
from everos.entrypoints.cli.commands._backfill_cmd import run_backfill
from everos.infra.persistence.lancedb import (
    AgentCase,
)
from everos.memory.models import mint_cluster_id
from everos.service import OpsService, build_application, build_providers
from everos.service import backfill as backfill_mod
from tests.fakes.storage import StorageView

_DIM = 1024


class _StubEmbedder(EmbeddingProvider):
    dim = _DIM

    async def embed(self, text: str) -> list[float]:
        return [float(len(text) % 7)] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [[float(len(t) % 7)] * self.dim for t in texts]


_PROVIDERS: dict[str, Any] = {"embedding": None}
"""The embedder the test configured; ``_ops()`` builds the app over it."""


@pytest.fixture
async def backfill_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Path]:
    """Tmp memory root + stub embedder.

    Phase 3 touches LanceDB (agent_case / agent_skill), sqlite (cluster
    tables + md_change_state), and the real filesystem (SKILL.md write +
    cascade scan) — mirrors ``backfill_runtime`` in
    ``test_backfill_phase2.py``, plus resetting the strategy module's
    ``_writer`` singleton so it re-resolves against ``tmp_path``.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()

    _PROVIDERS["embedding"] = _StubEmbedder()

    yield tmp_path


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
    # ``strategies/__init__.py`` re-exports the strategy function under this
    # same dotted name, shadowing the submodule as a package attribute —
    # ``importlib.import_module`` (not ``import ... as``) reaches the module.
    # The backfill engine's LLM comes from ``_ops`` (a configured-looking
    # client is what lets the strategy run at all).
    skill_mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")
    with patch.object(skill_mod, "AgentSkillExtractor") as mock_extractor_cls:
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=emitted)
        yield emitted


def _ops() -> OpsService:
    """The ops facade over the embedding provider the test configured.

    The phases read their embedding provider from the application; tests
    here pick it in ``_PROVIDERS``, so the application is built over
    whatever that holds when the phase runs.
    """
    settings = load_settings()
    providers = dataclasses.replace(
        build_providers(settings),
        embedding=_PROVIDERS["embedding"],
        # The backfill engine binds its LLM when it builds the strategy
        # deps; a configured-looking client is what lets skills run at all.
        llm=object(),  # type: ignore[arg-type]
    )
    return build_application(settings, providers=providers).ops


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


async def _seed_cluster(
    storage_view: StorageView, *, agent_id: str = "agent1", case_entry_id: str = "ac1"
) -> str:
    """Seed the sqlite ``cluster`` + ``cluster_member`` pair Phase 3 scans."""
    await backfill_mod._ensure_cluster_schema(storage_view.database)
    cluster_id = mint_cluster_id()
    algo_cluster = AlgoCluster(
        id=cluster_id,
        centroid=np.zeros(_DIM, dtype=np.float32),
        count=1,
        last_ts=1_700_000_000_000,
        preview=[f"task intent {case_entry_id}"],
        members=[case_entry_id],
    )
    await storage_view.sqlite.cluster.upsert_with_members(
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
    storage_view: StorageView,
) -> None:
    await backfill_mod._ensure_cluster_schema(storage_view.database)

    code = await run_backfill(_ops(), phase="skills", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 0
    assert "Nothing to backfill" in out
    assert await storage_view.repos.agent_skill.count() == 0


async def test_declined_confirmation_writes_nothing_and_exits_one(
    backfill_runtime: Path, monkeypatch: pytest.MonkeyPatch, storage_view: StorageView
) -> None:
    await storage_view.repos.agent_case.add([_agent_case("ac1")])
    await _seed_cluster(storage_view)
    monkeypatch.setattr("typer.confirm", lambda *a, **k: False)

    code = await run_backfill(_ops(), phase="skills", auto_yes=False)

    assert code == 1
    assert await storage_view.repos.agent_skill.count() == 0


async def test_missing_embedding_capability_returns_exit_two(
    backfill_runtime: Path, monkeypatch: pytest.MonkeyPatch, storage_view: StorageView
) -> None:
    """Mirrors Phase 1 / Phase 2's own guard: no embedding provider means
    ``extract_agent_skill``'s own top-K cosine ranking (and the cascade
    handler's re-embed on write-back) can't run, so the phase fails
    clean with exit 2 rather than emitting events into a broken pipeline."""

    await storage_view.repos.agent_case.add([_agent_case("ac1")])
    await _seed_cluster(storage_view)
    _PROVIDERS["embedding"] = None

    code = await run_backfill(_ops(), phase="skills", auto_yes=True)

    assert code == 2
    assert await storage_view.repos.agent_skill.count() == 0


async def test_skills_phase_grows_agent_skill_count(
    backfill_runtime: Path,
    stub_llm_extraction: list[AlgoAgentSkill],
    capsys: pytest.CaptureFixture[str],
    storage_view: StorageView,
) -> None:
    """End-to-end: cluster with no skill yet → Phase 3 replays the event
    the clustering pass never got a listener for → real ``SKILL.md``
    write → real cascade sync → ``storage_view.repos.agent_skill.count()`` grows."""
    await storage_view.repos.agent_case.add([_agent_case("ac1")])
    await _seed_cluster(storage_view)

    skills_before = await storage_view.repos.agent_skill.count()
    assert skills_before == 0

    code = await run_backfill(_ops(), phase="skills", auto_yes=True)
    out = capsys.readouterr().out

    skills_after = await storage_view.repos.agent_skill.count()
    assert code == 0
    assert skills_after == skills_before + 1
    assert "phase 3 complete" in out

    rows = await storage_view.repos.agent_skill.find_where(
        "owner_id = 'agent1'", limit=10
    )
    assert len(rows) == 1
    assert rows[0].name == "summarise_doc"
    assert rows[0].cluster_id is not None
