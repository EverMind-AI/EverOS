"""Integration test for ``everos cascade backfill --phase clusters``.

Exercises the real Phase 2 scan + synthetic-event-emission path (see
``memory.cascade._backfill._run_phase_clusters``): seeds real episode /
agent-case rows in LanceDB, drives ``run_backfill(phase="clusters", ...)``,
and asserts the phase synthesizes exactly one ``EpisodeExtracted`` /
``AgentCaseExtracted`` event per row and fans them into the OME engine.

The OME engine itself is replaced with an in-memory spy
(``_FakeClusterEngine``, injected via monkeypatching
``_backfill._build_cluster_engine``) so this test exercises the scan +
event-synthesis contract without paying for a real APScheduler + sqlite
jobstore runtime — that end-to-end wiring is already covered by
``tests/integration/test_ome_strategies_integration.py``.

Covers: empty DB (nothing to backfill, no engine touched), declined
confirmation (no events emitted, exit 1), a small DB with episodes +
agent cases (correct event types / counts / fields, exit 0), and a
missing embedding capability (exit 2, mirroring Phase 1's guard).

One test (``test_real_offline_engine_grows_cluster_count``) does NOT
install the fake spy: it drives Phase 2 through the actual
``_build_cluster_engine`` path — a real ``OfflineEngine`` + APScheduler
runtime with ``trigger_profile_clustering`` registered — and asserts
``cluster_repo.count()`` grows. This is the one check the fake-engine
tests above cannot make: they prove the scan + event-synthesis contract,
but never exercise whether the real engine wiring actually drives a
cluster into existence.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingCapability, EmbeddingProvider
from everos.component.utils.datetime import get_utc_now
from everos.config import load_settings
from everos.entrypoints.cli.commands._backfill_cmd import run_backfill
from everos.infra.ome.events import BaseEvent
from everos.infra.persistence.lancedb import (
    AgentCase,
    Episode,
    agent_case_repo,
    dispose_connection,
    episode_repo,
)
from everos.infra.persistence.sqlite import cluster_repo, sqlite_manager
from everos.memory.cascade import _backfill as backfill_mod
from everos.memory.events import AgentCaseExtracted, EpisodeExtracted

_DIM = 1024


class _StubEmbedder(EmbeddingProvider):
    dim = _DIM

    async def embed(self, text: str) -> list[float]:
        return [float(len(text) % 7)] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [[float(len(t) % 7)] * self.dim for t in texts]


class _FakeClusterEngine:
    """Spy replacing the real OME engine: records emitted events instead of
    dispatching them through APScheduler."""

    def __init__(self) -> None:
        self.emitted: list[BaseEvent] = []
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def emit(self, event: BaseEvent) -> None:
        self.emitted.append(event)

    async def wait_idle(self, *, timeout: float = 30.0) -> bool:  # noqa: ASYNC109
        return True

    async def stop(self) -> None:
        self.stopped = True


@pytest.fixture
async def backfill_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Path]:
    """Tmp memory root + stub embedder + isolated sqlite engine.

    Phase 2 touches both LanceDB (episode / agent_case scan) and sqlite
    (``cluster_repo.count()``), so both process-wide singletons are
    disposed and rebuilt against ``tmp_path`` around each test — mirrors
    ``backfill_runtime`` in ``test_backfill_phase1.py`` plus the sqlite
    reset from ``test_ome_strategies_integration.py``.
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

    yield tmp_path

    await dispose_connection()
    if sqlite_manager._engine is not None:
        await sqlite_manager.dispose_engine()


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> _FakeClusterEngine:
    """Install the spy engine in place of ``_build_cluster_engine``."""
    engine = _FakeClusterEngine()
    monkeypatch.setattr(backfill_mod, "_build_cluster_engine", lambda: engine)
    return engine


def _episode(entry_id: str, *, owner_id: str = "u1") -> Episode:
    return Episode(
        id=f"{owner_id}_{entry_id}",
        entry_id=entry_id,
        owner_id=owner_id,
        owner_type="user",
        session_id="s1",
        timestamp=get_utc_now(),
        parent_id=f"mc_{entry_id}",
        sender_ids=[owner_id],
        episode=f"episode body {entry_id}",
        episode_tokens=f"episode body {entry_id}",
        md_path="users/u1/episodes/episode-2026-01-01.md",
        content_sha256=hashlib.sha256(entry_id.encode()).hexdigest(),
        vector=[0.1] * _DIM,
    )


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
        md_path="agents/agent1/.cases/agent_case-2026-01-01.md",
        content_sha256=hashlib.sha256(entry_id.encode()).hexdigest(),
        vector=[0.1] * _DIM,
    )


async def test_empty_db_reports_nothing_to_backfill(
    backfill_runtime: Path,
    fake_engine: _FakeClusterEngine,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = await run_backfill(phase="clusters", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 0
    assert "Nothing to backfill" in out
    assert fake_engine.emitted == []
    assert fake_engine.started is False


async def test_declined_confirmation_emits_nothing_and_exits_one(
    backfill_runtime: Path,
    fake_engine: _FakeClusterEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await episode_repo.add([_episode("ep1")])
    monkeypatch.setattr("typer.confirm", lambda *a, **k: False)

    code = await run_backfill(phase="clusters", auto_yes=False)

    assert code == 1
    assert fake_engine.emitted == []
    assert fake_engine.started is False


async def test_episodes_and_cases_synthesize_matching_events(
    backfill_runtime: Path,
    fake_engine: _FakeClusterEngine,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await episode_repo.add([_episode("ep1"), _episode("ep2")])
    await agent_case_repo.add([_agent_case("ac1")])

    code = await run_backfill(phase="clusters", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 0
    assert fake_engine.started is True
    assert fake_engine.stopped is True
    assert len(fake_engine.emitted) == 3

    episode_events = [e for e in fake_engine.emitted if isinstance(e, EpisodeExtracted)]
    case_events = [e for e in fake_engine.emitted if isinstance(e, AgentCaseExtracted)]
    assert len(episode_events) == 2
    assert len(case_events) == 1

    assert all(e.source == "pipeline" for e in episode_events)
    assert all(e.event_id.startswith("backfill_") for e in fake_engine.emitted)
    assert {e.owner_id for e in episode_events} == {"u1"}
    assert case_events[0].agent_id == "agent1"
    assert case_events[0].task_intent == "task intent ac1"
    assert case_events[0].quality_score == pytest.approx(0.8)

    assert "phase 2 complete" in out


async def test_missing_embedding_capability_returns_exit_two(
    backfill_runtime: Path,
    fake_engine: _FakeClusterEngine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors Phase 1's own guard: no embedding provider configured means
    the strategies can't re-embed, so the phase fails clean with exit 2
    rather than emitting events into strategies that will crash."""
    import everos.component.embedding.accessor as acc

    await episode_repo.add([_episode("ep1")])
    monkeypatch.setattr(acc, "_capability", EmbeddingCapability(provider=None))

    code = await run_backfill(phase="clusters", auto_yes=True)

    assert code == 2
    assert fake_engine.emitted == []
    assert fake_engine.started is False


async def test_reflection_merged_episodes_excluded_from_synthesis(
    backfill_runtime: Path,
    fake_engine: _FakeClusterEngine,
) -> None:
    """Reflection-merged episodes (``parent_type="cluster"``) must not be
    re-synthesized as pipeline events.

    ``trigger_profile_clustering`` excludes Reflection output via
    ``applies_to=lambda e: e.source == "pipeline"`` — but that guard is
    defeated if the Episode scan hands it a synthetic event whose
    ``memcell_id`` is actually a cluster id (see Important 1 review
    finding on ``_backfill._scan_all_rows`` / ``_run_phase_clusters``).
    Only the pipeline-sourced row should reach the fake engine.
    """
    pipeline_episode = _episode("ep1")
    merged_episode = _episode("ep_merged").model_copy(
        update={
            "id": "u1_ep_merged",
            "parent_type": "cluster",
            "parent_id": "cluster_abc123",
        }
    )
    await episode_repo.add([pipeline_episode, merged_episode])

    code = await run_backfill(phase="clusters", auto_yes=True)

    assert code == 0
    episode_events = [e for e in fake_engine.emitted if isinstance(e, EpisodeExtracted)]
    assert len(episode_events) == 1
    assert episode_events[0].memcell_id == pipeline_episode.parent_id


async def test_real_offline_engine_grows_cluster_count(
    backfill_runtime: Path,
) -> None:
    """Phase 2 through the REAL ``_build_cluster_engine`` — no fake spy.

    Deliberately does not use the ``fake_engine`` fixture: it drives an
    actual ``OfflineEngine`` + APScheduler runtime with
    ``trigger_profile_clustering`` registered (see
    ``_backfill._build_cluster_engine``), proving the phase's own engine
    wiring — not just the scan/synthesis contract the other tests in
    this module pin — actually grows ``cluster_repo.count()``.

    ``trigger_profile_clustering`` re-embeds each episode's text via
    ``get_embedding_capability().require()`` — which ``backfill_runtime``
    already stubs with the deterministic ``_StubEmbedder``, so no
    per-strategy patch is needed here. No agent cases are seeded, so
    ``trigger_skill_clustering`` (the engine's other registered
    strategy, whose ``cluster_by_llm`` fallback needs an LLM) never
    fires, keeping this test LLM-free.
    """
    await episode_repo.add([_episode(f"ep{i}") for i in range(5)])

    # ``run_backfill`` itself creates the ``cluster`` table lazily (see
    # ``_ensure_cluster_schema``) — call it here too so the "before" count
    # can be taken without racing that lazy create.
    await backfill_mod._ensure_cluster_schema()
    clusters_before = await cluster_repo.count()
    assert clusters_before == 0

    code = await run_backfill(phase="clusters", auto_yes=True)

    clusters_after = await cluster_repo.count()
    assert code == 0
    assert clusters_after > clusters_before
