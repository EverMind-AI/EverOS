"""End-to-end: emit pipeline event → strategies dispatch → SUCCESS + log lines."""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import uuid
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from everalgo.types import AgentCase, AtomicFact, ChatMessage, Foresight, MemCell
from structlog.testing import capture_logs

from everos.memory.events import (
    AgentCaseExtracted,
    AgentPipelineStarted,
    EpisodeExtracted,
    UserPipelineStarted,
)


class _DeterministicHashEmbedder:
    """Hash-seeded RNG embedder for clustering e2e.

    Same input text → same unit vector; distinct inputs → distinct directions
    (sha256-seeded ``numpy.random.default_rng``). The vectors aren't
    semantically meaningful, but they ARE deterministic and well-spread, so
    ``cluster_by_geometry`` / ``cluster_by_llm``'s nearest-neighbor logic
    has real signal to work with — unlike a MagicMock returning a constant
    vector, which collapses every cosine similarity to 1.0.
    """

    dim: int = 1024

    async def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        norm = float(np.linalg.norm(vec)) or 1.0
        vec /= norm
        return vec.tolist()

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def _sample_memcell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="alice likes hiking",
                timestamp=1_700_000_000_000,
                sender_id="u_alice",
            ),
            ChatMessage(
                id="m2",
                role="user",
                content="bob plans a trip",
                timestamp=1_700_000_001_000,
                sender_id="u_bob",
            ),
            ChatMessage(
                id="m3",
                role="assistant",
                content="sounds good",
                timestamp=1_700_000_002_000,
                sender_id="agent",
            ),
        ],
        timestamp=1_700_000_002_000,
    )


@pytest.mark.asyncio
async def test_emit_dispatches_both_strategies_to_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real OfflineEngine + APScheduler runtime; extractors + LLM mocked.

    Verifies the full chain: emit(event) → dispatcher (3 gates) → APS one-shot
    job → Runner.run → strategy body → mark_success.
    """
    import importlib

    from everos.core.persistence import MemoryRoot
    from everos.infra.ome.records import RunStatus

    svc = importlib.import_module("everos.service.memorize")

    # Redirect MemoryRoot.default() to tmp_path so _get_engine() writes ome.db
    # under the test's isolated temp directory instead of the real ~/.everos.
    monkeypatch.setattr(
        MemoryRoot,
        "default",
        classmethod(lambda cls: MemoryRoot(root=tmp_path)),
    )
    # Reset singletons so they rebuild against the patched MemoryRoot.
    monkeypatch.setattr(svc, "_ome_engine", None, raising=False)
    _af_mod = importlib.import_module("everos.memory.strategies.extract_atomic_facts")
    _fs_mod = importlib.import_module("everos.memory.strategies.extract_foresight")
    monkeypatch.setattr(_af_mod, "_writer", None, raising=False)
    monkeypatch.setattr(_fs_mod, "_writer", None, raising=False)

    fake_fact = AtomicFact(
        owner_id="u_alice", content="hi", timestamp=1_700_000_000_000
    )
    fake_foresight = Foresight(
        owner_id="u_alice",
        foresight="x",
        evidence="y",
        timestamp=1_700_000_000_000,
    )

    with (
        patch(
            "everos.memory.strategies.extract_atomic_facts.AtomicFactExtractor"
        ) as mock_af,
        patch(
            "everos.memory.strategies.extract_foresight.ForesightExtractor"
        ) as mock_fs,
        patch(
            "everos.memory.strategies.extract_atomic_facts.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_foresight.get_llm_client",
            return_value=object(),
        ),
        capture_logs() as logs,
    ):
        mock_af.return_value.aextract_from_text = AsyncMock(return_value=[fake_fact])
        mock_fs.return_value.aextract = AsyncMock(return_value=[fake_foresight])

        # Ensure the sqlite dir exists before the engine creates ome.db.
        (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
        (tmp_path / "ome.toml").write_text("# test\n")
        await _setup_system_db_schema(monkeypatch)

        engine = svc._get_engine()
        await engine.start()
        try:
            # Foresight still subscribes to UserPipelineStarted.
            await engine.emit(
                UserPipelineStarted(
                    memcell_id="mc_a",
                    session_id="s1",
                    memcell=_sample_memcell(),
                )
            )
            # Atomic facts now subscribes to EpisodeExtracted.
            await engine.emit(
                EpisodeExtracted(
                    memcell_id="mc_a",
                    episode_entry_id="ep_20260517_0001",
                    episode_text="alice likes hiking",
                    episode_timestamp_ms=1_700_000_000_000,
                    owner_id="u_alice",
                    session_id="s1",
                )
            )

            # Poll until both strategies reach SUCCESS (max 5 s).
            af_rows: list = []
            fs_rows: list = []
            for _ in range(50):
                await asyncio.sleep(0.1)
                af_rows = await engine.list_runs(
                    "extract_atomic_facts", status=RunStatus.SUCCESS
                )
                fs_rows = await engine.list_runs(
                    "extract_foresight", status=RunStatus.SUCCESS
                )
                if af_rows and fs_rows:
                    break

            assert af_rows, "expected SUCCESS RunRecord for extract_atomic_facts"
            assert fs_rows, "expected SUCCESS RunRecord for extract_foresight"
            assert af_rows[0].strategy_name == "extract_atomic_facts"
            assert fs_rows[0].strategy_name == "extract_foresight"
        finally:
            await engine.stop()
            await _teardown_system_db_schema()

    af_logs = [r for r in logs if r.get("event") == "atomic_facts_extracted"]
    fs_logs = [r for r in logs if r.get("event") == "foresights_extracted"]
    assert af_logs, "expected atomic_facts_extracted log line"
    assert fs_logs, "expected foresights_extracted log line"
    # extract_atomic_facts: 1 EpisodeExtracted → 1 fact for u_alice
    # extract_foresight: 2 senders × 1 foresight each = 2
    assert af_logs[0]["count"] == 1
    assert fs_logs[0]["count"] == 2


async def _setup_system_db_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rebuild the sqlite system.db engine + schema against the active tmp_path.

    The ``sqlite_manager`` engine is a process-wide singleton; without
    resetting it between tests the second e2e would reuse the first
    test's tmp engine (and miss the table create_all on this test's
    fresh tmp_path). ``SQLModel.metadata.create_all`` mirrors what
    :class:`SqliteLifespanProvider` runs at app startup.

    Pair with :func:`_teardown_system_db_schema` in the test's ``finally``
    block — the engine created here owns an aiosqlite worker thread that
    must be closed explicitly, or it lingers past the event loop and
    raises ``RuntimeError: Event loop is closed`` from the worker.
    """
    from sqlmodel import SQLModel

    from everos.infra.persistence.sqlite import sqlite_manager

    if sqlite_manager._engine is not None:
        await sqlite_manager.dispose_engine()
    monkeypatch.setattr(sqlite_manager, "_engine", None, raising=False)
    monkeypatch.setattr(sqlite_manager, "_session_factory", None, raising=False)
    engine = sqlite_manager.get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def _teardown_system_db_schema() -> None:
    """Dispose the per-test sqlite engine so its worker thread doesn't outlive
    the event loop (counterpart of :func:`_setup_system_db_schema`)."""
    from everos.infra.persistence.sqlite import sqlite_manager

    if sqlite_manager._engine is not None:
        await sqlite_manager.dispose_engine()


def _agent_memcell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="please summarise",
                timestamp=1_700_000_000_000,
                sender_id="u_alice",
            ),
            ChatMessage(
                id="m2",
                role="assistant",
                content="here's the summary",
                timestamp=1_700_000_001_000,
                sender_id="agent_42",
            ),
        ],
        timestamp=1_700_000_001_000,
    )


@pytest.mark.asyncio
async def test_emit_dispatches_agent_case_strategy_to_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror of the user-side e2e for the agent track.

    Verifies the full agent chain: AgentPipelineStarted emit → dispatcher
    (3 gates) → APS one-shot job → Runner.run → extract_agent_case body →
    mark_success. Catches breakage in event class wiring, trigger matching,
    engine registration, and the agent-side mock plumbing that unit tests
    bypass by calling the strategy function directly.
    """
    import importlib

    from everos.core.persistence import MemoryRoot
    from everos.infra.ome.records import RunStatus

    svc = importlib.import_module("everos.service.memorize")

    monkeypatch.setattr(
        MemoryRoot,
        "default",
        classmethod(lambda cls: MemoryRoot(root=tmp_path)),
    )
    monkeypatch.setattr(svc, "_ome_engine", None, raising=False)
    _ac_mod = importlib.import_module("everos.memory.strategies.extract_agent_case")
    monkeypatch.setattr(_ac_mod, "_writer", None, raising=False)

    fake_case = AgentCase(
        id=uuid.uuid4().hex,
        timestamp=1_700_000_001_000,
        task_intent="summarise the doc",
        approach="read + condense",
        quality_score=0.8,
        key_insight="",
    )

    with (
        patch(
            "everos.memory.strategies.extract_agent_case.AgentCaseExtractor"
        ) as mock_ac,
        patch(
            "everos.memory.strategies.extract_agent_case.get_llm_client",
            return_value=object(),
        ),
        capture_logs() as logs,
    ):
        mock_ac.return_value.aextract = AsyncMock(return_value=[fake_case])

        (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
        (tmp_path / "ome.toml").write_text("# test\n")
        await _setup_system_db_schema(monkeypatch)

        engine = svc._get_engine()
        await engine.start()
        try:
            await engine.emit(
                AgentPipelineStarted(
                    memcell_id="mc_a",
                    session_id="s1",
                    memcell=_agent_memcell(),
                )
            )

            ac_rows: list = []
            for _ in range(50):
                await asyncio.sleep(0.1)
                ac_rows = await engine.list_runs(
                    "extract_agent_case", status=RunStatus.SUCCESS
                )
                if ac_rows:
                    break

            assert ac_rows, "expected SUCCESS RunRecord for extract_agent_case"
            assert ac_rows[0].strategy_name == "extract_agent_case"
        finally:
            await engine.stop()
            await _teardown_system_db_schema()

    ac_logs = [r for r in logs if r.get("event") == "agent_case_extracted"]
    assert ac_logs, "expected agent_case_extracted log line"
    assert ac_logs[0]["owner_ids"] == ["agent_42"]
    assert ac_logs[0]["fanout"] == 1
    assert ac_logs[0]["quality_score"] == 0.8


@pytest.mark.asyncio
async def test_skill_chain_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chain: AgentCaseExtracted → trigger_skill_clustering (sqlite) →
    SkillClusterUpdated → extract_agent_skill → SUCCESS.

    Real ``cluster_by_llm`` algorithm path: hash-based deterministic
    embedder feeds the top-K nearest-neighbor stage, a ``FakeLLMClient``
    returns ``{"idx": "new"}`` so the algo picks the "brand-new cluster"
    branch — but the recall + skip-threshold + prompt-render + JSON-parse
    pipeline is all real. Only mocked: LanceDB reads (case + skill),
    ``AgentSkillExtractor`` (downstream extractor; out of scope), and
    the markdown writer.
    """
    import importlib
    from unittest.mock import MagicMock

    from everalgo.testing.fake_llm import FakeLLMClient
    from everalgo.types import AgentSkill as AlgoAgentSkill

    from everos.core.persistence import MemoryRoot
    from everos.infra.ome.records import RunStatus

    svc = importlib.import_module("everos.service.memorize")
    skill_mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")

    monkeypatch.setattr(
        MemoryRoot,
        "default",
        classmethod(lambda cls: MemoryRoot(root=tmp_path)),
    )
    monkeypatch.setattr(svc, "_ome_engine", None, raising=False)
    monkeypatch.setattr(skill_mod, "_writer", None, raising=False)

    embedder = _DeterministicHashEmbedder()
    # FakeLLMClient: cluster_by_llm only invokes it when top-K similarity
    # falls below llm_skip_threshold (default 0.85). With a single new
    # cluster in an empty owner set, the recall stage returns no candidates
    # at all — so the LLM is never asked. Provide a "{idx: new}" response
    # anyway as belt-and-suspenders for future scenarios with seeded clusters.
    fake_llm = FakeLLMClient(responses=['{"idx": "new"}'])

    target_lance = MagicMock()
    target_lance.entry_id = "ac_20260517_0001"
    target_lance.timestamp = _dt.datetime(2026, 5, 17, tzinfo=_dt.UTC)
    target_lance.task_intent = "summarise the doc"
    target_lance.approach = "read + condense"
    target_lance.quality_score = 0.8
    target_lance.key_insight = ""

    emitted_skill = AlgoAgentSkill(
        id=uuid.uuid4().hex,
        cluster_id="",
        name="summarise_doc",
        description="how to summarise docs",
        content="step 1: read; step 2: condense",
        confidence=0.7,
        maturity_score=0.5,
        source_case_ids=["ac_20260517_0001"],
    )

    with (
        patch(
            "everos.memory.strategies.trigger_skill_clustering.get_embedder",
            return_value=embedder,
        ),
        patch(
            "everos.memory.strategies.trigger_skill_clustering.get_llm_client",
            return_value=fake_llm,
        ),
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_case_repo"
        ) as mock_case_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_skill_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_agent_skill.AgentSkillExtractor"
        ) as mock_extractor_cls,
        patch(
            "everos.memory.strategies.extract_agent_skill.AgentSkillWriter"
        ) as mock_writer_cls,
        capture_logs() as logs,
    ):
        mock_case_repo.find_by_owner_entry = AsyncMock(return_value=target_lance)
        mock_case_repo.find_by_owner_entries = AsyncMock(return_value=[])
        # Empty cluster (no prior skills) → small-cluster scalar path.
        mock_skill_repo.count_in_cluster = AsyncMock(return_value=0)
        mock_skill_repo.find_in_cluster = AsyncMock(return_value=[])
        mock_extractor_cls.return_value.aextract = AsyncMock(
            return_value=[emitted_skill]
        )
        mock_writer_cls.return_value.write_main = AsyncMock(return_value=None)

        (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
        (tmp_path / "ome.toml").write_text("# test\n")
        await _setup_system_db_schema(monkeypatch)

        engine = svc._get_engine()
        await engine.start()
        try:
            await engine.emit(
                AgentCaseExtracted(
                    memcell_id="mc_a",
                    case_entry_id="ac_20260517_0001",
                    task_intent="summarise the doc",
                    quality_score=0.8,
                    case_timestamp_ms=1_700_000_001_000,
                    agent_id="agent_42",
                )
            )

            clu_rows: list = []
            skill_rows: list = []
            for _ in range(50):
                await asyncio.sleep(0.1)
                clu_rows = await engine.list_runs(
                    "trigger_skill_clustering", status=RunStatus.SUCCESS
                )
                skill_rows = await engine.list_runs(
                    "extract_agent_skill", status=RunStatus.SUCCESS
                )
                if clu_rows and skill_rows:
                    break

            assert clu_rows, "expected SUCCESS for trigger_skill_clustering"
            assert skill_rows, "expected SUCCESS for extract_agent_skill"
        finally:
            await engine.stop()
            await _teardown_system_db_schema()

    cluster_logs = [r for r in logs if r.get("event") == "skill_cluster_updated"]
    skill_logs = [r for r in logs if r.get("event") == "agent_skills_extracted"]
    assert cluster_logs, "expected skill_cluster_updated log line"
    assert skill_logs, "expected agent_skills_extracted log line"
    # Writer received exactly one SKILL.md write call with cluster_id stamped.
    write_args = mock_writer_cls.return_value.write_main.call_args
    fm = write_args.kwargs["frontmatter"]
    assert fm.cluster_id == cluster_logs[0]["cluster_id"]
    assert fm.name == "summarise_doc"


@pytest.mark.asyncio
async def test_profile_chain_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chain: EpisodeExtracted → trigger_profile_clustering (sqlite) →
    ProfileClusterUpdated → extract_user_profile → SUCCESS.

    Real ``cluster_by_geometry`` (cosine + time-window) with a hash-based
    deterministic embedder so the geometry stage operates on well-spread
    unit vectors. Real ``cluster_repo`` sqlite. ``memcell_repo`` is still
    mocked (a real memcell row would require the boundary stage to run
    first; out of scope for the chain emit test). ``ProfileExtractor`` /
    md reader/writer mocked as algo + IO seams.
    """
    import importlib
    from unittest.mock import MagicMock

    from everalgo.types import Profile as AlgoProfile

    from everos.core.persistence import MemoryRoot
    from everos.infra.ome.records import RunStatus

    svc = importlib.import_module("everos.service.memorize")
    profile_mod = importlib.import_module(
        "everos.memory.strategies.extract_user_profile"
    )

    monkeypatch.setattr(
        MemoryRoot,
        "default",
        classmethod(lambda cls: MemoryRoot(root=tmp_path)),
    )
    monkeypatch.setattr(svc, "_ome_engine", None, raising=False)
    monkeypatch.setattr(profile_mod, "_writer", None, raising=False)
    monkeypatch.setattr(profile_mod, "_reader", None, raising=False)

    embedder = _DeterministicHashEmbedder()

    fake_memcell_row = MagicMock()
    fake_memcell_row.memcell_id = "mc_aaaaaaaaaaa1"
    fake_memcell_row.payload_json = MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="alice likes hiking",
                timestamp=1_700_000_001_000,
                sender_id="u_alice",
            ),
        ],
        timestamp=1_700_000_001_000,
    ).model_dump_json()

    new_profile = AlgoProfile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "Alice is a hiker.",
            "timestamp": 1_700_000_001_000,
            "explicit_info": ["lives in tokyo"],
            "implicit_traits": [],
        }
    )

    fake_episode_row = MagicMock()
    fake_episode_row.parent_type = "memcell"
    fake_episode_row.parent_id = "mc_aaaaaaaaaaa1"
    fake_episode_row.entry_id = "ep_20260517_0001"

    with (
        patch(
            "everos.memory.strategies.trigger_profile_clustering.get_embedder",
            return_value=embedder,
        ),
        patch(
            "everos.memory.strategies.extract_user_profile.episode_repo"
        ) as mock_episode_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.memcell_repo"
        ) as mock_memcell_repo,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileReader"
        ) as mock_reader_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileWriter"
        ) as mock_writer_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileExtractor"
        ) as mock_extractor_cls,
        patch(
            "everos.memory.strategies.extract_user_profile.get_llm_client",
            return_value=object(),
        ),
        capture_logs() as logs,
    ):
        mock_episode_repo.find_by_owner_entries = AsyncMock(
            return_value=[fake_episode_row]
        )
        mock_memcell_repo.find_by_ids = AsyncMock(return_value=[fake_memcell_row])
        mock_reader_cls.return_value.read = AsyncMock(return_value=None)
        mock_writer_cls.return_value.write = AsyncMock(return_value=None)
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=new_profile)

        (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
        (tmp_path / "ome.toml").write_text("# test\n")
        await _setup_system_db_schema(monkeypatch)

        engine = svc._get_engine()
        await engine.start()
        try:
            await engine.emit(
                EpisodeExtracted(
                    memcell_id="mc_aaaaaaaaaaa1",
                    episode_entry_id="ep_20260517_0001",
                    episode_text="alice likes hiking",
                    episode_timestamp_ms=1_700_000_001_000,
                    owner_id="u_alice",
                    session_id="s_integration",
                )
            )

            clu_rows: list = []
            prof_rows: list = []
            for _ in range(50):
                await asyncio.sleep(0.1)
                clu_rows = await engine.list_runs(
                    "trigger_profile_clustering", status=RunStatus.SUCCESS
                )
                prof_rows = await engine.list_runs(
                    "extract_user_profile", status=RunStatus.SUCCESS
                )
                if clu_rows and prof_rows:
                    break

            assert clu_rows, "expected SUCCESS for trigger_profile_clustering"
            assert prof_rows, "expected SUCCESS for extract_user_profile"
        finally:
            await engine.stop()
            await _teardown_system_db_schema()

    cluster_logs = [r for r in logs if r.get("event") == "profile_cluster_updated"]
    profile_logs = [r for r in logs if r.get("event") == "user_profile_extracted"]
    assert cluster_logs, "expected profile_cluster_updated log line"
    assert profile_logs, "expected user_profile_extracted log line"
    assert profile_logs[0]["owner_id"] == "u_alice"
    assert profile_logs[0]["mode"] == "INIT"
