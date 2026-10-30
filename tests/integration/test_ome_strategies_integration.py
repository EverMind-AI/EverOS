"""End-to-end: emit pipeline event → strategies dispatch → SUCCESS + log lines."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from everalgo.types import AtomicFact, ChatMessage, Foresight, MemCell
from structlog.testing import capture_logs

from everos.config import OmeSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.factories import build_indexes, build_stores
from everos.memory.deps import MemoryDeps
from everos.memory.events import (
    AgentCaseExtracted,
    AgentPipelineStarted,
    EpisodeExtracted,
    UserPipelineStarted,
)
from everos.memory.models import Episode
from everos.memory.ports import MemoryRef, Scope, Stored
from everos.service.bootstrap import _build_engine
from tests.fakes import (
    InMemoryAgentSkillStore,
    InMemoryEpisodeIndex,
    InMemoryMemcellStore,
    InMemoryProfileStore,
    agent_track_llm,
    agent_track_script,
    agent_trajectory,
    bundled_instructions,
    make_indexes,
    unscripted_llm,
)
from tests.fakes.storage import Storage


class _DeterministicHashEmbedder:
    """Hash-seeded RNG embedder for clustering e2e.

    Same input text → same unit vector; distinct inputs → distinct directions
    (sha256-seeded ``numpy.random.default_rng``). The vectors aren't
    semantically meaningful, but they ARE deterministic and well-spread, so
    ``cluster_by_geometry`` / ``acluster_by_llm``'s nearest-neighbor logic
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


_installed: dict[str, MemoryDeps] = {}


def _install_deps(
    storage: Storage,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    llm: object | None = None,
    embedding: object | None = None,
    stores: dict[str, object] | None = None,
    indexes: dict[str, object] | None = None,
) -> MemoryDeps:
    """Bind the deps the application engine registers its strategies with.

    Production stores over ``storage`` (markdown + the sqlite system db),
    with single stores replaced where a test stands in for them; ``indexes``
    switches to in-memory indexes (with the given replacements).
    """
    real = build_stores(storage.root, storage.sqlite)
    deps = MemoryDeps(
        llm=llm if llm is not None else unscripted_llm(),
        embedding=embedding,
        stores=dataclasses.replace(real, **(stores or {})),
        indexes=(
            make_indexes(**indexes)
            if indexes is not None
            else build_indexes(storage.index)
        ),
        instructions=bundled_instructions(),
    )
    _installed["deps"] = deps
    return deps


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> None:
    """Real OfflineEngine + APScheduler runtime; extractors + LLM mocked.

    Verifies the full chain: emit(event) → dispatcher (3 gates) → APS one-shot
    job → Runner.run → strategy body → mark_success.
    """

    from everos.infra.ome.records import RunStatus

    # Redirect MemoryRoot.resolve() to tmp_path so the engine writes ome.db
    # under the test's isolated temp directory instead of the real ~/.everos.
    monkeypatch.setattr(
        MemoryRoot,
        "resolve",
        classmethod(lambda cls: MemoryRoot(root=tmp_path)),
    )

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
        capture_logs() as logs,
    ):
        _install_deps(storage, monkeypatch, tmp_path)
        mock_af.return_value.aextract_from_text = AsyncMock(return_value=[fake_fact])
        mock_fs.return_value.aextract = AsyncMock(return_value=[fake_foresight])

        # Ensure the sqlite dir exists before the engine creates ome.db.
        (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
        # `extract_foresight` now ships disabled (see its module docstring).
        # This test needs a `UserPipelineStarted` subscriber to cover the
        # second trigger route, so it opts back in through the very `ome.toml`
        # key that docstring points users at — which makes the opt-in path
        # itself covered, rather than working around the new default.
        (tmp_path / "ome.toml").write_text(
            "[strategies.extract_foresight]\nenabled = true\n"
        )
        await storage.create_schema()

        engine = _build_engine(
            MemoryRoot(root=tmp_path), _installed["deps"], OmeSettings()
        )
        await engine.start()
        try:
            # `ConfigReloader.start()` fires its initial load as a task, so
            # `engine.start()` returns before `ome.toml` has been applied.
            # An emit inside that window is judged against the coded defaults
            # and silently dropped by the enabled gate — the event is not
            # redelivered once the override lands. Wait for the override to
            # be visible in the registry before emitting.
            for _ in range(50):
                if any(
                    m.name == "extract_foresight" and m.enabled
                    for m in engine._registry.all()
                ):
                    break
                await asyncio.sleep(0.1)
            else:
                pytest.fail("ome.toml override never reached the registry")

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
            await storage.database.dispose()

    af_logs = [r for r in logs if r.get("event") == "atomic_facts_extracted"]
    fs_logs = [r for r in logs if r.get("event") == "foresights_extracted"]
    assert af_logs, "expected atomic_facts_extracted log line"
    assert fs_logs, "expected foresights_extracted log line"
    # extract_atomic_facts: 1 EpisodeExtracted → 1 fact for u_alice
    # extract_foresight: 2 senders × 1 foresight each = 2
    assert af_logs[0]["count"] == 1
    assert fs_logs[0]["count"] == 2


@pytest.mark.asyncio
async def test_emit_dispatches_agent_case_strategy_to_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> None:
    """Mirror of the user-side e2e for the agent track.

    Verifies the full agent chain: AgentPipelineStarted emit → dispatcher
    (3 gates) → APS one-shot job → Runner.run → extract_agent_case body →
    mark_success. Catches breakage in event class wiring, trigger matching,
    engine registration, and the agent-side mock plumbing that unit tests
    bypass by calling the strategy function directly.
    """

    from everos.infra.ome.records import RunStatus

    monkeypatch.setattr(
        MemoryRoot,
        "resolve",
        classmethod(lambda cls: MemoryRoot(root=tmp_path)),
    )

    with capture_logs() as logs:
        _install_deps(storage, monkeypatch, tmp_path, llm=agent_track_llm())

        (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
        (tmp_path / "ome.toml").write_text("# test\n")
        await storage.create_schema()

        engine = _build_engine(
            MemoryRoot(root=tmp_path), _installed["deps"], OmeSettings()
        )
        await engine.start()
        try:
            await engine.emit(
                AgentPipelineStarted(
                    memcell_id="mc_a", session_id="s1", memcell=agent_trajectory()
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
            await storage.database.dispose()

    ac_logs = [r for r in logs if r.get("event") == "agent_case_extracted"]
    assert ac_logs, "expected agent_case_extracted log line"
    assert ac_logs[0]["owner_ids"] == ["agent_42"]
    assert ac_logs[0]["fanout"] == 1
    assert ac_logs[0]["quality_score"] == 0.8


@pytest.mark.asyncio
async def test_skill_chain_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> None:
    """Chain: AgentCaseExtracted → trigger_skill_clustering (sqlite) →
    SkillClusterUpdated → extract_agent_skill → SUCCESS.

    Real ``acluster_by_llm`` and ``AgentSkillExtractor``: a hash-based
    deterministic embedder feeds the top-K nearest-neighbor stage (no
    existing cluster, so a new one is kept without asking the model) and
    the extractor answers from the canned agent-track script. Stood in: the
    index reads (case + skill, in-memory and empty) and the skill store
    (in-memory).
    """

    from everos.infra.ome.records import RunStatus

    monkeypatch.setattr(
        MemoryRoot,
        "resolve",
        classmethod(lambda cls: MemoryRoot(root=tmp_path)),
    )

    embedder = _DeterministicHashEmbedder()
    # With no existing cluster the clustering operator never asks the model;
    # the skill extractor answers from the canned agent-track script.
    skill_store = InMemoryAgentSkillStore()
    with capture_logs() as logs:
        # Index reads are in-memory and empty: no prior skills, no lineage.
        _install_deps(
            storage,
            monkeypatch,
            tmp_path,
            llm=agent_track_llm(),
            embedding=embedder,
            stores={"agent_skill": skill_store},
            indexes={},
        )

        (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
        (tmp_path / "ome.toml").write_text("# test\n")
        await storage.create_schema()

        engine = _build_engine(
            MemoryRoot(root=tmp_path), _installed["deps"], OmeSettings()
        )
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
            await storage.database.dispose()

    cluster_logs = [r for r in logs if r.get("event") == "skill_cluster_updated"]
    skill_logs = [r for r in logs if r.get("event") == "agent_skills_extracted"]
    assert cluster_logs, "expected skill_cluster_updated log line"
    assert skill_logs, "expected agent_skills_extracted log line"
    # Exactly one skill written, with the cluster id stamped.
    agent = Scope(
        app_id="default", project_id="default", owner_id="agent_42", owner_type="agent"
    )
    (written,) = skill_store.skills[agent].values()
    assert written.cluster_id == cluster_logs[0]["cluster_id"]
    want = json.loads(agent_track_script("skill_extract"))["operations"][0]["data"]
    assert written.name == want["name"]


@pytest.mark.asyncio
async def test_profile_chain_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> None:
    """Chain: EpisodeExtracted → trigger_profile_clustering (sqlite) →
    ProfileClusterUpdated → extract_user_profile → SUCCESS.

    Real ``cluster_by_geometry`` (cosine + time-window) with a hash-based
    deterministic embedder so the geometry stage operates on well-spread
    unit vectors. Real sqlite cluster store. The memcell store and the
    episode index are in-memory and seeded (a real memcell row would require
    the boundary stage to run first; out of scope for the chain test), the
    profile store is in-memory and ``ProfileExtractor`` is patched.
    """

    from everalgo.types import Profile as AlgoProfile

    from everos.infra.ome.records import RunStatus

    monkeypatch.setattr(
        MemoryRoot,
        "resolve",
        classmethod(lambda cls: MemoryRoot(root=tmp_path)),
    )

    embedder = _DeterministicHashEmbedder()

    memcells = InMemoryMemcellStore()
    memcells.seed(
        "mc_aaaaaaaaaaa1",
        MemCell(
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
        ),
    )

    new_profile = AlgoProfile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "Alice is a hiker.",
            "timestamp": 1_700_000_001_000,
            "explicit_info": [{"description": "lives in tokyo"}],
            "implicit_traits": [],
        }
    )

    alice = Scope(
        app_id="default", project_id="default", owner_id="u_alice", owner_type="user"
    )
    episode_index = InMemoryEpisodeIndex()
    episode_index.seed(
        Stored(
            ref=MemoryRef(
                id="u_alice_ep_20260517_0001",
                entry_id="ep_20260517_0001",
                md_path="ep.md",
            ),
            scope=alice,
            item=Episode(
                owner_id="u_alice",
                episode="alice likes hiking",
                summary="",
                timestamp=1_700_000_001_000,
                parent_id="mc_aaaaaaaaaaa1",
            ),
        )
    )

    with (
        patch(
            "everos.memory.strategies.extract_user_profile.ProfileExtractor"
        ) as mock_extractor_cls,
        capture_logs() as logs,
    ):
        _install_deps(
            storage,
            monkeypatch,
            tmp_path,
            embedding=embedder,
            stores={"memcell": memcells, "profile": InMemoryProfileStore()},
            indexes={"episode": episode_index},
        )
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=new_profile)

        (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
        (tmp_path / "ome.toml").write_text("# test\n")
        await storage.create_schema()

        engine = _build_engine(
            MemoryRoot(root=tmp_path), _installed["deps"], OmeSettings()
        )
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
            await storage.database.dispose()

    cluster_logs = [r for r in logs if r.get("event") == "profile_cluster_updated"]
    profile_logs = [r for r in logs if r.get("event") == "user_profile_extracted"]
    assert cluster_logs, "expected profile_cluster_updated log line"
    assert profile_logs, "expected user_profile_extracted log line"
    assert profile_logs[0]["owner_id"] == "u_alice"
    assert profile_logs[0]["mode"] == "INIT"
