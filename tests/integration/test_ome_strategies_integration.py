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
from everalgo.types import ChatMessage, MemCell
from structlog.testing import capture_logs

from everos.config import Settings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.factories import build_indexes, build_stores
from everos.memory.deps import MemoryDeps
from everos.memory.events import (
    AgentCaseExtracted,
    AgentPipelineStarted,
    EpisodeExtracted,
)
from everos.memory.models import Episode
from everos.memory.ports import MemoryRef, Scope, Stored
from everos.service.bootstrap import _build_engine
from tests.fakes import (
    InMemoryAgentSkillStore,
    InMemoryEpisodeIndex,
    InMemoryEpisodeStore,
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


@pytest.mark.asyncio
async def test_emit_dispatches_agent_case_strategy_to_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> None:
    """The agent track end to end.

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
            MemoryRoot(root=tmp_path), _installed["deps"], Settings()
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
            MemoryRoot(root=tmp_path), _installed["deps"], Settings()
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
    """Chain: EpisodeExtracted → trigger_profile_clustering (sqlite) and,
    from the same event, extract_user_profile → SUCCESS for both.

    The profile update reads the event's segment itself; the clustering still
    runs (Reflection reads the clusters) and no longer drives the profile.
    Real ``cluster_by_geometry`` (cosine + time-window) with a hash-based
    deterministic embedder so the geometry stage operates on well-spread
    unit vectors. Real sqlite cluster store. The memcell store and the
    episode index are in-memory and seeded (a real memcell row would require
    the boundary stage to run first; out of scope for the chain test), the
    profile store is in-memory and ``ProfileExtractor`` is patched.
    """

    from everalgo.types import Profile as AlgoProfile
    from everalgo.user_memory.profile import ItemOrigin, ProfileUpdate

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
    episode = Episode(
        owner_id="u_alice",
        episode="alice likes hiking",
        summary="",
        timestamp=1_700_000_001_000,
        parent_id="mc_aaaaaaaaaaa1",
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
            item=episode,
        )
    )
    # The episode is in md too: clustering re-checks it is still there.
    episodes = InMemoryEpisodeStore()
    episodes.seed(alice, "ep_20260517_0001", episode)

    with (
        patch(
            "everos.memory.strategies._profiles.ProfileExtractor"
        ) as mock_extractor_cls,
        capture_logs() as logs,
    ):
        _install_deps(
            storage,
            monkeypatch,
            tmp_path,
            embedding=embedder,
            stores={
                "memcell": memcells,
                "profile": InMemoryProfileStore(),
                "episode": episodes,
            },
            indexes={"episode": episode_index},
        )
        mock_extractor_cls.return_value.aupdate = AsyncMock(
            return_value=ProfileUpdate(
                profile=new_profile,
                explicit_info=(
                    ItemOrigin(change="added", sources=("mc_aaaaaaaaaaa1",)),
                ),
                implicit_traits=(),
            )
        )

        (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
        (tmp_path / "ome.toml").write_text("# test\n")
        await storage.create_schema()

        engine = _build_engine(
            MemoryRoot(root=tmp_path), _installed["deps"], Settings()
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
    assert profile_logs[0]["mode"] == "FIRST"
