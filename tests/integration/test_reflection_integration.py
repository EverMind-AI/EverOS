"""End-to-end Reflection INIT cycle integration test.

Drives ``ReflectionOrchestrator.run()`` with real SQLite (cluster +
report repos), real LanceDB (episode + atomic_fact stores), real
EpisodeWriter (md files), a ``FakeLLMClient``-backed
``EpisodeReflector``, and a stub embedder. Verifies the full flow:

    select candidates → merge episodes (FakeLLM) → write merged md →
    return EpisodeExtracted (the fact extraction it triggers is not run
    here) → finish: deprecate originals → update cluster membership →
    write report

White-box surfaces: sqlite cluster_member / reflection_report tables,
LanceDB episode.deprecated_by column, md file existence + frontmatter.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
import pytest
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.user_memory.reflect import EpisodeReflector

from everos.config import LanceDBSettings, load_settings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.index.adapters import (
    AtomicFactIndexAdapter,
    EpisodeIndexAdapter,
)
from everos.infra.persistence.lancedb import (
    LanceIndexContract,
    LanceRepoBase,
    open_lancedb_connection,
)
from everos.infra.persistence.lancedb.base import LanceDailyLogRepoBase
from everos.infra.persistence.markdown import (
    MarkdownAtomicFactStore,
    MarkdownEpisodeStore,
)
from everos.infra.persistence.records.atomic_fact import AtomicFact
from everos.infra.persistence.records.episode import Episode as LanceEpisode
from everos.infra.persistence.sqlite import (
    SqliteClusterStore,
    SqliteReflectionReportStore,
)
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.events import EpisodeExtracted
from everos.memory.models import ReflectionReport
from everos.memory.ports import Scope
from everos.memory.reflection import ReflectionFinisher, ReflectionOrchestrator
from tests.fakes.storage import Storage

# ---------------------------------------------------------------------------
# Stub embedder
# ---------------------------------------------------------------------------


class _StubEmbedder:
    """Return deterministic 1024-dim vectors seeded by input text."""

    dim: int = 1024

    async def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype(np.float32)
        norm = float(np.linalg.norm(vec)) or 1.0
        vec /= norm
        return vec.tolist()


type _Reflect = Callable[
    [Scope], Awaitable[tuple[list[EpisodeExtracted], list[ReflectionReport]]]
]


def _reflection(
    storage: Storage,
    memory_root: MemoryRoot,
    ep_repo: object,
    af_repo: object,
    reflector: object,
) -> _Reflect:
    """Run both Reflection steps the way the two strategies chain them.

    ``reflect_episodes`` merges and returns ``EpisodeExtracted``; after the
    merged episode's facts are extracted (not run here), ``finish_reflection``
    retires the members. Returns the merge events and the finish reports.
    """
    clusters = SqliteClusterStore(storage.sqlite.cluster)
    episodes = MarkdownEpisodeStore(memory_root)
    reports_store = SqliteReflectionReportStore(storage.sqlite.reflection_report)
    orchestrator = ReflectionOrchestrator(
        cluster_store=clusters,
        episode_store=episodes,
        episode_index=EpisodeIndexAdapter(ep_repo),
        report_store=reports_store,
        reflector=reflector,
    )
    finisher = ReflectionFinisher(
        cluster_store=clusters,
        episode_store=episodes,
        atomic_fact_store=MarkdownAtomicFactStore(memory_root),
        atomic_fact_index=AtomicFactIndexAdapter(af_repo),
        report_store=reports_store,
        embedder=_StubEmbedder(),
    )

    async def run(
        scope: Scope,
    ) -> tuple[list[EpisodeExtracted], list[ReflectionReport]]:
        merged = await orchestrator.run(scope=scope)
        reports = []
        for event in merged:
            assert event.reflection is not None
            report = await finisher.finish(
                scope=scope,
                merged_entry_id=event.episode_entry_id,
                context=event.reflection,
            )
            if report is not None:
                reports.append(report)
        return merged, reports

    return run


# ---------------------------------------------------------------------------
# LanceDB repo wrappers (inject table directly)
# ---------------------------------------------------------------------------


class _EpisodeRepo(
    LanceIndexContract[LanceEpisode], LanceDailyLogRepoBase[LanceEpisode]
):
    schema = LanceEpisode


class _AtomicFactRepo(
    LanceIndexContract[AtomicFact], LanceDailyLogRepoBase[AtomicFact]
):
    schema = AtomicFact


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_locks() -> None:
    """Drop per-table write locks + partition locks between tests."""
    LanceRepoBase._reset_locks_for_tests()
    _reset_for_tests()


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ome.toml").write_text("# test\n")
    return mr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_scope(owner_id: str) -> Scope:
    return Scope(
        app_id="default", project_id="default", owner_id=owner_id, owner_type="user"
    )


def _make_lance_episode(
    *,
    entry_id: str,
    owner_id: str,
    episode: str,
    timestamp: _dt.datetime,
    parent_type: str = "memcell",
    parent_id: str,
    session_id: str | None = "s_test",
    md_path: str = "",
) -> LanceEpisode:
    """Build a LanceDB Episode row with required fields."""
    digest = hashlib.sha256(episode.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(1024).astype(np.float32)
    vec /= float(np.linalg.norm(vec)) or 1.0

    return LanceEpisode(
        id=f"{owner_id}_{entry_id}",
        entry_id=entry_id,
        owner_id=owner_id,
        owner_type="user",
        app_id="default",
        project_id="default",
        session_id=session_id,
        timestamp=timestamp,
        parent_type=parent_type,
        parent_id=parent_id,
        sender_ids=[owner_id],
        subject="test",
        summary=None,
        episode=episode,
        episode_tokens=episode.lower(),
        md_path=md_path,
        content_sha256=hashlib.sha256(episode.encode()).hexdigest(),
        deprecated_by=None,
        vector=vec.tolist(),
    )


def _make_lance_fact(
    *,
    entry_id: str,
    owner_id: str,
    fact: str,
    parent_id: str,
    parent_type: str = "memcell",
    timestamp: _dt.datetime,
) -> AtomicFact:
    """Build a LanceDB AtomicFact row."""
    digest = hashlib.sha256(fact.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(1024).astype(np.float32)
    vec /= float(np.linalg.norm(vec)) or 1.0

    return AtomicFact(
        id=f"{owner_id}_{entry_id}",
        entry_id=entry_id,
        owner_id=owner_id,
        owner_type="user",
        app_id="default",
        project_id="default",
        session_id="s_test",
        timestamp=timestamp,
        parent_type=parent_type,
        parent_id=parent_id,
        sender_ids=[owner_id],
        fact=fact,
        fact_tokens=fact.lower(),
        md_path="",
        content_sha256=hashlib.sha256(fact.encode()).hexdigest(),
        deprecated_by=None,
        vector=vec.tolist(),
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reflection_init_merges_cluster_episodes(
    tmp_path: Path,
    memory_root: MemoryRoot,
    monkeypatch: pytest.MonkeyPatch,
    storage: Storage,
) -> None:
    """Full INIT Reflection cycle: 3 episodes in a cluster -> merge -> verify.

    Verifies:
        - ReflectionReport created with mode="init", source_count=3
        - Source episodes deprecated in LanceDB (deprecated_by is set)
        - Merged episode written to md with parent_type="cluster"
        - Cluster members updated: old 3 removed, 1 new episode member
        - Report.merged_entry_id matches the new cluster member's member_id
        - Atomic facts for source episodes are deprecated
    """
    # -- Redirect MemoryRoot.resolve() to tmp_path.
    monkeypatch.setattr(
        MemoryRoot,
        "resolve",
        classmethod(lambda cls: MemoryRoot(root=tmp_path)),
    )
    monkeypatch.setenv("EVEROS_LLM__API_KEY", "fake-key")
    monkeypatch.setenv("EVEROS_LLM__BASE_URL", "https://fake.example.com")

    load_settings.cache_clear()

    await storage.create_schema()

    try:
        # -- LanceDB setup: create tables + insert test episodes.
        conn = await open_lancedb_connection(memory_root.lancedb_dir, LanceDBSettings())
        ep_table = await conn.create_table("episode", schema=LanceEpisode)
        af_table = await conn.create_table("atomic_fact", schema=AtomicFact)

        ep_repo = _EpisodeRepo(table=ep_table)
        af_repo = _AtomicFactRepo(table=af_table)

        owner_id = "u_test"
        cluster_id = "cl_test_reflect"

        ts1 = _dt.datetime(2026, 6, 10, 10, 0, 0, tzinfo=_dt.UTC)
        ts2 = _dt.datetime(2026, 6, 10, 11, 0, 0, tzinfo=_dt.UTC)
        ts3 = _dt.datetime(2026, 6, 10, 12, 0, 0, tzinfo=_dt.UTC)

        ep1 = _make_lance_episode(
            entry_id="ep_20260610_0001",
            owner_id=owner_id,
            episode="Andrew has no pets",
            timestamp=ts1,
            parent_id="mc_001",
        )
        ep2 = _make_lance_episode(
            entry_id="ep_20260610_0002",
            owner_id=owner_id,
            episode="Andrew adopted Toby",
            timestamp=ts2,
            parent_id="mc_002",
        )
        ep3 = _make_lance_episode(
            entry_id="ep_20260610_0003",
            owner_id=owner_id,
            episode="Andrew adopted Buddy",
            timestamp=ts3,
            parent_id="mc_003",
        )

        await ep_repo.add([ep1, ep2, ep3])

        # Insert atomic facts linked to the source memcells.
        fact1 = _make_lance_fact(
            entry_id="af_20260610_0001",
            owner_id=owner_id,
            fact="Andrew has no pets",
            parent_id="ep_20260610_0001",
            timestamp=ts1,
        )
        fact2 = _make_lance_fact(
            entry_id="af_20260610_0002",
            owner_id=owner_id,
            fact="Andrew adopted Toby",
            parent_id="ep_20260610_0002",
            timestamp=ts2,
        )
        fact3 = _make_lance_fact(
            entry_id="af_20260610_0003",
            owner_id=owner_id,
            fact="Andrew adopted Buddy",
            parent_id="ep_20260610_0003",
            timestamp=ts3,
        )
        await af_repo.add([fact1, fact2, fact3])

        # -- SQLite: create cluster + cluster_members.
        centroid = np.zeros(1024, dtype=np.float32)
        algo_cluster = AlgoCluster(
            id=cluster_id,
            centroid=centroid,
            count=3,
            last_ts=int(ts3.timestamp() * 1000),
            preview=["Andrew has no pets", "Andrew adopted Toby"],
            members=["ep_20260610_0001", "ep_20260610_0002", "ep_20260610_0003"],
        )
        await storage.sqlite.cluster.upsert_with_members(
            algo_cluster,
            owner_id=owner_id,
            owner_type="user",
            kind="user_memory",
            member_type="episode",
        )

        # Verify cluster members are created.
        members_before = await storage.sqlite.cluster.get_members_with_type(cluster_id)
        assert len(members_before) == 3

        # -- Build the EpisodeReflector with FakeLLM.
        merged_content = (
            "Andrew initially had no pets. He later adopted a dog named Toby, "
            "and then adopted another dog named Buddy."
        )
        merged_title = "Andrew's pet adoption journey"
        reflect_response = json.dumps(
            {"content": merged_content, "title": merged_title}
        )
        fake_llm = FakeLLMClient(responses=[reflect_response])
        reflector = EpisodeReflector(llm=fake_llm)

        # -- Build the orchestrator with real repos.
        orchestrator = _reflection(storage, memory_root, ep_repo, af_repo, reflector)

        # -- Run the orchestrator.
        emitted, reports = await orchestrator(_user_scope(owner_id))

        # -- Verify: exactly one report created.
        assert len(reports) == 1
        report = reports[0]
        assert report.mode == "init"
        assert report.source_count == 3
        assert report.status == "completed"
        assert report.cluster_id == cluster_id

        merged_entry_id = report.merged_entry_id

        # -- Verify: source episodes deprecated in LanceDB.
        for ep in [ep1, ep2, ep3]:
            rows = await ep_repo.find_where(
                f"entry_id = '{ep.entry_id}' AND owner_id = '{owner_id}'"
            )
            assert len(rows) == 1, f"expected 1 row for {ep.entry_id}"
            assert rows[0].deprecated_by == merged_entry_id, (
                f"{ep.entry_id} should be deprecated by {merged_entry_id}"
            )

        # -- Verify: atomic facts deprecated in LanceDB.
        for fact in [fact1, fact2, fact3]:
            rows = await af_repo.find_where(
                f"entry_id = '{fact.entry_id}' AND owner_id = '{owner_id}'"
            )
            assert len(rows) == 1, f"expected 1 row for {fact.entry_id}"
            assert rows[0].deprecated_by == merged_entry_id, (
                f"{fact.entry_id} should be deprecated by {merged_entry_id}"
            )

        # -- Verify: cluster membership updated.
        members_after = await storage.sqlite.cluster.get_members_with_type(cluster_id)
        assert len(members_after) == 1, (
            f"expected 1 member after merge, got {len(members_after)}"
        )
        new_member_id, new_member_type = members_after[0]
        assert new_member_type == "episode"
        assert new_member_id == merged_entry_id

        # -- Verify: merged episode written to md.
        users_dir = memory_root.users_dir("default", "default")
        episode_files = sorted(
            (users_dir / owner_id / "episodes").rglob("episode-*.md")
        )
        assert len(episode_files) == 1
        md_text = episode_files[0].read_text()
        assert merged_content in md_text
        assert "parent_type" in md_text
        assert "cluster" in md_text

        # -- Verify: the merge returned its EpisodeExtracted event.
        assert len(emitted) == 1
        emitted_event = emitted[0]
        assert emitted_event.episode_entry_id == merged_entry_id
        assert emitted_event.owner_id == owner_id
        assert emitted_event.source == "reflection"

        # -- Verify: report in sqlite matches.
        db_report = await storage.sqlite.reflection_report.get_latest_for_cluster(
            cluster_id
        )
        assert db_report is not None
        assert db_report.merged_entry_id == merged_entry_id
        assert db_report.mode == "init"

    finally:
        conn.close()
        await storage.database.dispose()


@pytest.mark.asyncio
async def test_reflection_update_merges_new_episodes_with_existing_merged(
    tmp_path: Path,
    memory_root: MemoryRoot,
    monkeypatch: pytest.MonkeyPatch,
    storage: Storage,
) -> None:
    """INIT -> add 4th episode -> UPDATE cycle: verify incremental merge.

    Verifies:
        - Second report has mode="update", source_count=2 (old merged + mc_004)
        - Old merged episode (v1) deprecated in LanceDB
        - New merged episode (v2) exists with updated content
        - mc_004's episode deprecated in LanceDB
        - Cluster members: exactly 1, member_type="episode", member_id = v2
        - Old atomic facts from v1's sources remain deprecated
    """
    monkeypatch.setattr(
        MemoryRoot,
        "resolve",
        classmethod(lambda cls: MemoryRoot(root=tmp_path)),
    )
    monkeypatch.setenv("EVEROS_LLM__API_KEY", "fake-key")
    monkeypatch.setenv("EVEROS_LLM__BASE_URL", "https://fake.example.com")

    load_settings.cache_clear()

    await storage.create_schema()

    try:
        # -- LanceDB setup.
        conn = await open_lancedb_connection(memory_root.lancedb_dir, LanceDBSettings())
        ep_table = await conn.create_table("episode", schema=LanceEpisode)
        af_table = await conn.create_table("atomic_fact", schema=AtomicFact)

        ep_repo = _EpisodeRepo(table=ep_table)
        af_repo = _AtomicFactRepo(table=af_table)

        owner_id = "u_test"
        cluster_id = "cl_test_update"

        ts1 = _dt.datetime(2026, 6, 10, 10, 0, 0, tzinfo=_dt.UTC)
        ts2 = _dt.datetime(2026, 6, 10, 11, 0, 0, tzinfo=_dt.UTC)
        ts3 = _dt.datetime(2026, 6, 10, 12, 0, 0, tzinfo=_dt.UTC)

        ep1 = _make_lance_episode(
            entry_id="ep_20260610_0001",
            owner_id=owner_id,
            episode="Andrew has no pets",
            timestamp=ts1,
            parent_id="mc_001",
        )
        ep2 = _make_lance_episode(
            entry_id="ep_20260610_0002",
            owner_id=owner_id,
            episode="Andrew adopted Toby",
            timestamp=ts2,
            parent_id="mc_002",
        )
        ep3 = _make_lance_episode(
            entry_id="ep_20260610_0003",
            owner_id=owner_id,
            episode="Andrew adopted Buddy",
            timestamp=ts3,
            parent_id="mc_003",
        )
        await ep_repo.add([ep1, ep2, ep3])

        # Atomic facts for source episodes.
        fact1 = _make_lance_fact(
            entry_id="af_20260610_0001",
            owner_id=owner_id,
            fact="Andrew has no pets",
            parent_id="ep_20260610_0001",
            timestamp=ts1,
        )
        fact2 = _make_lance_fact(
            entry_id="af_20260610_0002",
            owner_id=owner_id,
            fact="Andrew adopted Toby",
            parent_id="ep_20260610_0002",
            timestamp=ts2,
        )
        fact3 = _make_lance_fact(
            entry_id="af_20260610_0003",
            owner_id=owner_id,
            fact="Andrew adopted Buddy",
            parent_id="ep_20260610_0003",
            timestamp=ts3,
        )
        await af_repo.add([fact1, fact2, fact3])

        # -- SQLite: cluster with 3 memcell members.
        centroid = np.zeros(1024, dtype=np.float32)
        algo_cluster = AlgoCluster(
            id=cluster_id,
            centroid=centroid,
            count=3,
            last_ts=int(ts3.timestamp() * 1000),
            preview=["Andrew has no pets", "Andrew adopted Toby"],
            members=["ep_20260610_0001", "ep_20260610_0002", "ep_20260610_0003"],
        )
        await storage.sqlite.cluster.upsert_with_members(
            algo_cluster,
            owner_id=owner_id,
            owner_type="user",
            kind="user_memory",
            member_type="episode",
        )

        # -- Phase 1: INIT merge.
        init_response = json.dumps(
            {"content": "Andrew adopted Toby and Buddy.", "title": "Andrew pets v1"}
        )
        update_response = json.dumps(
            {
                "content": "Andrew adopted Toby, Buddy, and Scout.",
                "title": "Andrew pets v2",
            }
        )
        fake_llm = FakeLLMClient(responses=[init_response, update_response])
        reflector = EpisodeReflector(llm=fake_llm)

        orchestrator = _reflection(storage, memory_root, ep_repo, af_repo, reflector)

        _, reports_init = await orchestrator(_user_scope(owner_id))

        assert len(reports_init) == 1
        report_init = reports_init[0]
        assert report_init.mode == "init"
        merged_v1_entry_id = report_init.merged_entry_id

        # The extraction the merge triggers is not run here, so the merged
        # episode is not inserted into LanceDB.  Simulate the
        # real pipeline by inserting the merged v1 episode manually.
        merged_v1_ep = _make_lance_episode(
            entry_id=merged_v1_entry_id,
            owner_id=owner_id,
            episode="Andrew adopted Toby and Buddy.",
            timestamp=ts3,
            parent_type="cluster",
            parent_id=cluster_id,
            session_id=None,
        )
        await ep_repo.add([merged_v1_ep])

        # -- Phase 2: add a 4th episode + cluster member, then run UPDATE.
        ts4 = _dt.datetime(2026, 6, 10, 13, 0, 0, tzinfo=_dt.UTC)
        ep4 = _make_lance_episode(
            entry_id="ep_20260610_0004",
            owner_id=owner_id,
            episode="Andrew adopted Scout",
            timestamp=ts4,
            parent_id="mc_004",
        )
        await ep_repo.add([ep4])

        await storage.sqlite.cluster.add_member(
            cluster_id, "ep_20260610_0004", "episode"
        )

        # Fresh orchestrator, same FakeLLM (next pop = update_response).
        orchestrator2 = _reflection(storage, memory_root, ep_repo, af_repo, reflector)
        _, reports_update = await orchestrator2(_user_scope(owner_id))

        # -- Verify: second report is UPDATE with source_count=2.
        assert len(reports_update) == 1
        report_update = reports_update[0]
        assert report_update.mode == "update"
        assert report_update.source_count == 2
        merged_v2_entry_id = report_update.merged_entry_id

        # -- Verify: old merged episode (v1) deprecated.
        v1_rows = await ep_repo.find_where(
            f"entry_id = '{merged_v1_entry_id}' AND owner_id = '{owner_id}'"
        )
        assert len(v1_rows) == 1
        assert v1_rows[0].deprecated_by == merged_v2_entry_id

        # -- Verify: mc_004's episode deprecated.
        mc4_rows = await ep_repo.find_where(
            f"parent_type = 'memcell' AND parent_id = 'mc_004' "
            f"AND owner_id = '{owner_id}'"
        )
        assert len(mc4_rows) == 1
        assert mc4_rows[0].deprecated_by == merged_v2_entry_id

        # -- Verify: new merged episode (v2) written to markdown.
        #    (the extraction the merge triggers is not run here, so v2
        #    is not yet in LanceDB — verify via the md file instead.)
        users_dir = memory_root.users_dir("default", "default")
        episode_files = sorted(
            (users_dir / owner_id / "episodes").rglob("episode-*.md")
        )
        assert len(episode_files) >= 1
        # Both v1 and v2 land in the same daily-log file; check full text.
        all_md = "\n".join(f.read_text() for f in episode_files)
        assert "Andrew adopted Toby, Buddy, and Scout." in all_md
        assert "parent_type" in all_md and "cluster" in all_md

        # -- Verify: cluster members = exactly 1, type=episode, id=v2.
        members_final = await storage.sqlite.cluster.get_members_with_type(cluster_id)
        assert len(members_final) == 1
        final_mid, final_mtype = members_final[0]
        assert final_mtype == "episode"
        assert final_mid == merged_v2_entry_id

        # -- Verify: original atomic facts still deprecated by v1
        #    (they were deprecated in the INIT phase; UPDATE does not touch them).
        for fact in [fact1, fact2, fact3]:
            rows = await af_repo.find_where(
                f"entry_id = '{fact.entry_id}' AND owner_id = '{owner_id}'"
            )
            assert len(rows) == 1
            assert rows[0].deprecated_by == merged_v1_entry_id, (
                f"{fact.entry_id} should still be deprecated by v1"
            )

    finally:
        conn.close()
        await storage.database.dispose()


@pytest.mark.asyncio
async def test_reflected_episodes_visible_in_search_deprecated_excluded(
    tmp_path: Path,
    memory_root: MemoryRoot,
    monkeypatch: pytest.MonkeyPatch,
    storage: Storage,
) -> None:
    """After INIT Reflection, search filters correctly include/exclude episodes.

    Verifies:
        - ``deprecated_by IS NULL`` returns only the merged episode
        - Merged episode has parent_type="cluster" and session_id=None
        - Adding session_id filter excludes the merged episode (session_id IS NULL)
    """
    monkeypatch.setattr(
        MemoryRoot,
        "resolve",
        classmethod(lambda cls: MemoryRoot(root=tmp_path)),
    )
    monkeypatch.setenv("EVEROS_LLM__API_KEY", "fake-key")
    monkeypatch.setenv("EVEROS_LLM__BASE_URL", "https://fake.example.com")

    load_settings.cache_clear()

    await storage.create_schema()

    try:
        # -- LanceDB setup.
        conn = await open_lancedb_connection(memory_root.lancedb_dir, LanceDBSettings())
        ep_table = await conn.create_table("episode", schema=LanceEpisode)
        af_table = await conn.create_table("atomic_fact", schema=AtomicFact)

        ep_repo = _EpisodeRepo(table=ep_table)
        af_repo = _AtomicFactRepo(table=af_table)

        owner_id = "u_test"
        cluster_id = "cl_test_search"

        ts1 = _dt.datetime(2026, 6, 10, 10, 0, 0, tzinfo=_dt.UTC)
        ts2 = _dt.datetime(2026, 6, 10, 11, 0, 0, tzinfo=_dt.UTC)
        ts3 = _dt.datetime(2026, 6, 10, 12, 0, 0, tzinfo=_dt.UTC)

        ep1 = _make_lance_episode(
            entry_id="ep_20260610_0001",
            owner_id=owner_id,
            episode="Andrew has no pets",
            timestamp=ts1,
            parent_id="mc_001",
        )
        ep2 = _make_lance_episode(
            entry_id="ep_20260610_0002",
            owner_id=owner_id,
            episode="Andrew adopted Toby",
            timestamp=ts2,
            parent_id="mc_002",
        )
        ep3 = _make_lance_episode(
            entry_id="ep_20260610_0003",
            owner_id=owner_id,
            episode="Andrew adopted Buddy",
            timestamp=ts3,
            parent_id="mc_003",
        )
        await ep_repo.add([ep1, ep2, ep3])

        # Atomic facts (needed for the orchestrator to complete).
        fact1 = _make_lance_fact(
            entry_id="af_20260610_0001",
            owner_id=owner_id,
            fact="Andrew has no pets",
            parent_id="ep_20260610_0001",
            timestamp=ts1,
        )
        fact2 = _make_lance_fact(
            entry_id="af_20260610_0002",
            owner_id=owner_id,
            fact="Andrew adopted Toby",
            parent_id="ep_20260610_0002",
            timestamp=ts2,
        )
        fact3 = _make_lance_fact(
            entry_id="af_20260610_0003",
            owner_id=owner_id,
            fact="Andrew adopted Buddy",
            parent_id="ep_20260610_0003",
            timestamp=ts3,
        )
        await af_repo.add([fact1, fact2, fact3])

        # -- SQLite: cluster with 3 memcell members.
        centroid = np.zeros(1024, dtype=np.float32)
        algo_cluster = AlgoCluster(
            id=cluster_id,
            centroid=centroid,
            count=3,
            last_ts=int(ts3.timestamp() * 1000),
            preview=["Andrew has no pets", "Andrew adopted Toby"],
            members=["ep_20260610_0001", "ep_20260610_0002", "ep_20260610_0003"],
        )
        await storage.sqlite.cluster.upsert_with_members(
            algo_cluster,
            owner_id=owner_id,
            owner_type="user",
            kind="user_memory",
            member_type="episode",
        )

        # -- Run INIT Reflection.
        merged_content = (
            "Andrew initially had no pets. He later adopted a dog named Toby, "
            "and then adopted another dog named Buddy."
        )
        merged_title = "Andrew's pet adoption journey"
        reflect_response = json.dumps(
            {"content": merged_content, "title": merged_title}
        )
        fake_llm = FakeLLMClient(responses=[reflect_response])
        reflector = EpisodeReflector(llm=fake_llm)

        orchestrator = _reflection(storage, memory_root, ep_repo, af_repo, reflector)

        _, reports = await orchestrator(_user_scope(owner_id))
        assert len(reports) == 1
        merged_entry_id = reports[0].merged_entry_id

        # The extraction the merge triggers is not run here, so the merged
        # episode is not inserted into LanceDB.  Simulate the
        # real pipeline by inserting the merged episode manually.
        merged_ep = _make_lance_episode(
            entry_id=merged_entry_id,
            owner_id=owner_id,
            episode=merged_content,
            timestamp=ts3,
            parent_type="cluster",
            parent_id=cluster_id,
            session_id=None,
        )
        await ep_repo.add([merged_ep])

        # -- Verify: non-deprecated episodes = only the merged one.
        active_rows = await ep_repo.find_where(
            f"owner_id = '{owner_id}' AND deprecated_by IS NULL"
        )
        assert len(active_rows) == 1
        merged_row = active_rows[0]
        assert merged_row.entry_id == merged_entry_id
        assert merged_row.parent_type == "cluster"
        assert merged_row.session_id is None

        # -- Verify: session_id filter excludes the merged episode.
        session_rows = await ep_repo.find_where(
            f"owner_id = '{owner_id}' AND deprecated_by IS NULL "
            f"AND session_id = 's_test'"
        )
        assert len(session_rows) == 0

    finally:
        conn.close()
        await storage.database.dispose()
