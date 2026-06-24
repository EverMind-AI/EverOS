"""Tests for :class:`ReflectionOrchestrator`.

All seven constructor dependencies are mocked. Tests verify:
- candidate selection filtering logic (INIT vs UPDATE)
- full INIT-mode flow with merge + deprecate
- UPDATE-mode old_episode passthrough
- LLM failure skips cluster gracefully
- empty candidates return empty list
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from everos.infra.ome.testing import FakeStrategyContext
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.reflection.orchestrator import (
    _MAX_CLUSTERS_PER_RUN,
    ReflectionOrchestrator,
    _merged_episode_to_entry_body,
    _ts_to_ms,
)


@pytest.fixture(autouse=True)
def _isolate_locks() -> None:
    _reset_for_tests()


# ── Helpers ───────────────────────────────────────────────────────────────


@dataclass
class _FakeAlgoResult:
    """Minimal stand-in for ``everalgo.types.Episode``."""

    owner_id: str | None
    episode: str
    subject: str
    timestamp: int


@dataclass
class _FakeEpisodeRow:
    """Minimal stand-in for a LanceDB Episode row."""

    id: str
    entry_id: str
    owner_id: str
    owner_type: str = "user"
    app_id: str = "default"
    project_id: str = "default"
    session_id: str | None = "s_test"
    timestamp: _dt.datetime = _dt.datetime(2026, 6, 1, tzinfo=_dt.UTC)
    parent_type: str = "memcell"
    parent_id: str = "mc_aaa"
    sender_ids: list[str] = field(default_factory=list)
    subject: str | None = "test subject"
    summary: str | None = None
    episode: str = "test episode text"
    episode_tokens: str = "test episode text"
    md_path: str = "/tmp/test.md"
    content_sha256: str = "abc123"
    deprecated_by: str | None = None
    vector: list[float] | None = None


def _make_episode_row(
    entry_id: str = "ep_20260601_0001",
    parent_id: str = "mc_aaa",
    parent_type: str = "memcell",
    owner_id: str = "u_alice",
    **kwargs: object,
) -> _FakeEpisodeRow:
    return _FakeEpisodeRow(
        id=f"{owner_id}_{entry_id}",
        entry_id=entry_id,
        owner_id=owner_id,
        parent_id=parent_id,
        parent_type=parent_type,
        **kwargs,
    )


def _make_entry_id(formatted: str = "ep_20260614_0001") -> MagicMock:
    eid = MagicMock()
    eid.format.return_value = formatted
    eid.date = _dt.date(2026, 6, 14)
    return eid


def _build_orchestrator(
    *,
    cluster_repo: MagicMock | None = None,
    episode_store: MagicMock | None = None,
    atomic_fact_store: MagicMock | None = None,
    episode_writer: MagicMock | None = None,
    report_repo: MagicMock | None = None,
    reflector: MagicMock | None = None,
    embedder: MagicMock | None = None,
) -> ReflectionOrchestrator:
    return ReflectionOrchestrator(
        cluster_repo=cluster_repo or MagicMock(),
        episode_store=episode_store or MagicMock(),
        atomic_fact_store=atomic_fact_store or MagicMock(),
        episode_writer=episode_writer or MagicMock(),
        report_repo=report_repo or MagicMock(),
        reflector=reflector or MagicMock(),
        embedder=embedder or MagicMock(),
    )


# ── Tests ─────────────────────────────────────────────────────────────────


async def test_select_candidates_init_and_update() -> None:
    """Unreflected clusters with >=2 members are INIT candidates.
    Reflected clusters with >1 member are UPDATE candidates.
    """
    cluster_repo = MagicMock()
    report_repo = MagicMock()

    report_repo.list_reflected_cluster_ids = AsyncMock(return_value={"cl_reflected"})
    cluster_repo.list_ids_and_member_counts = AsyncMock(
        return_value=[
            ("cl_new_3", 3),
            ("cl_new_1", 1),  # only 1 member -> skip
            ("cl_reflected", 2),  # reflected + 2 members -> UPDATE
            ("cl_reflected_1", 1),  # reflected + 1 member -> skip
        ]
    )

    orch = _build_orchestrator(cluster_repo=cluster_repo, report_repo=report_repo)
    result = await orch._select_candidates(
        owner_id="u_alice",
        kind="user_memory",
        app_id="default",
        project_id="default",
    )

    assert result == ["cl_new_3", "cl_reflected"]


async def test_select_candidates_respects_max_limit() -> None:
    """More than ``_MAX_CLUSTERS_PER_RUN`` candidates are truncated."""
    cluster_repo = MagicMock()
    report_repo = MagicMock()
    report_repo.list_reflected_cluster_ids = AsyncMock(return_value=set())
    cluster_repo.list_ids_and_member_counts = AsyncMock(
        return_value=[(f"cl_{i:03d}", i + 2) for i in range(_MAX_CLUSTERS_PER_RUN + 5)]
    )

    orch = _build_orchestrator(cluster_repo=cluster_repo, report_repo=report_repo)
    result = await orch._select_candidates(
        owner_id="u_alice",
        kind="user_memory",
        app_id="default",
        project_id="default",
    )
    assert len(result) == _MAX_CLUSTERS_PER_RUN


async def test_empty_candidates_returns_empty() -> None:
    """No qualifying clusters -> run() returns empty list immediately."""
    cluster_repo = MagicMock()
    report_repo = MagicMock()
    report_repo.list_reflected_cluster_ids = AsyncMock(return_value=set())
    cluster_repo.list_ids_and_member_counts = AsyncMock(
        return_value=[("cl_only_one", 1)]
    )

    orch = _build_orchestrator(cluster_repo=cluster_repo, report_repo=report_repo)
    ctx = FakeStrategyContext()
    reports = await orch.run(ctx=ctx, owner_id="u_alice")
    assert reports == []


async def test_run_init_mode_merges_and_deprecates() -> None:
    """Full INIT flow: 2 episode members -> merge -> write -> deprecate."""
    cluster_repo = MagicMock()
    episode_store = MagicMock()
    atomic_fact_store = MagicMock()
    episode_writer = MagicMock()
    report_repo = MagicMock()
    reflector = MagicMock()
    embedder = MagicMock()

    # SELECT: 1 candidate cluster.
    report_repo.list_reflected_cluster_ids = AsyncMock(return_value=set())
    cluster_repo.list_ids_and_member_counts = AsyncMock(return_value=[("cl_abc", 2)])

    # Step 0: orphan detection returns empty.
    episode_store.find_where = AsyncMock(return_value=[])

    # Step 1: cluster members (episode type).
    cluster_repo.get_members_with_type = AsyncMock(
        return_value=[("ep_20260601_0001", "episode"), ("ep_20260601_0002", "episode")]
    )

    # Step 2: fetch episodes by entry_id.
    ep1 = _make_episode_row(
        entry_id="ep_20260601_0001", parent_id="mc_001", owner_id="u_alice"
    )
    ep2 = _make_episode_row(
        entry_id="ep_20260601_0002", parent_id="mc_002", owner_id="u_alice"
    )
    episode_store.find_by_owner_entries = AsyncMock(return_value=[ep1, ep2])

    # Step 4: algo reflector returns merged episode.
    algo_result = _FakeAlgoResult(
        owner_id=None,
        episode="merged episode text",
        subject="merged subject",
        timestamp=1717200000000,
    )
    reflector.areflect = AsyncMock(return_value=algo_result)

    # Step 5: episode writer returns entry id.
    entry_id_mock = _make_entry_id("ep_20260614_0001")
    episode_writer.append_entries = AsyncMock(return_value=[entry_id_mock])
    episode_writer.patch_frontmatter = AsyncMock()

    # Step 6: wait_for_event succeeds.
    ctx = FakeStrategyContext()

    # Step 7: deprecate -> cluster operations.
    cluster_repo.remove_members = AsyncMock()
    cluster_repo.add_member = AsyncMock()
    cluster_repo.update_metadata = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1024)

    # LanceDB episode store update for deprecation.
    episode_store.update = AsyncMock()

    # Atomic fact store.
    atomic_fact_store.update = AsyncMock()

    # Report repo.
    report_repo.create = AsyncMock()

    orch = _build_orchestrator(
        cluster_repo=cluster_repo,
        episode_store=episode_store,
        atomic_fact_store=atomic_fact_store,
        episode_writer=episode_writer,
        report_repo=report_repo,
        reflector=reflector,
        embedder=embedder,
    )

    reports = await orch.run(ctx=ctx, owner_id="u_alice")

    # Reflector was called in INIT mode (no old_episode kwarg).
    reflector.areflect.assert_awaited_once()
    call_kwargs = reflector.areflect.call_args
    assert "old_episode" not in (call_kwargs.kwargs or {})

    # Episode was written.
    episode_writer.append_entries.assert_awaited_once()

    # EpisodeExtracted was emitted.
    assert len(ctx.emitted) == 1
    event = ctx.emitted[0]
    assert event.source == "reflection"
    assert event.session_id is None

    # Cluster updated.
    cluster_repo.remove_members.assert_awaited_once()
    cluster_repo.add_member.assert_awaited_once_with(
        "cl_abc", "ep_20260614_0001", "episode"
    )

    # Report created.
    report_repo.create.assert_awaited_once()
    assert len(reports) == 1


async def test_run_update_mode_uses_old_episode() -> None:
    """UPDATE flow: cluster has 1 merged episode + 1 original episode."""
    cluster_repo = MagicMock()
    episode_store = MagicMock()
    atomic_fact_store = MagicMock()
    episode_writer = MagicMock()
    report_repo = MagicMock()
    reflector = MagicMock()
    embedder = MagicMock()

    # SELECT.
    report_repo.list_reflected_cluster_ids = AsyncMock(return_value={"cl_update"})
    cluster_repo.list_ids_and_member_counts = AsyncMock(return_value=[("cl_update", 2)])

    # Orphan detection.
    episode_store.find_where = AsyncMock(return_value=[])

    # Members: both episode type (old merged + new original).
    cluster_repo.get_members_with_type = AsyncMock(
        return_value=[("ep_20260612_0001", "episode"), ("ep_20260613_0001", "episode")]
    )

    # Episodes.
    old_merged = _make_episode_row(
        entry_id="ep_20260612_0001",
        parent_id="cl_update",
        parent_type="cluster",
        owner_id="u_alice",
        episode="old merged text",
    )
    new_ep = _make_episode_row(
        entry_id="ep_20260613_0001",
        parent_id="mc_004",
        owner_id="u_alice",
        episode="new episode text",
    )
    episode_store.find_by_owner_entries = AsyncMock(return_value=[new_ep, old_merged])

    # Reflector.
    algo_result = _FakeAlgoResult(
        owner_id=None,
        episode="updated merged text",
        subject="updated subject",
        timestamp=1717200000000,
    )
    reflector.areflect = AsyncMock(return_value=algo_result)

    # Writer.
    entry_id_mock = _make_entry_id("ep_20260614_0002")
    episode_writer.append_entries = AsyncMock(return_value=[entry_id_mock])
    episode_writer.patch_frontmatter = AsyncMock()

    # Deprecate deps.
    cluster_repo.remove_members = AsyncMock()
    cluster_repo.add_member = AsyncMock()
    cluster_repo.update_metadata = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1024)
    atomic_fact_store.update = AsyncMock()
    episode_store.update = AsyncMock()
    report_repo.create = AsyncMock()

    ctx = FakeStrategyContext()
    orch = _build_orchestrator(
        cluster_repo=cluster_repo,
        episode_store=episode_store,
        atomic_fact_store=atomic_fact_store,
        episode_writer=episode_writer,
        report_repo=report_repo,
        reflector=reflector,
        embedder=embedder,
    )

    reports = await orch.run(ctx=ctx, owner_id="u_alice")

    # Reflector called with old_episode kwarg (UPDATE mode).
    reflector.areflect.assert_awaited_once()
    _, kwargs = reflector.areflect.call_args
    assert "old_episode" in kwargs

    assert len(reports) == 1


async def test_llm_failure_skips_cluster() -> None:
    """Reflector raising an exception skips the cluster, continues."""
    cluster_repo = MagicMock()
    episode_store = MagicMock()
    report_repo = MagicMock()
    reflector = MagicMock()

    # SELECT: 1 candidate.
    report_repo.list_reflected_cluster_ids = AsyncMock(return_value=set())
    cluster_repo.list_ids_and_member_counts = AsyncMock(return_value=[("cl_fail", 2)])

    # Orphan detection.
    episode_store.find_where = AsyncMock(return_value=[])

    # Members.
    cluster_repo.get_members_with_type = AsyncMock(
        return_value=[("ep_001", "episode"), ("ep_002", "episode")]
    )

    # Episodes.
    ep1 = _make_episode_row(entry_id="ep_001", parent_id="mc_a", owner_id="u_alice")
    ep2 = _make_episode_row(entry_id="ep_002", parent_id="mc_b", owner_id="u_alice")
    episode_store.find_by_owner_entries = AsyncMock(return_value=[ep1, ep2])

    # Reflector fails.
    reflector.areflect = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    ctx = FakeStrategyContext()
    orch = _build_orchestrator(
        cluster_repo=cluster_repo,
        episode_store=episode_store,
        report_repo=report_repo,
        reflector=reflector,
    )

    reports = await orch.run(ctx=ctx, owner_id="u_alice")
    assert reports == []
    assert len(ctx.emitted) == 0


# ── Unit helpers ──────────────────────────────────────────────────────────


def test_merged_episode_to_entry_body_shape() -> None:
    """Verify the inline/sections shape for a merged episode."""
    result = _FakeAlgoResult(
        owner_id=None,
        episode="merged text",
        subject="merged subject",
        timestamp=1717200000000,
    )
    inline, sections = _merged_episode_to_entry_body(
        result, "cl_abc", "u_alice", "2026-06-01T00:00:00+00:00"
    )
    assert inline["parent_type"] == "cluster"
    assert inline["parent_id"] == "cl_abc"
    assert inline["owner_id"] == "u_alice"
    assert "session_id" not in inline
    assert sections["Subject"] == "merged subject"
    assert sections["Content"] == "merged text"


def test_ts_to_ms_datetime() -> None:
    """datetime -> milliseconds conversion."""
    dt = _dt.datetime(2026, 6, 1, tzinfo=_dt.UTC)
    ms = _ts_to_ms(dt)
    assert isinstance(ms, int)
    assert ms > 0


def test_ts_to_ms_int_passthrough() -> None:
    """int -> int passthrough."""
    assert _ts_to_ms(1717200000000) == 1717200000000
