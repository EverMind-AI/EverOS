"""Tests for :class:`DecisionReflectionOrchestrator`.

Constructor dependencies are mocked. Tests verify:
- candidate selection filtering (INIT vs UPDATE) with ``kind=decision``
- full INIT-mode flow with merge + deprecate (no atomic facts)
- UPDATE-mode ``old_decision`` passthrough
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
from everos.memory.events import DecisionExtracted, EpisodeExtracted
from everos.memory.reflection.decision_orchestrator import (
    _MAX_CLUSTERS_PER_RUN,
    DecisionReflectionOrchestrator,
    _merged_decision_to_entry_body,
    _ts_to_ms,
)


@pytest.fixture(autouse=True)
def _isolate_locks() -> None:
    _reset_for_tests()


@dataclass
class _FakeAlgoResult:
    """Minimal stand-in for ``everalgo.types.Decision``."""

    owner_id: str | None
    title: str
    decision: str
    reason: str
    timestamp: int
    impact: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class _FakeDecisionRow:
    """Minimal stand-in for a LanceDB Decision row."""

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
    title: str = "test title"
    decision: str = "test decision text"
    reason: str = "test reason"
    impact: str | None = None
    tags: list[str] = field(default_factory=lambda: ["runtime"])
    md_path: str = "users/u_alice/decisions/decision-2026-06-01.md"
    content_sha256: str = "abc123"
    deprecated_by: str | None = None
    vector: list[float] | None = None


def _make_decision_row(
    entry_id: str = "dc_20260601_0001",
    parent_id: str = "mc_aaa",
    parent_type: str = "memcell",
    owner_id: str = "u_alice",
    **kwargs: object,
) -> _FakeDecisionRow:
    return _FakeDecisionRow(
        id=f"{owner_id}_{entry_id}",
        entry_id=entry_id,
        owner_id=owner_id,
        parent_id=parent_id,
        parent_type=parent_type,
        **kwargs,  # type: ignore[arg-type]
    )


def _make_entry_id(formatted: str = "dc_20260614_0001") -> MagicMock:
    eid = MagicMock()
    eid.format.return_value = formatted
    eid.date = _dt.date(2026, 6, 14)
    return eid


def _build_orchestrator(
    *,
    cluster_repo: MagicMock | None = None,
    decision_store: MagicMock | None = None,
    decision_writer: MagicMock | None = None,
    report_repo: MagicMock | None = None,
    reflector: MagicMock | None = None,
    embedder: MagicMock | None = None,
) -> DecisionReflectionOrchestrator:
    return DecisionReflectionOrchestrator(
        cluster_repo=cluster_repo or MagicMock(),
        decision_store=decision_store or MagicMock(),
        decision_writer=decision_writer or MagicMock(),
        report_repo=report_repo or MagicMock(),
        reflector=reflector or MagicMock(),
        embedder=embedder or MagicMock(),
    )


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
            ("cl_new_1", 1),
            ("cl_reflected", 2),
            ("cl_reflected_1", 1),
        ]
    )

    orch = _build_orchestrator(cluster_repo=cluster_repo, report_repo=report_repo)
    result = await orch._select_candidates(
        owner_id="u_alice",
        kind="decision",
        app_id="default",
        project_id="default",
    )

    assert result == ["cl_new_3", "cl_reflected"]
    cluster_repo.list_ids_and_member_counts.assert_awaited_once_with(
        "u_alice", "decision", app_id="default", project_id="default"
    )


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
        kind="decision",
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
    cluster_repo.list_ids_and_member_counts.assert_awaited_once_with(
        "u_alice", "decision", app_id="default", project_id="default"
    )


async def test_run_init_mode_merges_and_deprecates() -> None:
    """Full INIT flow: 2 decision members -> merge -> write -> deprecate."""
    cluster_repo = MagicMock()
    decision_store = MagicMock()
    decision_writer = MagicMock()
    report_repo = MagicMock()
    reflector = MagicMock()
    embedder = MagicMock()

    report_repo.list_reflected_cluster_ids = AsyncMock(return_value=set())
    cluster_repo.list_ids_and_member_counts = AsyncMock(return_value=[("cl_abc", 2)])
    decision_store.find_where = AsyncMock(return_value=[])
    cluster_repo.get_members_with_type = AsyncMock(
        return_value=[
            ("dc_20260601_0001", "decision"),
            ("dc_20260601_0002", "decision"),
        ]
    )

    dc1 = _make_decision_row(
        entry_id="dc_20260601_0001", parent_id="mc_001", owner_id="u_alice"
    )
    dc2 = _make_decision_row(
        entry_id="dc_20260601_0002", parent_id="mc_002", owner_id="u_alice"
    )
    decision_store.find_by_owner_entries = AsyncMock(return_value=[dc1, dc2])

    algo_result = _FakeAlgoResult(
        owner_id=None,
        title="merged title",
        decision="merged decision text",
        reason="merged reason",
        impact="merged impact",
        tags=["runtime"],
        timestamp=1717200000000,
    )
    reflector.areflect = AsyncMock(return_value=algo_result)

    entry_id_mock = _make_entry_id("dc_20260614_0001")
    decision_writer.append_entries = AsyncMock(return_value=[entry_id_mock])
    decision_writer.patch_frontmatter = AsyncMock()

    ctx = FakeStrategyContext()
    ctx.wait_for_event = AsyncMock()  # type: ignore[method-assign]

    cluster_repo.remove_members = AsyncMock()
    cluster_repo.add_member = AsyncMock()
    cluster_repo.update_metadata = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1024)
    decision_store.update = AsyncMock()
    report_repo.create = AsyncMock()

    orch = _build_orchestrator(
        cluster_repo=cluster_repo,
        decision_store=decision_store,
        decision_writer=decision_writer,
        report_repo=report_repo,
        reflector=reflector,
        embedder=embedder,
    )

    reports = await orch.run(ctx=ctx, owner_id="u_alice")

    reflector.areflect.assert_awaited_once()
    call_kwargs = reflector.areflect.call_args
    assert "old_decision" not in (call_kwargs.kwargs or {})

    decision_writer.append_entries.assert_awaited_once()
    ctx.wait_for_event.assert_not_awaited()

    assert len(ctx.emitted) == 1
    event = ctx.emitted[0]
    assert isinstance(event, DecisionExtracted)
    assert not isinstance(event, EpisodeExtracted)
    assert event.source == "reflection"
    assert event.session_id is None
    assert event.decision_entry_id == "dc_20260614_0001"
    assert event.decision_text == "merged decision text"

    cluster_repo.remove_members.assert_awaited_once()
    cluster_repo.add_member.assert_awaited_once_with(
        "cl_abc", "dc_20260614_0001", "decision"
    )
    embedder.embed.assert_awaited_once_with("merged decision text")
    decision_store.update.assert_awaited()

    report_repo.create.assert_awaited_once()
    created = report_repo.create.await_args.args[0]
    assert created.deprecated_fact_count == 0
    assert len(reports) == 1


async def test_run_update_mode_uses_old_decision() -> None:
    """UPDATE flow: cluster has 1 merged decision + 1 original decision."""
    cluster_repo = MagicMock()
    decision_store = MagicMock()
    decision_writer = MagicMock()
    report_repo = MagicMock()
    reflector = MagicMock()
    embedder = MagicMock()

    report_repo.list_reflected_cluster_ids = AsyncMock(return_value={"cl_update"})
    cluster_repo.list_ids_and_member_counts = AsyncMock(return_value=[("cl_update", 2)])
    decision_store.find_where = AsyncMock(return_value=[])
    cluster_repo.get_members_with_type = AsyncMock(
        return_value=[
            ("dc_20260612_0001", "decision"),
            ("dc_20260613_0001", "decision"),
        ]
    )

    old_merged = _make_decision_row(
        entry_id="dc_20260612_0001",
        parent_id="cl_update",
        parent_type="cluster",
        owner_id="u_alice",
        decision="old merged text",
    )
    new_dc = _make_decision_row(
        entry_id="dc_20260613_0001",
        parent_id="mc_004",
        owner_id="u_alice",
        decision="new decision text",
    )
    decision_store.find_by_owner_entries = AsyncMock(return_value=[new_dc, old_merged])

    algo_result = _FakeAlgoResult(
        owner_id=None,
        title="updated title",
        decision="updated merged text",
        reason="updated reason",
        timestamp=1717200000000,
    )
    reflector.areflect = AsyncMock(return_value=algo_result)

    entry_id_mock = _make_entry_id("dc_20260614_0002")
    decision_writer.append_entries = AsyncMock(return_value=[entry_id_mock])
    decision_writer.patch_frontmatter = AsyncMock()

    cluster_repo.remove_members = AsyncMock()
    cluster_repo.add_member = AsyncMock()
    cluster_repo.update_metadata = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1024)
    decision_store.update = AsyncMock()
    report_repo.create = AsyncMock()

    ctx = FakeStrategyContext()
    orch = _build_orchestrator(
        cluster_repo=cluster_repo,
        decision_store=decision_store,
        decision_writer=decision_writer,
        report_repo=report_repo,
        reflector=reflector,
        embedder=embedder,
    )

    reports = await orch.run(ctx=ctx, owner_id="u_alice")

    reflector.areflect.assert_awaited_once()
    _, kwargs = reflector.areflect.call_args
    assert "old_decision" in kwargs
    assert len(reports) == 1


async def test_llm_failure_skips_cluster() -> None:
    """Reflector raising an exception skips the cluster, continues."""
    cluster_repo = MagicMock()
    decision_store = MagicMock()
    report_repo = MagicMock()
    reflector = MagicMock()

    report_repo.list_reflected_cluster_ids = AsyncMock(return_value=set())
    cluster_repo.list_ids_and_member_counts = AsyncMock(return_value=[("cl_fail", 2)])
    decision_store.find_where = AsyncMock(return_value=[])
    cluster_repo.get_members_with_type = AsyncMock(
        return_value=[("dc_001", "decision"), ("dc_002", "decision")]
    )

    dc1 = _make_decision_row(entry_id="dc_001", parent_id="mc_a", owner_id="u_alice")
    dc2 = _make_decision_row(entry_id="dc_002", parent_id="mc_b", owner_id="u_alice")
    decision_store.find_by_owner_entries = AsyncMock(return_value=[dc1, dc2])
    reflector.areflect = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    ctx = FakeStrategyContext()
    orch = _build_orchestrator(
        cluster_repo=cluster_repo,
        decision_store=decision_store,
        report_repo=report_repo,
        reflector=reflector,
    )

    reports = await orch.run(ctx=ctx, owner_id="u_alice")
    assert reports == []
    assert len(ctx.emitted) == 0


def test_merged_decision_to_entry_body_shape() -> None:
    """Verify the inline/sections shape for a merged decision."""
    result = _FakeAlgoResult(
        owner_id=None,
        title="merged title",
        decision="merged text",
        reason="merged reason",
        impact="merged impact",
        tags=["runtime"],
        timestamp=1717200000000,
    )
    inline, sections = _merged_decision_to_entry_body(
        result, "cl_abc", "u_alice", "2026-06-01T00:00:00+00:00"
    )
    assert inline["parent_type"] == "cluster"
    assert inline["parent_id"] == "cl_abc"
    assert inline["owner_id"] == "u_alice"
    assert inline["tags"] == ["runtime"]
    assert "session_id" not in inline
    assert sections["Title"] == "merged title"
    assert sections["Decision"] == "merged text"
    assert sections["Reason"] == "merged reason"
    assert sections["Impact"] == "merged impact"


def test_merged_decision_omits_empty_impact() -> None:
    result = _FakeAlgoResult(
        owner_id=None,
        title="t",
        decision="d",
        reason="r",
        timestamp=1717200000000,
    )
    _, sections = _merged_decision_to_entry_body(
        result, "cl_abc", "u_alice", "2026-06-01T00:00:00+00:00"
    )
    assert "Impact" not in sections


def test_ts_to_ms_datetime() -> None:
    dt = _dt.datetime(2026, 6, 1, tzinfo=_dt.UTC)
    ms = _ts_to_ms(dt)
    assert isinstance(ms, int)
    assert ms > 0


def test_ts_to_ms_int_passthrough() -> None:
    assert _ts_to_ms(1717200000000) == 1717200000000
