"""Tests for :class:`ReflectionOrchestrator` and :class:`ReflectionFinisher`.

Every storage dependency is a mock of its port. Tests verify:
- candidate selection filtering logic (INIT vs UPDATE)
- full INIT-mode merge, returning the merged episode with its context
- UPDATE-mode old_episode passthrough
- the finisher deprecates in markdown only (episodes and their facts),
  never in the index — the cascade carries it there
- LLM failure skips cluster gracefully
- empty candidates return empty list
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from everos.infra.persistence.lancedb import render_predicate
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.events import EpisodeExtracted, ReflectedEpisode, ReflectionContext
from everos.memory.models import AtomicFact, Episode
from everos.memory.ports import MemoryRef, Scope, SearchHit, Stored
from everos.memory.reflection import ReflectionFinisher, ReflectionOrchestrator
from everos.memory.reflection.orchestrator import _MAX_CLUSTERS_PER_RUN

_SCOPE = Scope(
    app_id="default", project_id="default", owner_id="u_alice", owner_type="user"
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
    summary: str = "merged summary"


def _fact_hit(entry_id: str, md_path: str) -> SearchHit[AtomicFact]:
    """A filtered-scan match of one live atomic fact."""
    return SearchHit(
        stored=Stored(
            ref=MemoryRef(id=f"u_alice_{entry_id}", entry_id=entry_id, md_path=md_path),
            scope=_SCOPE,
            item=AtomicFact(
                owner_id="u_alice",
                content="a fact",
                timestamp=0,
                parent_id="ep_20260601_0001",
            ),
        ),
        score=None,
    )


def _stored_episode(
    entry_id: str = "ep_20260601_0001",
    parent_id: str = "mc_aaa",
    parent_type: str = "memcell",
    episode: str = "test episode text",
    md_path: str = "default_app/default_project/users/u_alice/episodes/e.md",
) -> Stored[Episode]:
    return Stored(
        ref=MemoryRef(id=f"u_alice_{entry_id}", entry_id=entry_id, md_path=md_path),
        scope=_SCOPE,
        item=Episode(
            owner_id="u_alice",
            episode=episode,
            summary="",
            timestamp=1_780_272_000_000,
            subject="test subject",
            session_id="s_test",
            parent_id=parent_id,
            parent_type=parent_type,  # type: ignore[arg-type]
        ),
    )


def _merged_ref(entry_id: str) -> MemoryRef:
    return MemoryRef(
        id=f"u_alice_{entry_id}",
        entry_id=entry_id,
        md_path="default_app/default_project/users/u_alice/episodes/m.md",
    )


@dataclass
class _Ports:
    cluster_store: MagicMock
    episode_store: MagicMock
    episode_index: MagicMock
    atomic_fact_store: MagicMock
    atomic_fact_index: MagicMock
    report_store: MagicMock
    reflector: MagicMock
    embedder: MagicMock

    def orchestrator(self) -> ReflectionOrchestrator:
        return ReflectionOrchestrator(
            cluster_store=self.cluster_store,
            episode_store=self.episode_store,
            episode_index=self.episode_index,
            report_store=self.report_store,
            reflector=self.reflector,
        )

    def finisher(self) -> ReflectionFinisher:
        return ReflectionFinisher(
            cluster_store=self.cluster_store,
            episode_store=self.episode_store,
            atomic_fact_store=self.atomic_fact_store,
            atomic_fact_index=self.atomic_fact_index,
            report_store=self.report_store,
            embedder=self.embedder,
        )


@pytest.fixture
def ports() -> _Ports:
    return _Ports(*(MagicMock() for _ in range(8)))


def _arm_cluster_flow(
    ports: _Ports,
    *,
    cluster_id: str,
    members: list[tuple[str, str]],
    episodes: list[Stored[Episode]],
    merged_entry_id: str,
    reflected: set[str] | None = None,
) -> None:
    """Wire every port call one successful cluster merge makes."""
    ports.report_store.reflected_cluster_ids = AsyncMock(
        return_value=reflected or set()
    )
    ports.cluster_store.list_ids_and_member_counts = AsyncMock(
        return_value=[(cluster_id, len(members))]
    )
    ports.episode_index.orphans = AsyncMock(return_value=[])
    ports.cluster_store.members = AsyncMock(return_value=members)
    ports.episode_index.find_by_entry_ids = AsyncMock(return_value=episodes)
    ports.reflector.areflect = AsyncMock(
        return_value=_FakeAlgoResult(
            owner_id=None,
            episode="merged episode text",
            subject="merged subject",
            timestamp=1717200000000,
        )
    )
    ports.episode_store.append_many = AsyncMock(
        return_value=[_merged_ref(merged_entry_id)]
    )
    ports.episode_store.deprecate = AsyncMock()
    ports.atomic_fact_index.find = AsyncMock(return_value=[])
    ports.atomic_fact_store.deprecate = AsyncMock()
    ports.cluster_store.remove_members = AsyncMock()
    ports.cluster_store.add_member = AsyncMock()
    ports.cluster_store.update_metadata = AsyncMock()
    ports.embedder.embed = AsyncMock(return_value=[0.1] * 1024)
    ports.report_store.create = AsyncMock()


# ── Tests ─────────────────────────────────────────────────────────────────


async def test_select_candidates_init_and_update(ports: _Ports) -> None:
    """Unreflected clusters with >=2 members are INIT candidates.
    Reflected clusters with >1 member are UPDATE candidates.
    """
    ports.report_store.reflected_cluster_ids = AsyncMock(return_value={"cl_reflected"})
    ports.cluster_store.list_ids_and_member_counts = AsyncMock(
        return_value=[
            ("cl_new_3", 3),
            ("cl_new_1", 1),  # only 1 member -> skip
            ("cl_reflected", 2),  # reflected + 2 members -> UPDATE
            ("cl_reflected_1", 1),  # reflected + 1 member -> skip
        ]
    )

    result = await ports.orchestrator()._select_candidates(
        scope=_SCOPE, kind="user_memory"
    )

    assert result == ["cl_new_3", "cl_reflected"]
    ports.cluster_store.list_ids_and_member_counts.assert_awaited_once_with(
        _SCOPE, "user_memory"
    )


async def test_select_candidates_respects_max_limit(ports: _Ports) -> None:
    """More than ``_MAX_CLUSTERS_PER_RUN`` candidates are truncated."""
    ports.report_store.reflected_cluster_ids = AsyncMock(return_value=set())
    ports.cluster_store.list_ids_and_member_counts = AsyncMock(
        return_value=[(f"cl_{i:03d}", i + 2) for i in range(_MAX_CLUSTERS_PER_RUN + 5)]
    )

    result = await ports.orchestrator()._select_candidates(
        scope=_SCOPE, kind="user_memory"
    )

    assert len(result) == _MAX_CLUSTERS_PER_RUN


async def test_empty_candidates_returns_empty(ports: _Ports) -> None:
    """No qualifying clusters -> run() returns empty list immediately."""
    ports.report_store.reflected_cluster_ids = AsyncMock(return_value=set())
    ports.cluster_store.list_ids_and_member_counts = AsyncMock(
        return_value=[("cl_only_one", 1)]
    )

    merged = await ports.orchestrator().run(scope=_SCOPE)

    assert merged == []


async def test_deprecate_facts_patches_fact_files_scoped_to_the_project(
    ports: _Ports,
) -> None:
    """``parent_id`` restarts per (app, project, day), so the fact lookup
    carries the scope; the marks land in each fact's own md file."""
    scope = Scope(
        app_id="claude-code", project_id="alpha", owner_id="u_alice", owner_type="user"
    )
    ports.atomic_fact_index.find = AsyncMock(
        return_value=[
            _fact_hit(eid, md_path)
            for eid, md_path in (
                ("af_1", "facts/a.md"),
                ("af_2", "facts/a.md"),
                ("af_3", "facts/b.md"),
            )
        ]
    )
    ports.atomic_fact_store.deprecate = AsyncMock()

    marked = await ports.finisher()._deprecate_facts(
        scope=scope,
        parent_ids={"ep_20260601_0001"},
        merged_entry_id="ep_20260614_0001",
    )

    assert marked == 3
    rendered = render_predicate(ports.atomic_fact_index.find.await_args.args[0])
    assert "(app_id = 'claude-code')" in rendered
    assert "(project_id = 'alpha')" in rendered
    assert "'ep_20260601_0001'" in rendered
    assert "deprecated_by IS NULL" in rendered
    calls = {
        c.kwargs["md_path"]: c.kwargs["replaced"]
        for c in ports.atomic_fact_store.deprecate.await_args_list
    }
    assert calls == {
        "facts/a.md": {"af_1": "ep_20260614_0001", "af_2": "ep_20260614_0001"},
        "facts/b.md": {"af_3": "ep_20260614_0001"},
    }


async def test_deprecate_facts_without_parents_reads_nothing(ports: _Ports) -> None:
    ports.atomic_fact_index.find = AsyncMock()

    marked = await ports.finisher()._deprecate_facts(
        scope=_SCOPE, parent_ids=set(), merged_entry_id="ep_m"
    )

    assert marked == 0
    ports.atomic_fact_index.find.assert_not_awaited()


async def test_merged_episode_is_written_with_a_cluster_parent(ports: _Ports) -> None:
    ports.episode_store.append_many = AsyncMock(return_value=[_merged_ref("ep_m")])

    entry_id = await ports.orchestrator()._write_merged_episode(
        cluster_id="cl_abc",
        scope=_SCOPE,
        algo_result=_FakeAlgoResult(  # type: ignore[arg-type]
            owner_id=None, episode="merged text", subject="topic", timestamp=0
        ),
        last_ts=1_780_272_000_000,
    )

    assert entry_id == "ep_m"
    scope, (merged,) = ports.episode_store.append_many.await_args.args
    assert scope == _SCOPE
    assert merged == Episode(
        owner_id="u_alice",
        episode="merged text",
        summary="merged summary",
        timestamp=1_780_272_000_000,
        subject="topic",
        parent_id="cl_abc",
        parent_type="cluster",
    )


async def test_run_init_mode_merges_and_deprecates(ports: _Ports) -> None:
    """Full INIT flow: 2 episode members -> merge -> write -> (facts) ->
    finish: deprecate both originals and move the cluster onto the merge."""
    ep1 = _stored_episode(entry_id="ep_20260601_0001", parent_id="mc_001")
    ep2 = _stored_episode(entry_id="ep_20260601_0002", parent_id="mc_002")
    _arm_cluster_flow(
        ports,
        cluster_id="cl_abc",
        members=[("ep_20260601_0001", "episode"), ("ep_20260601_0002", "episode")],
        episodes=[ep1, ep2],
        merged_entry_id="ep_20260614_0001",
    )

    (merged,) = await ports.orchestrator().run(scope=_SCOPE)

    # Reflector was called in INIT mode (no old_episode kwarg).
    ports.reflector.areflect.assert_awaited_once()
    assert "old_episode" not in (ports.reflector.areflect.call_args.kwargs or {})
    ports.episode_store.append_many.assert_awaited_once()
    assert isinstance(merged, EpisodeExtracted)
    assert merged.source == "reflection"
    assert merged.session_id is None
    assert merged.reflection is not None
    assert merged.reflection.mode == "init"
    # Nothing is retired before the merged episode's facts exist.
    ports.episode_store.deprecate.assert_not_called()

    report = await ports.finisher().finish(
        scope=_SCOPE,
        merged_entry_id=merged.episode_entry_id,
        context=merged.reflection,
    )

    # Deprecation is one md patch for the file holding both originals.
    ports.episode_store.deprecate.assert_awaited_once_with(
        _SCOPE,
        replaced={
            "ep_20260601_0001": "ep_20260614_0001",
            "ep_20260601_0002": "ep_20260614_0001",
        },
        md_path=ep1.ref.md_path,
    )
    ports.cluster_store.remove_members.assert_awaited_once()
    ports.cluster_store.add_member.assert_awaited_once_with(
        "cl_abc", "ep_20260614_0001", "episode"
    )
    ports.report_store.create.assert_awaited_once_with(report)


async def test_run_update_mode_uses_old_episode(ports: _Ports) -> None:
    """UPDATE flow: cluster has 1 merged episode + 1 original episode."""
    old_merged = _stored_episode(
        entry_id="ep_20260612_0001",
        parent_id="cl_update",
        parent_type="cluster",
        episode="old merged text",
    )
    new_ep = _stored_episode(
        entry_id="ep_20260613_0001", parent_id="mc_004", episode="new episode text"
    )
    _arm_cluster_flow(
        ports,
        cluster_id="cl_update",
        members=[("ep_20260612_0001", "episode"), ("ep_20260613_0001", "episode")],
        episodes=[new_ep, old_merged],
        merged_entry_id="ep_20260614_0002",
        reflected={"cl_update"},
    )

    (merged,) = await ports.orchestrator().run(scope=_SCOPE)

    ports.reflector.areflect.assert_awaited_once()
    _, kwargs = ports.reflector.areflect.call_args
    assert "old_episode" in kwargs
    assert merged.reflection is not None
    assert merged.reflection.mode == "update"


async def test_llm_failure_skips_cluster(ports: _Ports) -> None:
    """Reflector raising an exception skips the cluster, continues."""
    _arm_cluster_flow(
        ports,
        cluster_id="cl_fail",
        members=[("ep_001", "episode"), ("ep_002", "episode")],
        episodes=[
            _stored_episode(entry_id="ep_001", parent_id="mc_a"),
            _stored_episode(entry_id="ep_002", parent_id="mc_b"),
        ],
        merged_entry_id="ep_m",
    )
    ports.reflector.areflect = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    merged = await ports.orchestrator().run(scope=_SCOPE)

    assert merged == []
    ports.episode_store.append_many.assert_not_called()


async def test_call_reflector_emits_consolidate_generation_span(
    ports: _Ports,
) -> None:
    """_call_reflector wraps the reflector call in an everos.reflect.consolidate
    generation span (token usage lands here via the LLM client wrapper)."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from everos.config.settings import ObservabilitySettings
    from everos.core.observability.tracing import (
        force_flush,
        init_tracing,
        shutdown_tracing,
    )

    ports.reflector.areflect = AsyncMock(
        return_value=_FakeAlgoResult(
            owner_id=None, episode="merged", subject="s", timestamp=1717200000000
        )
    )
    orch = ports.orchestrator()

    exporter = InMemorySpanExporter()
    shutdown_tracing()
    init_tracing(
        ObservabilitySettings(enabled=True, endpoint="http://collector.invalid"),
        span_processor=SimpleSpanProcessor(exporter),
    )
    try:
        result = await orch._call_reflector(
            episodes=[_stored_episode()],
            merged_entry_ids=[],
            is_update=False,
            owner_id="u_alice",
        )
        force_flush()
    finally:
        shutdown_tracing()

    assert result is not None
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "everos.reflect.consolidate" in spans
    assert (
        spans["everos.reflect.consolidate"].attributes["langfuse.observation.type"]
        == "generation"
    )


async def test_deprecation_marks_markdown_and_never_writes_the_index(
    ports: _Ports,
) -> None:
    """Originals are deprecated by md patches only; the cascade owns the index."""
    ep1 = _stored_episode(entry_id="ep_1", parent_id="mc_1", md_path="a.md")
    ep2 = _stored_episode(entry_id="ep_2", parent_id="mc_2", md_path="b.md")
    _arm_cluster_flow(
        ports,
        cluster_id="cl_abc",
        members=[("ep_1", "episode"), ("ep_2", "episode")],
        episodes=[ep1, ep2],
        merged_entry_id="ep_m",
    )
    ports.atomic_fact_index.find = AsyncMock(return_value=[_fact_hit("af_1", "f.md")])

    report = await ports.finisher().finish(
        scope=_SCOPE, merged_entry_id="ep_m", context=_context(ep1, ep2)
    )

    episode_patches = {
        c.kwargs["md_path"]: c.kwargs["replaced"]
        for c in ports.episode_store.deprecate.await_args_list
    }
    assert episode_patches == {"a.md": {"ep_1": "ep_m"}, "b.md": {"ep_2": "ep_m"}}
    ports.atomic_fact_store.deprecate.assert_awaited_once_with(
        _SCOPE, replaced={"af_1": "ep_m"}, md_path="f.md"
    )
    ports.episode_index.update.assert_not_called()
    ports.atomic_fact_index.update.assert_not_called()
    assert report is not None
    assert report.source_members == [("ep_1", "episode"), ("ep_2", "episode")]
    assert report.deprecated_fact_count == 1
    ports.report_store.create.assert_awaited_once_with(report)
    ports.cluster_store.update_metadata.assert_awaited_once()
    ports.embedder.embed.assert_awaited_once_with("merged")


async def test_finish_skips_when_no_original_member_remains(ports: _Ports) -> None:
    """Members re-read under the lock: none of the snapshot left → no-op."""
    ep1 = _stored_episode(entry_id="ep_1", parent_id="mc_1")
    ports.cluster_store.members = AsyncMock(return_value=[("ep_other", "episode")])
    ports.episode_store.deprecate = AsyncMock()
    ports.report_store.create = AsyncMock()

    report = await ports.finisher().finish(
        scope=_SCOPE, merged_entry_id="ep_m", context=_context(ep1)
    )

    assert report is None
    ports.episode_store.deprecate.assert_not_called()
    ports.report_store.create.assert_not_called()


def _context(*episodes: Stored[Episode]) -> ReflectionContext:
    return ReflectionContext(
        cluster_id="cl_abc",
        mode="init",
        source_members=[(ep.ref.entry_id, "episode") for ep in episodes],
        episodes=[
            ReflectedEpisode(
                entry_id=ep.ref.entry_id,
                md_path=ep.ref.md_path,
                parent_id=ep.item.parent_id,
            )
            for ep in episodes
        ],
        merged_episode="merged",
        last_ts=1_780_272_000_000,
    )
