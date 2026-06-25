"""Tests for :func:`extract_agent_skill`.

Mocked seams: ``cluster_repo`` (sqlite), ``agent_case_repo`` /
``agent_skill_repo`` (LanceDB), ``get_embedder`` (component),
``AgentSkillExtractor`` (algo), ``AgentSkillWriter`` (md). Each
retry-class exception (cluster missing / case-not-indexed) bubbles up so
OME's ``max_retries`` machinery catches the race instead of the strategy
implementing its own backoff loop.

LanceDB repo behaviour itself (predicate isolation, cosine ranking,
``_distance`` stripping) lives under
``tests/unit/test_infra/test_lancedb/test_repos/``; strategy tests only
verify routing decisions and orchestration glue.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.types import AgentSkill as AlgoAgentSkill

from everos.component.embedding import (
    EmbeddingNotConfiguredError,
    EmbeddingServiceError,
)
from everos.infra.ome.testing import FakeStrategyContext
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.events import SkillClusterUpdated
from everos.memory.strategies.extract_agent_skill import (
    MAX_SKILLS_IN_PROMPT,
    MAX_SUPPORTING_CASES,
    _CaseNotYetIndexedError,
    _ClusterMissingError,
    _collect_supporting_entry_ids,
    _resolve_query_vector,
    _select_existing_skills,
    _select_supporting_cases,
    extract_agent_skill,
)


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


def _event(
    *,
    cluster_id: str = "cl_xxxxxxxxxxx1",
    case_entry_id: str = "ac_20260517_0001",
    agent_id: str = "agent_42",
) -> SkillClusterUpdated:
    return SkillClusterUpdated(
        case_entry_id=case_entry_id,
        cluster_id=cluster_id,
        agent_id=agent_id,
    )


def _algo_cluster(
    *,
    cluster_id: str = "cl_xxxxxxxxxxx1",
    members: list[str] | None = None,
) -> AlgoCluster:
    return AlgoCluster(
        id=cluster_id,
        centroid=np.zeros(1024, dtype=np.float32),
        count=len(members or ["ac_20260517_0001"]),
        last_ts=1_700_000_000_000,
        preview=[],
        members=members or ["ac_20260517_0001"],
    )


def _lance_case(
    entry_id: str,
    *,
    quality_score: float = 0.8,
    timestamp: _dt.datetime | None = None,
    vector: list[float] | None = None,
    task_intent: str | None = None,
) -> MagicMock:
    """Stand-in for a LanceDB AgentCase row (only fields the strategy reads)."""
    case = MagicMock()
    case.entry_id = entry_id
    case.timestamp = timestamp or _dt.datetime(2026, 5, 17, tzinfo=_dt.UTC)
    case.task_intent = (
        task_intent if task_intent is not None else f"intent of {entry_id}"
    )
    case.approach = f"approach of {entry_id}"
    case.quality_score = quality_score
    case.key_insight = ""
    case.vector = vector or []
    return case


def _lance_skill(
    *,
    name: str = "old_skill",
    cluster_id: str = "cl_xxxxxxxxxxx1",
    source_case_ids: list[str] | None = None,
) -> MagicMock:
    skill = MagicMock()
    skill.id = f"agent_42_{name}"
    skill.cluster_id = cluster_id
    skill.name = name
    skill.description = f"desc {name}"
    skill.content = f"content {name}"
    skill.confidence = 0.5
    skill.maturity_score = 0.5
    skill.source_case_ids = source_case_ids or []
    return skill


def _algo_skill(name: str = "summarise_doc") -> AlgoAgentSkill:
    return AlgoAgentSkill(
        id="dummyuuid",
        cluster_id="",  # caller will post-stamp
        name=name,
        description=f"how to {name}",
        content="full body of the skill",
        confidence=0.7,
        maturity_score=0.5,
        source_case_ids=["ac_20260517_0001"],
    )


# ── strategy meta + retry-class errors ───────────────────────────────────


async def test_strategy_meta_is_attached() -> None:
    meta = extract_agent_skill.meta
    assert meta.name == "extract_agent_skill"
    assert SkillClusterUpdated in meta.trigger.on
    assert meta.emits == frozenset()
    assert meta.max_retries == 3


async def test_raises_when_cluster_missing_for_retry() -> None:
    """No cluster row yet — OME will retry the run."""
    with patch(
        "everos.memory.strategies.extract_agent_skill.cluster_repo"
    ) as mock_repo:
        mock_repo.get_with_members = AsyncMock(return_value=None)
        with pytest.raises(_ClusterMissingError):
            await extract_agent_skill(_event(), FakeStrategyContext())


async def test_raises_when_target_case_not_yet_in_lancedb() -> None:
    """LanceDB has not yet indexed the freshly-written case — let OME retry."""
    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_case_repo"
        ) as mock_case_repo,
    ):
        mock_cluster_repo.get_with_members = AsyncMock(return_value=_algo_cluster())
        mock_case_repo.find_by_owner_entry = AsyncMock(return_value=None)
        with pytest.raises(_CaseNotYetIndexedError):
            await extract_agent_skill(_event(), FakeStrategyContext())


# ── end-to-end orchestration (mocked) ────────────────────────────────────


@pytest.mark.asyncio
async def test_extracts_and_persists_with_cluster_id_stamped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end (mocked): extractor emits skills → writer stamps cluster_id."""
    target = _lance_case("ac_20260517_0001", vector=[0.1] * 1024)
    supporting = [_lance_case("ac_20260517_0000")]
    existing = [_lance_skill(name="old_skill", source_case_ids=["ac_20260517_0000"])]
    emitted = [_algo_skill(name="summarise_doc"), _algo_skill(name="batch_then_synth")]

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.cluster_repo"
        ) as mock_cluster_repo,
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
    ):
        mock_cluster_repo.get_with_members = AsyncMock(
            return_value=_algo_cluster(members=["ac_20260517_0000", "ac_20260517_0001"])
        )
        mock_case_repo.find_by_owner_entry = AsyncMock(return_value=target)
        mock_case_repo.find_by_owner_entries = AsyncMock(return_value=supporting)
        # Small cluster path: count ≤ K → scalar fetch returns existing.
        mock_skill_repo.count_in_cluster = AsyncMock(return_value=len(existing))
        mock_skill_repo.find_in_cluster = AsyncMock(return_value=existing)
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=emitted)
        mock_writer_cls.return_value.write_main = AsyncMock(return_value=None)
        mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")
        monkeypatch.setattr(mod, "_writer", None, raising=False)

        await extract_agent_skill(_event(), FakeStrategyContext())

    extractor_call = mock_extractor_cls.return_value.aextract.call_args
    target_arg = extractor_call.args[0]
    assert target_arg.id == "ac_20260517_0001"
    assert target_arg.task_intent == "intent of ac_20260517_0001"
    assert [s.name for s in extractor_call.kwargs["existing_relevant_skills"]] == [
        "old_skill"
    ]
    assert [c.id for c in extractor_call.kwargs["supporting_cases"]] == [
        "ac_20260517_0000"
    ]

    write_calls = mock_writer_cls.return_value.write_main.call_args_list
    assert len(write_calls) == 2
    for call, expected in zip(write_calls, emitted, strict=True):
        agent_id_arg, skill_name_arg = call.args
        fm = call.kwargs["frontmatter"]
        assert agent_id_arg == "agent_42"
        assert skill_name_arg == expected.name
        assert fm.cluster_id == "cl_xxxxxxxxxxx1"
        assert fm.name == expected.name
        assert fm.confidence == expected.confidence
        assert call.kwargs["body"] == expected.content


# ── _select_existing_skills routing (cluster size × vector availability) ─


async def test_select_existing_skills_small_cluster_uses_scalar_fetch() -> None:
    """``total ≤ K`` short-circuits — no ranking needed for fully-inclusive set."""
    target = _lance_case("ac_001", vector=[0.5] * 1024)
    skills = [_lance_skill(name=f"s{i}") for i in range(3)]

    with patch(
        "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
    ) as mock_repo:
        mock_repo.count_in_cluster = AsyncMock(return_value=3)
        mock_repo.find_in_cluster = AsyncMock(return_value=skills)
        mock_repo.find_topk_relevant_in_cluster = AsyncMock()

        got = await _select_existing_skills(
            agent_id="a", cluster_id="cl_x", target=target
        )

    assert got == skills
    mock_repo.find_topk_relevant_in_cluster.assert_not_awaited()
    mock_repo.find_in_cluster.assert_awaited_once_with(
        owner_id="a", cluster_id="cl_x", limit=MAX_SKILLS_IN_PROMPT
    )


async def test_select_existing_skills_large_cluster_with_vector_uses_topk() -> None:
    """``total > K`` and target carries vector → cosine top-K path."""
    target = _lance_case("ac_001", vector=[0.5] * 1024)
    topk_skills = [_lance_skill(name=f"s{i}") for i in range(MAX_SKILLS_IN_PROMPT)]

    with patch(
        "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
    ) as mock_repo:
        mock_repo.count_in_cluster = AsyncMock(return_value=MAX_SKILLS_IN_PROMPT + 5)
        mock_repo.find_topk_relevant_in_cluster = AsyncMock(return_value=topk_skills)
        mock_repo.find_in_cluster = AsyncMock()

        got = await _select_existing_skills(
            agent_id="a", cluster_id="cl_x", target=target
        )

    assert got == topk_skills
    mock_repo.find_in_cluster.assert_not_awaited()
    call_kwargs = mock_repo.find_topk_relevant_in_cluster.await_args.kwargs
    assert call_kwargs["query_vector"] == [0.5] * 1024
    assert call_kwargs["top_k"] == MAX_SKILLS_IN_PROMPT


async def test_select_existing_skills_large_cluster_recomputes_embedding() -> None:
    """``total > K`` but case has no vector → re-embed ``task_intent`` on the fly."""
    target = _lance_case("ac_001", vector=[], task_intent="how to summarise docs")
    topk_skills = [_lance_skill(name=f"s{i}") for i in range(MAX_SKILLS_IN_PROMPT)]
    fresh_vec = [0.42] * 1024

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(return_value=fresh_vec)

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.get_embedder",
            return_value=mock_embedder,
        ),
    ):
        mock_repo.count_in_cluster = AsyncMock(return_value=MAX_SKILLS_IN_PROMPT + 5)
        mock_repo.find_topk_relevant_in_cluster = AsyncMock(return_value=topk_skills)
        mock_repo.find_in_cluster = AsyncMock()

        got = await _select_existing_skills(
            agent_id="a", cluster_id="cl_x", target=target
        )

    assert got == topk_skills
    mock_embedder.embed.assert_awaited_once_with("how to summarise docs")
    call_kwargs = mock_repo.find_topk_relevant_in_cluster.await_args.kwargs
    assert call_kwargs["query_vector"] == fresh_vec


async def test_select_existing_skills_falls_back_to_scalar_when_embed_fails() -> None:
    """``total > K`` + no vector + embedder fails → scalar fetch capped at K."""
    target = _lance_case("ac_001", vector=[], task_intent="how to summarise docs")
    scalar_skills = [_lance_skill(name=f"s{i}") for i in range(MAX_SKILLS_IN_PROMPT)]

    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(side_effect=EmbeddingServiceError("provider down"))

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.get_embedder",
            return_value=mock_embedder,
        ),
    ):
        mock_repo.count_in_cluster = AsyncMock(return_value=MAX_SKILLS_IN_PROMPT + 5)
        mock_repo.find_in_cluster = AsyncMock(return_value=scalar_skills)
        mock_repo.find_topk_relevant_in_cluster = AsyncMock()

        got = await _select_existing_skills(
            agent_id="a", cluster_id="cl_x", target=target
        )

    assert got == scalar_skills
    mock_repo.find_topk_relevant_in_cluster.assert_not_awaited()
    mock_repo.find_in_cluster.assert_awaited_once_with(
        owner_id="a", cluster_id="cl_x", limit=MAX_SKILLS_IN_PROMPT
    )


# ── _resolve_query_vector layered fallback ───────────────────────────────


async def test_resolve_query_vector_prefers_persisted_vector() -> None:
    """When ``target.vector`` is set, reuse it; never call the embedder."""
    target = _lance_case("ac_001", vector=[0.3] * 1024)
    with patch(
        "everos.memory.strategies.extract_agent_skill.get_embedder"
    ) as mock_get_embedder:
        got = await _resolve_query_vector(target)
    assert got == [0.3] * 1024
    mock_get_embedder.assert_not_called()


async def test_resolve_query_vector_returns_empty_when_no_text_either() -> None:
    """No persisted vector + no task_intent → ``[]`` (no policy here)."""
    target = _lance_case("ac_001", vector=[], task_intent="")
    with patch(
        "everos.memory.strategies.extract_agent_skill.get_embedder"
    ) as mock_get_embedder:
        got = await _resolve_query_vector(target)
    assert got == []
    mock_get_embedder.assert_not_called()


async def test_resolve_query_vector_swallows_embedder_not_configured() -> None:
    """Missing embedder config is a deployment issue, not a strategy fault."""
    target = _lance_case("ac_001", vector=[], task_intent="hello")
    mock_embedder = MagicMock()
    mock_embedder.embed = AsyncMock(
        side_effect=EmbeddingNotConfiguredError("no api key")
    )
    with patch(
        "everos.memory.strategies.extract_agent_skill.get_embedder",
        return_value=mock_embedder,
    ):
        got = await _resolve_query_vector(target)
    assert got == []


# ── _select_supporting_cases ranking + cap ───────────────────────────────


async def test_select_supporting_cases_ranks_by_quality_then_timestamp() -> None:
    """Hydrated cases sort ``(quality_score desc, timestamp desc)``."""
    skills = [
        _lance_skill(name="s1", source_case_ids=["ac_a", "ac_b", "ac_c"]),
    ]
    case_a = _lance_case(
        "ac_a",
        quality_score=0.4,
        timestamp=_dt.datetime(2026, 5, 1, tzinfo=_dt.UTC),
    )
    case_b = _lance_case(
        "ac_b",
        quality_score=0.9,
        timestamp=_dt.datetime(2026, 5, 1, tzinfo=_dt.UTC),
    )
    case_c = _lance_case(
        "ac_c",
        quality_score=0.9,
        timestamp=_dt.datetime(2026, 5, 10, tzinfo=_dt.UTC),
    )

    with patch(
        "everos.memory.strategies.extract_agent_skill.agent_case_repo"
    ) as mock_case_repo:
        # Order intentionally scrambled to prove the strategy sorts.
        mock_case_repo.find_by_owner_entries = AsyncMock(
            return_value=[case_a, case_b, case_c]
        )

        got = await _select_supporting_cases(
            skills,
            agent_id="a",
            exclude_entry_id="ac_target",
            app_id="default",
            project_id="default",
        )

    assert [c.entry_id for c in got] == ["ac_c", "ac_b", "ac_a"]


async def test_select_supporting_cases_caps_at_max_supporting() -> None:
    """Hydrated set is truncated to ``MAX_SUPPORTING_CASES``."""
    ids = [f"ac_{i:03d}" for i in range(MAX_SUPPORTING_CASES + 3)]
    skills = [_lance_skill(name="s1", source_case_ids=ids)]
    hydrated = [
        _lance_case(eid, quality_score=0.5 + 0.01 * i) for i, eid in enumerate(ids)
    ]

    with patch(
        "everos.memory.strategies.extract_agent_skill.agent_case_repo"
    ) as mock_case_repo:
        mock_case_repo.find_by_owner_entries = AsyncMock(return_value=hydrated)
        got = await _select_supporting_cases(
            skills,
            agent_id="a",
            exclude_entry_id="ac_target",
            app_id="default",
            project_id="default",
        )

    assert len(got) == MAX_SUPPORTING_CASES


async def test_select_supporting_cases_skips_repo_when_no_lineage_ids() -> None:
    """No usable source ids → ``[]`` without a repo round trip."""
    skills = [_lance_skill(name="s1", source_case_ids=[])]
    with patch(
        "everos.memory.strategies.extract_agent_skill.agent_case_repo"
    ) as mock_case_repo:
        mock_case_repo.find_by_owner_entries = AsyncMock()
        got = await _select_supporting_cases(
            skills,
            agent_id="a",
            exclude_entry_id="ac_target",
            app_id="default",
            project_id="default",
        )
    assert got == []
    mock_case_repo.find_by_owner_entries.assert_not_awaited()


# ── _collect_supporting_entry_ids dedup + exclude ────────────────────────


def test_collect_supporting_entry_ids_dedups_and_excludes_target() -> None:
    """Source ids fold across skills; duplicates and the target id drop out."""
    skill_a = MagicMock()
    skill_a.source_case_ids = ["ac_a", "ac_b", "ac_target"]
    skill_b = MagicMock()
    skill_b.source_case_ids = ["ac_b", "ac_c"]  # ac_b duplicates skill_a's lineage
    skill_empty = MagicMock()
    skill_empty.source_case_ids = []

    got = _collect_supporting_entry_ids(
        [skill_a, skill_b, skill_empty], exclude="ac_target"
    )
    assert got == ["ac_a", "ac_b", "ac_c"]


def test_collect_supporting_entry_ids_handles_empty_input() -> None:
    """No skills → no supporting cases."""
    assert _collect_supporting_entry_ids([], exclude="ac_anything") == []


# ── partition lock (agent_id-level serialisation) ────────────────────────


async def _run_serialisation_probe(
    agent_id_run_a: str, agent_id_run_b: str
) -> list[str]:
    """Drive two extract_agent_skill runs and record their critical-section order.

    Mocks every I/O seam so the only async work inside the locked region
    is a tiny ``asyncio.sleep`` masquerading as the LLM call. The returned
    log is the strict enter/leave sequence both runs go through.
    """
    log: list[str] = []

    async def mock_aextract(case, **_kwargs):
        log.append(f"enter:{case.id}")
        await asyncio.sleep(0.01)
        log.append(f"leave:{case.id}")
        return []

    target_a = _lance_case("ac_run_a", vector=[0.1] * 1024)
    target_b = _lance_case("ac_run_b", vector=[0.1] * 1024)

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.cluster_repo"
        ) as mock_cluster_repo,
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
        patch("everos.memory.strategies.extract_agent_skill.AgentSkillWriter"),
    ):
        mock_cluster_repo.get_with_members = AsyncMock(
            return_value=_algo_cluster(members=["ac_run_a", "ac_run_b"])
        )
        mock_case_repo.find_by_owner_entry = AsyncMock(
            side_effect=lambda owner, entry, **_kw: (
                target_a if entry == "ac_run_a" else target_b
            )
        )
        mock_case_repo.find_by_owner_entries = AsyncMock(return_value=[])
        mock_skill_repo.count_in_cluster = AsyncMock(return_value=0)
        mock_skill_repo.find_in_cluster = AsyncMock(return_value=[])
        mock_extractor_cls.return_value.aextract = mock_aextract
        await asyncio.gather(
            extract_agent_skill(
                _event(agent_id=agent_id_run_a, case_entry_id="ac_run_a"),
                FakeStrategyContext(),
            ),
            extract_agent_skill(
                _event(agent_id=agent_id_run_b, case_entry_id="ac_run_b"),
                FakeStrategyContext(),
            ),
        )
    return log


async def test_partition_lock_serialises_runs_on_same_agent() -> None:
    """Two runs sharing ``agent_id`` must not overlap critical sections."""
    log = await _run_serialisation_probe("agent_42", "agent_42")
    assert log in (
        ["enter:ac_run_a", "leave:ac_run_a", "enter:ac_run_b", "leave:ac_run_b"],
        ["enter:ac_run_b", "leave:ac_run_b", "enter:ac_run_a", "leave:ac_run_a"],
    )


async def test_partition_lock_lets_different_agents_run_in_parallel() -> None:
    """Runs on distinct ``agent_id`` must overlap (no false serialisation)."""
    log = await _run_serialisation_probe("agent_42", "agent_43")
    assert log.index("enter:ac_run_a") < log.index("leave:ac_run_b")
    assert log.index("enter:ac_run_b") < log.index("leave:ac_run_a")
