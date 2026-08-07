"""Tests for :func:`extract_agent_skill`.

Mocked seams: ``cluster_repo`` (sqlite), ``agent_case_repo`` (LanceDB,
supporting-cases lineage only), ``agent_skill_repo`` (LanceDB, relevance
ranking only — never an existence check), :class:`AgentSkillReader` /
:class:`AgentSkillWriter` (md — the source of truth for which skills exist),
:class:`EmbeddingCapability` (component, injected via
:func:`_install_embedder`), ``AgentSkillExtractor`` (algo).

The target case is reconstructed straight from the ``SkillClusterUpdated``
event payload (:func:`_to_algo_case_from_event`) — the strategy never probes
LanceDB for it, so the old case-not-yet-indexed retry path no longer exists.
Only the cluster-missing race (``_ClusterMissingError``) still bubbles up
for OME's ``max_retries`` machinery to catch.

LanceDB repo behaviour itself (predicate isolation, cosine ranking,
``_distance`` stripping) lives under
``tests/unit/test_infra/test_lancedb/test_repos/``; ``AgentSkillReader`` /
``AgentSkillWriter`` behaviour lives under
``tests/unit/test_infra/test_markdown/test_readers/``. Strategy tests only
verify routing decisions and orchestration glue.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import structlog.testing
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.types import AgentSkill as AlgoAgentSkill

from everos.component.embedding import EmbeddingCapability, EmbeddingProvider
from everos.core.persistence import MemoryRoot
from everos.infra.ome.testing import FakeStrategyContext
from everos.infra.persistence.markdown import (
    AgentSkillFrontmatter,
    AgentSkillWriter,
)
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.events import SkillClusterUpdated
from everos.memory.strategies.extract_agent_skill import (
    MAX_SKILLS_IN_PROMPT,
    MAX_SUPPORTING_CASES,
    _ClusterMissingError,
    _collect_supporting_entry_ids,
    _reap_renamed_skills,
    _select_existing_skills,
    _select_supporting_cases,
    extract_agent_skill,
)


class _StubEmbedder(EmbeddingProvider):
    """Minimal ``EmbeddingProvider`` for tests that only need the body-guard
    to pass and never inspect the embedded vector."""

    dim = 1024

    async def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def _install_embedder(
    monkeypatch: pytest.MonkeyPatch, embedder: EmbeddingProvider
) -> None:
    """Install ``embedder`` as the process-wide embedding capability.

    ``extract_agent_skill`` resolves the embedder via
    ``get_embedding_capability().available`` for its body-guard only (the
    query vector itself now travels on the event, so nothing here embeds on
    the fly). The autouse fixture in ``tests/conftest.py`` seeds
    ``Capability(provider=None)`` for hermeticity; this helper swaps in a
    live provider so the guard passes.
    """
    import everos.component.embedding.accessor as acc

    monkeypatch.setattr(acc, "_capability", EmbeddingCapability(provider=embedder))


@pytest.fixture
def embed_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convenience fixture: install a no-op stub embedder as the capability."""
    _install_embedder(monkeypatch, _StubEmbedder())


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


def _event(
    *,
    cluster_id: str = "cl_xxxxxxxxxxx1",
    case_entry_id: str = "ac_20260517_0001",
    agent_id: str = "agent_42",
    app_id: str = "default",
    project_id: str = "default",
    task_intent: str = "",
    approach: str = "",
    key_insight: str | None = None,
    quality_score: float = 0.0,
    case_timestamp_ms: int = 0,
    case_vector: list[float] | None = None,
) -> SkillClusterUpdated:
    return SkillClusterUpdated(
        case_entry_id=case_entry_id,
        cluster_id=cluster_id,
        agent_id=agent_id,
        app_id=app_id,
        project_id=project_id,
        task_intent=task_intent,
        approach=approach,
        key_insight=key_insight,
        quality_score=quality_score,
        case_timestamp_ms=case_timestamp_ms,
        case_vector=case_vector,
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
) -> MagicMock:
    """Stand-in for a LanceDB AgentCase row (supporting-cases lineage only)."""
    case = MagicMock()
    case.entry_id = entry_id
    case.timestamp = timestamp or _dt.datetime(2026, 5, 17, tzinfo=_dt.UTC)
    case.task_intent = f"intent of {entry_id}"
    case.approach = f"approach of {entry_id}"
    case.quality_score = quality_score
    case.key_insight = ""
    return case


def _frontmatter(
    name: str,
    *,
    agent_id: str = "a",
    cluster_id: str | None = "cl_x",
    source_case_ids: list[str] | None = None,
    confidence: float = 0.5,
    maturity_score: float = 0.5,
) -> AgentSkillFrontmatter:
    return AgentSkillFrontmatter(
        id=f"{agent_id}_{name}",
        agent_id=agent_id,
        name=name,
        description=f"desc {name}",
        confidence=confidence,
        maturity_score=maturity_score,
        source_case_ids=source_case_ids or [],
        cluster_id=cluster_id,
    )


def _reader_stub(fms: list[AgentSkillFrontmatter]) -> MagicMock:
    """Reader double: ``list_by_cluster`` returns each frontmatter paired
    with a synthetic body — mirroring the real ``AgentSkillReader``, which
    returns ``(frontmatter, body)`` pairs directly rather than requiring a
    second, name-based read to hydrate ``content``."""
    reader = MagicMock()
    reader.list_by_cluster = AsyncMock(
        return_value=[(fm, f"body of {fm.name}") for fm in fms]
    )
    return reader


def _lance_skill_row(name: str) -> MagicMock:
    """Stand-in for a LanceDB AgentSkill ranking row (only ``.name`` is read)."""
    row = MagicMock()
    row.name = name
    return row


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


# ── strategy meta + cluster-missing retry ────────────────────────────────


async def test_strategy_meta_is_attached() -> None:
    meta = extract_agent_skill.meta
    assert meta.name == "extract_agent_skill"
    assert SkillClusterUpdated in meta.trigger.on
    assert meta.emits == frozenset()
    assert meta.max_retries == 3


async def test_returns_without_side_effects_when_embedding_unavailable() -> None:
    """Capability unavailable → early return; no cluster fetch, no writes.

    Belt-and-suspenders with the upstream ``trigger_skill_clustering``
    gate: a direct emit of ``SkillClusterUpdated`` (tests, future
    features) should degrade cleanly without an OME retry.
    """
    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.get_embedding_capability",
            return_value=EmbeddingCapability(provider=None),
        ),
        patch("everos.memory.strategies.extract_agent_skill.cluster_repo") as mock_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_case_repo"
        ) as mock_case_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_skill_repo,
        structlog.testing.capture_logs() as captured,
    ):
        mock_repo.get_with_members = AsyncMock(
            side_effect=AssertionError("cluster_repo must not be touched"),
        )
        mock_case_repo.find_by_owner_entries = AsyncMock(
            side_effect=AssertionError("agent_case_repo must not be touched"),
        )
        mock_skill_repo.find_topk_relevant_in_cluster = AsyncMock(
            side_effect=AssertionError("agent_skill_repo must not be touched"),
        )

        await extract_agent_skill(_event(), FakeStrategyContext())

    gated = [
        e
        for e in captured
        if e.get("event") == "strategy_gated_off_embedding_unavailable"
    ]
    assert len(gated) == 1
    assert gated[0]["strategy_name"] == "extract_agent_skill"
    assert gated[0]["agent_id"] == "agent_42"


async def test_raises_when_cluster_missing_for_retry(embed_available: None) -> None:
    """No cluster row yet — OME will retry the run."""
    with patch(
        "everos.memory.strategies.extract_agent_skill.cluster_repo"
    ) as mock_repo:
        mock_repo.get_with_members = AsyncMock(return_value=None)
        with pytest.raises(_ClusterMissingError):
            await extract_agent_skill(_event(), FakeStrategyContext())


# ── target case + existing skills regression (event-read, md-first) ─────


async def test_reads_target_case_from_event_no_lancedb_probe(
    monkeypatch: pytest.MonkeyPatch,
    embed_available: None,
) -> None:
    """Strategy body reads task_intent / approach / key_insight straight off
    the SkillClusterUpdated event. It never calls
    agent_case_repo.find_by_owner_entry for the target case — that probe was
    the cascade-lag corruption path this PR rescues: under sustained
    cascade lag the run died after ``max_retries`` and OME dead-lettered it,
    so the fresh case was never distilled into a skill.
    """
    mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")
    probe_calls: list[str] = []

    async def fake_find(*args: object, **kwargs: object) -> None:
        probe_calls.append("find_by_owner_entry")
        return None

    monkeypatch.setattr(mod.agent_case_repo, "find_by_owner_entry", fake_find)

    mock_reader = MagicMock()
    mock_reader.list_by_cluster = AsyncMock(return_value=[])
    monkeypatch.setattr(mod, "_reader", mock_reader)
    monkeypatch.setattr(mod, "_writer", None, raising=False)

    emitted = [_algo_skill(name="revive_replica")]

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.cluster_repo"
        ) as mock_cluster_repo,
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
        mock_cluster_repo.get_with_members = AsyncMock(return_value=_algo_cluster())
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=emitted)
        mock_writer_cls.return_value.write_main = AsyncMock(return_value=None)

        event = _event(
            case_entry_id="c1",
            cluster_id="cl1",
            agent_id="a1",
            task_intent="restore replica",
            approach="stop, resync, verify",
            key_insight="watch oplog",
            quality_score=0.8,
            case_timestamp_ms=1_700_000_000_000,
            case_vector=[0.1] * 1024,
        )
        await extract_agent_skill(event, FakeStrategyContext())

    assert probe_calls == []  # LanceDB never touched for the target case

    extractor_call = mock_extractor_cls.return_value.aextract.call_args
    target_arg = extractor_call.args[0]
    assert target_arg.id == "c1"
    assert target_arg.task_intent == "restore replica"
    assert target_arg.approach == "stop, resync, verify"
    assert target_arg.key_insight == "watch oplog"
    assert target_arg.quality_score == 0.8
    assert target_arg.timestamp == 1_700_000_000_000


async def test_existing_skills_come_from_md_even_when_lancedb_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    embed_available: None,
) -> None:
    """LanceDB reports 0 skills in the cluster, but md has skill_X — strategy
    sees skill_X via the reader and feeds it to the extractor as an existing
    skill. Regresses the silent-clobber bug where a stale index made the LLM
    see no existing skill and emit ``add()`` for one that already existed.
    """
    mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")
    monkeypatch.setattr(
        MemoryRoot, "resolve", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    monkeypatch.setattr(mod, "_writer", None, raising=False)
    monkeypatch.setattr(mod, "_reader", None, raising=False)

    seed_writer = AgentSkillWriter(root=MemoryRoot(root=tmp_path))
    await seed_writer.write_main(
        "a1",
        "revive_replica",
        frontmatter=_frontmatter(
            "revive_replica",
            agent_id="a1",
            cluster_id="cl1",
            source_case_ids=["c0"],
            maturity_score=0.6,
        ),
        body="## Revive replica\nstop, resync, verify.",
    )

    captured: dict[str, list] = {}

    async def spy_aextract(target, *, existing_relevant_skills, supporting_cases):
        captured["existing"] = list(existing_relevant_skills)
        return []

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_skill_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_case_repo"
        ) as mock_case_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_agent_skill.AgentSkillExtractor"
        ) as mock_extractor_cls,
    ):
        mock_cluster_repo.get_with_members = AsyncMock(
            return_value=_algo_cluster(cluster_id="cl1", members=["c0", "c1"])
        )
        # Simulate cascade lag: LanceDB's skill ranking index is stale/empty.
        # Cluster is within MAX_SKILLS_IN_PROMPT, so this must never be hit.
        mock_skill_repo.find_topk_relevant_in_cluster = AsyncMock(
            side_effect=AssertionError(
                "must not be reached: cluster is within MAX_SKILLS_IN_PROMPT"
            )
        )
        mock_case_repo.find_by_owner_entries = AsyncMock(return_value=[])
        mock_extractor_cls.return_value.aextract = spy_aextract

        event = _event(
            cluster_id="cl1",
            agent_id="a1",
            case_entry_id="c1",
            case_vector=[0.1] * 1024,
        )
        await extract_agent_skill(event, FakeStrategyContext())

    assert len(captured["existing"]) == 1
    hydrated = captured["existing"][0]
    assert hydrated.name == "revive_replica"
    assert hydrated.source_case_ids == ["c0"]
    assert hydrated.content != ""
    assert "stop, resync, verify" in hydrated.content


async def test_existing_skills_reaches_llm_for_skill_whose_directory_has_a_space(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    embed_available: None,
) -> None:
    """End-to-end regression guard for the ``list_by_cluster`` drop bug: a
    ``skill_My Skill/`` directory written outside the writer (raw space,
    never sanitized) must reach ``existing_relevant_skills`` with
    non-empty ``content`` — not merely be enumerable by ``list_by_cluster``
    in isolation, but survive the full enumeration→hydration path this
    strategy drives without a second, name-based read dropping it one
    layer downstream.

    A prior fix made ``list_by_cluster`` enumerate this directory
    successfully, but ``_hydrate_algo_skills`` then re-read each selected
    skill by ``fm.name`` via ``read_main`` — which re-derives (and
    re-sanitizes) a path from ``"My Skill"`` to ``skill_My_Skill/``, a
    path that doesn't exist, and dropped the skill again. The fix removed
    that second read entirely: ``list_by_cluster`` now hands back the
    body it already read, so there is no name-based re-derivation left
    anywhere on this path.
    """
    mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")
    monkeypatch.setattr(
        MemoryRoot, "resolve", classmethod(lambda cls: MemoryRoot(root=tmp_path))
    )
    monkeypatch.setattr(mod, "_writer", None, raising=False)
    monkeypatch.setattr(mod, "_reader", None, raising=False)

    skill_dir = (
        MemoryRoot(root=tmp_path).agents_dir() / "a1" / "skills" / "skill_My Skill"
    )
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "id: a1_My Skill\n"
        "type: agent_skill\n"
        "agent_id: a1\n"
        "track: agent\n"
        "name: My Skill\n"
        "description: d\n"
        "confidence: 0.5\n"
        "maturity_score: 0.5\n"
        "cluster_id: cl1\n"
        "---\n"
        "The real skill body.\n",
        encoding="utf-8",
    )

    captured: dict[str, list] = {}

    async def spy_aextract(target, *, existing_relevant_skills, supporting_cases):
        captured["existing"] = list(existing_relevant_skills)
        return []

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.cluster_repo"
        ) as mock_cluster_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_skill_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_case_repo"
        ) as mock_case_repo,
        patch(
            "everos.memory.strategies.extract_agent_skill.get_llm_client",
            return_value=object(),
        ),
        patch(
            "everos.memory.strategies.extract_agent_skill.AgentSkillExtractor"
        ) as mock_extractor_cls,
    ):
        mock_cluster_repo.get_with_members = AsyncMock(
            return_value=_algo_cluster(cluster_id="cl1", members=["c0", "c1"])
        )
        mock_skill_repo.find_topk_relevant_in_cluster = AsyncMock(
            side_effect=AssertionError(
                "must not be reached: cluster is within MAX_SKILLS_IN_PROMPT"
            )
        )
        mock_case_repo.find_by_owner_entries = AsyncMock(return_value=[])
        mock_extractor_cls.return_value.aextract = spy_aextract

        event = _event(
            cluster_id="cl1",
            agent_id="a1",
            case_entry_id="c1",
            case_vector=[0.1] * 1024,
        )
        await extract_agent_skill(event, FakeStrategyContext())

    assert len(captured["existing"]) == 1
    hydrated = captured["existing"][0]
    assert hydrated.name == "My Skill"
    assert hydrated.content == "The real skill body."


# ── end-to-end orchestration (mocked) ────────────────────────────────────


async def test_extracts_and_persists_with_cluster_id_stamped(
    monkeypatch: pytest.MonkeyPatch,
    embed_available: None,
) -> None:
    """End-to-end (mocked): target reconstructed from event, existing skill
    comes from md, extractor emits skills → writer stamps cluster_id."""
    existing_fm = _frontmatter(
        "old_skill",
        agent_id="agent_42",
        cluster_id="cl_xxxxxxxxxxx1",
        source_case_ids=["ac_20260517_0000"],
    )
    supporting = [_lance_case("ac_20260517_0000")]
    emitted = [_algo_skill(name="summarise_doc"), _algo_skill(name="batch_then_synth")]

    mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")
    monkeypatch.setattr(mod, "_reader", _reader_stub([existing_fm]))
    monkeypatch.setattr(mod, "_writer", None, raising=False)

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
        mock_case_repo.find_by_owner_entries = AsyncMock(return_value=supporting)
        mock_skill_repo.find_topk_relevant_in_cluster = AsyncMock(
            side_effect=AssertionError(
                "must not rank: cluster within MAX_SKILLS_IN_PROMPT"
            )
        )
        mock_extractor_cls.return_value.aextract = AsyncMock(return_value=emitted)
        mock_writer_cls.return_value.write_main = AsyncMock(return_value=None)

        event = _event(
            task_intent="intent of ac_20260517_0001",
            approach="approach of ac_20260517_0001",
            quality_score=0.8,
            case_vector=[0.1] * 1024,
        )
        await extract_agent_skill(event, FakeStrategyContext())

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


async def test_select_existing_skills_small_cluster_returns_all_md_skills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``total ≤ K`` → every md skill is used; LanceDB is never asked to rank."""
    mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")
    fms = [_frontmatter(f"s{i}") for i in range(3)]
    monkeypatch.setattr(mod, "_reader", _reader_stub(fms))

    with patch(
        "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
    ) as mock_skill_repo:
        mock_skill_repo.find_topk_relevant_in_cluster = AsyncMock()

        got = await _select_existing_skills(
            agent_id="a",
            cluster_id="cl_x",
            app_id="default",
            project_id="default",
            case_vector=[0.5] * 1024,
        )

    assert [s.name for s in got] == ["s0", "s1", "s2"]
    assert all(s.content == f"body of {s.name}" for s in got)
    mock_skill_repo.find_topk_relevant_in_cluster.assert_not_awaited()


async def test_select_existing_skills_large_cluster_with_vector_uses_lancedb_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``total > K`` + ``case_vector`` present → LanceDB ranks, md hydrates body."""
    mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")
    n = MAX_SKILLS_IN_PROMPT + 5
    fms = [_frontmatter(f"s{i}") for i in range(n)]
    monkeypatch.setattr(mod, "_reader", _reader_stub(fms))

    ranked_names = [f"s{i}" for i in range(MAX_SKILLS_IN_PROMPT)]

    with patch(
        "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
    ) as mock_skill_repo:
        mock_skill_repo.find_topk_relevant_in_cluster = AsyncMock(
            return_value=[_lance_skill_row(name) for name in ranked_names]
        )

        got = await _select_existing_skills(
            agent_id="a",
            cluster_id="cl_x",
            app_id="default",
            project_id="default",
            case_vector=[0.5] * 1024,
        )

    assert [s.name for s in got] == ranked_names
    assert all(s.content == f"body of {s.name}" for s in got)
    call_kwargs = mock_skill_repo.find_topk_relevant_in_cluster.await_args.kwargs
    assert call_kwargs["query_vector"] == [0.5] * 1024
    assert call_kwargs["top_k"] == MAX_SKILLS_IN_PROMPT


async def test_select_existing_skills_large_cluster_no_vector_falls_back_to_md_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``total > K`` + no ``case_vector`` → md ordering, capped at K, logged."""
    mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")
    n = MAX_SKILLS_IN_PROMPT + 5
    fms = [_frontmatter(f"s{i}") for i in range(n)]
    monkeypatch.setattr(mod, "_reader", _reader_stub(fms))

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
        ) as mock_skill_repo,
        structlog.testing.capture_logs() as captured,
    ):
        mock_skill_repo.find_topk_relevant_in_cluster = AsyncMock(
            side_effect=AssertionError("must not rank without a case_vector")
        )

        got = await _select_existing_skills(
            agent_id="a",
            cluster_id="cl_x",
            app_id="default",
            project_id="default",
            case_vector=None,
        )

    assert [s.name for s in got] == [f"s{i}" for i in range(MAX_SKILLS_IN_PROMPT)]
    warned = [
        e
        for e in captured
        if e.get("event") == "agent_skill_topk_no_query_vector_md_fallback"
    ]
    assert len(warned) == 1
    assert warned[0]["md_count"] == n


async def test_select_existing_skills_appends_md_remainder_when_lancedb_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LanceDB ranking is itself cascade-lagged: it may return a name md no
    longer has (skipped), and may omit md names it does have (appended as
    filler) — neither may silently drop a skill from the prompt."""
    mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")
    n = MAX_SKILLS_IN_PROMPT + 2
    fms = [_frontmatter(f"s{i}") for i in range(n)]
    monkeypatch.setattr(mod, "_reader", _reader_stub(fms))

    # Ranked list omits "s0" (stale) and includes a ghost id md no longer has.
    ranked_names = [f"s{i}" for i in range(1, MAX_SKILLS_IN_PROMPT)] + ["ghost_skill"]

    with patch(
        "everos.memory.strategies.extract_agent_skill.agent_skill_repo"
    ) as mock_skill_repo:
        mock_skill_repo.find_topk_relevant_in_cluster = AsyncMock(
            return_value=[_lance_skill_row(name) for name in ranked_names]
        )

        got = await _select_existing_skills(
            agent_id="a",
            cluster_id="cl_x",
            app_id="default",
            project_id="default",
            case_vector=[0.1] * 1024,
        )

    names = [s.name for s in got]
    assert len(names) == MAX_SKILLS_IN_PROMPT
    assert "ghost_skill" not in names  # stale lance row silently skipped
    assert names[:-1] == [f"s{i}" for i in range(1, MAX_SKILLS_IN_PROMPT)]
    assert names[-1] == "s0"  # dropped-by-lance md skill appended, not lost


# ── _select_supporting_cases ranking + cap ───────────────────────────────


async def test_select_supporting_cases_ranks_by_quality_then_timestamp() -> None:
    """Hydrated cases sort ``(quality_score desc, timestamp desc)``."""
    skills = [_algo_skill(name="s1")]
    skills[0].source_case_ids = ["ac_a", "ac_b", "ac_c"]
    case_a = _lance_case(
        "ac_a", quality_score=0.4, timestamp=_dt.datetime(2026, 5, 1, tzinfo=_dt.UTC)
    )
    case_b = _lance_case(
        "ac_b", quality_score=0.9, timestamp=_dt.datetime(2026, 5, 1, tzinfo=_dt.UTC)
    )
    case_c = _lance_case(
        "ac_c", quality_score=0.9, timestamp=_dt.datetime(2026, 5, 10, tzinfo=_dt.UTC)
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
    skills = [_algo_skill(name="s1")]
    skills[0].source_case_ids = ids
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
    skills = [_algo_skill(name="s1")]
    skills[0].source_case_ids = []
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
    monkeypatch: pytest.MonkeyPatch,
    agent_id_run_a: str,
    agent_id_run_b: str,
) -> list[str]:
    """Drive two extract_agent_skill runs and record their critical-section order.

    Mocks every I/O seam so the only async work inside the locked region
    is a tiny ``asyncio.sleep`` masquerading as the LLM call. The returned
    log is the strict enter/leave sequence both runs go through.
    """
    mod = importlib.import_module("everos.memory.strategies.extract_agent_skill")
    log: list[str] = []

    async def mock_aextract(case, **_kwargs):
        log.append(f"enter:{case.id}")
        await asyncio.sleep(0.01)
        log.append(f"leave:{case.id}")
        return []

    mock_reader = MagicMock()
    mock_reader.list_by_cluster = AsyncMock(return_value=[])
    monkeypatch.setattr(mod, "_reader", mock_reader)
    monkeypatch.setattr(mod, "_writer", None, raising=False)

    with (
        patch(
            "everos.memory.strategies.extract_agent_skill.cluster_repo"
        ) as mock_cluster_repo,
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


async def test_partition_lock_serialises_runs_on_same_agent(
    monkeypatch: pytest.MonkeyPatch,
    embed_available: None,
) -> None:
    """Two runs sharing ``agent_id`` must not overlap critical sections."""
    log = await _run_serialisation_probe(monkeypatch, "agent_42", "agent_42")
    assert log in (
        ["enter:ac_run_a", "leave:ac_run_a", "enter:ac_run_b", "leave:ac_run_b"],
        ["enter:ac_run_b", "leave:ac_run_b", "enter:ac_run_a", "leave:ac_run_a"],
    )


async def test_partition_lock_lets_different_agents_run_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
    embed_available: None,
) -> None:
    """Runs on distinct ``agent_id`` must overlap (no false serialisation)."""
    log = await _run_serialisation_probe(monkeypatch, "agent_42", "agent_43")
    assert log.index("enter:ac_run_a") < log.index("leave:ac_run_b")
    assert log.index("enter:ac_run_b") < log.index("leave:ac_run_a")


# ── rename reconciliation (orphan directories) ──────────────────────────


def _identified_algo_skill(skill_id: str, name: str) -> AlgoAgentSkill:
    """Like :func:`_algo_skill` but with an explicit id — rename
    reconciliation keys off identity, so these tests must control it."""
    return AlgoAgentSkill(
        id=skill_id,
        cluster_id="cl1",
        name=name,
        description="d",
        content="body",
        confidence=0.8,
        maturity_score=0.5,
        source_case_ids=["case_a"],
    )


async def _write_skill(writer: AgentSkillWriter, name: str) -> Path:
    fm = AgentSkillFrontmatter(
        id=f"agent_42_{name}",
        agent_id="agent_42",
        name=name,
        description="d",
        confidence=0.8,
        maturity_score=0.5,
        cluster_id="cl1",
    )
    path = await writer.write_main("agent_42", name, frontmatter=fm, body="body")
    return path.parent


async def test_reap_removes_the_directory_a_rename_left_behind(
    tmp_path: Path,
) -> None:
    """An update that renames a skill must not leave its old directory.

    everalgo's ``_apply_update`` keeps ``prior.id`` while changing the
    name, so the emitted skill is written under a new directory and the
    old one would survive carrying the same ``cluster_id``. That is not a
    cosmetic leak: since this release the next extraction's
    ``existing_relevant_skills`` come from the markdown enumeration, so the
    orphan returns as a duplicate of a skill the LLM already renamed,
    which is how ``add``-instead-of-``update`` full-replace clobbering gets
    back in. Uses a real writer on a real tmp_path — the property under
    test is that the directory is gone from the filesystem.
    """
    writer = AgentSkillWriter(MemoryRoot(tmp_path))
    old_dir = await _write_skill(writer, "fix_django")
    new_dir = await _write_skill(writer, "fix_django_autoreload")
    assert old_dir.is_dir() and new_dir.is_dir()

    await _reap_renamed_skills(
        writer,
        {"agent_42_fix_django": "fix_django_autoreload"},
        existing_skills=[_identified_algo_skill("agent_42_fix_django", "fix_django")],
        agent_id="agent_42",
        app_id="default",
        project_id="default",
    )

    assert not old_dir.exists()
    assert (new_dir / "SKILL.md").is_file()


async def test_reap_keeps_a_prior_name_another_emitted_skill_claimed(
    tmp_path: Path,
) -> None:
    """Never delete a directory this same batch just wrote.

    With two ops in one extraction — rename ``a`` → ``b`` while a second
    op writes ``a`` — reaping ``a`` by prior name would remove a file
    written moments earlier in the same loop. The claimed-name guard is
    what prevents the reap from undoing its own caller.
    """
    writer = AgentSkillWriter(MemoryRoot(tmp_path))
    dir_a = await _write_skill(writer, "alpha")
    await _write_skill(writer, "beta")

    await _reap_renamed_skills(
        writer,
        {"agent_42_alpha": "beta", "other_id": "alpha"},
        existing_skills=[_identified_algo_skill("agent_42_alpha", "alpha")],
        agent_id="agent_42",
        app_id="default",
        project_id="default",
    )

    assert dir_a.is_dir()


async def test_reap_ignores_newly_added_skills(tmp_path: Path) -> None:
    """A fresh ``add`` carries a uuid4 id absent from the enumerated set.

    Identity is the only thing that survives a rename — ``_apply_update``
    preserves ``prior.id`` while ``_apply_add`` mints a new one — so an
    id that never appeared in ``existing_skills`` cannot be a rename, and
    nothing may be deleted on its account.
    """
    writer = AgentSkillWriter(MemoryRoot(tmp_path))
    kept = await _write_skill(writer, "existing_skill")

    await _reap_renamed_skills(
        writer,
        {"3f2a9c1e4b6d47f8a0c5e9b2d7143a6f": "brand_new_skill"},
        existing_skills=[
            _identified_algo_skill("agent_42_existing_skill", "existing_skill")
        ],
        agent_id="agent_42",
        app_id="default",
        project_id="default",
    )

    assert kept.is_dir()
