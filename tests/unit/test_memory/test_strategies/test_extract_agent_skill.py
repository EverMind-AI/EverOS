"""Tests for :func:`extract_agent_skill`.

Ports come from :mod:`tests.fakes` (cluster store, skill store and index,
agent-case index); the markdown skill store is used for real where the
property under test is on-disk (directory enumeration, rename reaping).
``AgentSkillExtractor`` is patched on the strategy module.

The target case is reconstructed straight from the ``SkillClusterUpdated``
event payload (:func:`_to_algo_case_from_event`) — the strategy never reads
the index for it. Only the cluster-missing race (``_ClusterMissingError``)
still bubbles up for OME's ``max_retries`` machinery to catch.

Store / index behaviour itself (markdown enumeration, cosine ranking) is
tested with the adapters; strategy tests only verify routing decisions and
orchestration glue.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog.testing
from everalgo.types import AgentSkill as AlgoAgentSkill

from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import MarkdownAgentSkillStore
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.deps import MemoryDeps
from everos.memory.events import SkillClusterUpdated
from everos.memory.models import AgentCase, AgentSkill, Cluster
from everos.memory.ports import MemoryRef, Scope, Stored
from everos.memory.strategies import EXTRACT_AGENT_SKILL_SPEC, extract_agent_skill
from everos.memory.strategies.extract_agent_skill import (
    MAX_SKILLS_IN_PROMPT,
    MAX_SUPPORTING_CASES,
    _ClusterMissingError,
    _collect_supporting_entry_ids,
    _reap_renamed_skills,
    _select_existing_skills,
    _select_supporting_cases,
)
from tests.fakes import make_deps, make_indexes, make_stores

_EXTRACTOR = "everos.memory.strategies.extract_agent_skill.AgentSkillExtractor"
_AGENT = Scope(
    app_id="default", project_id="default", owner_id="agent_42", owner_type="agent"
)
_A1 = Scope(app_id="default", project_id="default", owner_id="a1", owner_type="agent")
_A = Scope(app_id="default", project_id="default", owner_id="a", owner_type="agent")


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


def _seed_cluster(
    deps: MemoryDeps,
    scope: Scope = _AGENT,
    *,
    cluster_id: str = "cl_xxxxxxxxxxx1",
    members: list[str] | None = None,
) -> None:
    members = members or ["ac_20260517_0001"]
    deps.stores.cluster.seed(
        scope,
        "agent_case",
        Cluster(
            id=cluster_id,
            centroid=(0.0,) * 8,
            count=len(members),
            last_ts=1_700_000_000_000,
            members=members,
        ),
    )


def _skill(
    name: str,
    *,
    agent_id: str = "a",
    cluster_id: str = "cl_x",
    source_case_ids: list[str] | None = None,
) -> AgentSkill:
    return AgentSkill(
        id=f"{agent_id}_{name}",
        cluster_id=cluster_id,
        name=name,
        description=f"desc {name}",
        content=f"body of {name}",
        confidence=0.5,
        maturity_score=0.5,
        source_case_ids=source_case_ids or [],
    )


def _indexed_skill(
    scope: Scope, name: str, cluster_id: str = "cl_x"
) -> Stored[AgentSkill]:
    skill = _skill(name, agent_id=scope.owner_id, cluster_id=cluster_id)
    return Stored(
        ref=MemoryRef(id=skill.id, entry_id=name, md_path=f"{name}.md"),
        scope=scope,
        item=skill,
    )


def _indexed_case(
    scope: Scope, entry_id: str, *, quality_score: float = 0.8, timestamp: int = 1
) -> Stored[AgentCase]:
    return Stored(
        ref=MemoryRef(id=f"{scope.owner_id}_{entry_id}", entry_id=entry_id, md_path=""),
        scope=scope,
        item=AgentCase(
            owner_id=scope.owner_id,
            task_intent=f"intent of {entry_id}",
            approach=f"approach of {entry_id}",
            quality_score=quality_score,
            timestamp=timestamp,
            session_id="s1",
            parent_id="mc_1",
        ),
    )


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


def _markdown_deps(tmp_path: Path) -> MemoryDeps:
    return make_deps(
        stores=make_stores(agent_skill=MarkdownAgentSkillStore(MemoryRoot(tmp_path)))
    )


# ── spec + body-guard + cluster-missing retry ────────────────────────────


def test_spec() -> None:
    spec = EXTRACT_AGENT_SKILL_SPEC
    assert spec.name == "extract_agent_skill"
    assert SkillClusterUpdated in spec.trigger.on
    assert spec.emits == frozenset()
    assert spec.max_retries == 3


async def test_returns_without_side_effects_when_embedding_unavailable() -> None:
    """No embedding provider → early return; no cluster read, no writes.

    Belt-and-suspenders with the upstream ``trigger_skill_clustering``
    gate: a direct ``SkillClusterUpdated`` (tests, future features) should
    degrade cleanly without an OME retry.
    """
    deps = make_deps(embedding=None)
    with (
        patch(_EXTRACTOR) as extractor_cls,
        structlog.testing.capture_logs() as captured,
    ):
        events = await extract_agent_skill(_event(), deps)

    assert events == []
    extractor_cls.assert_not_called()
    gated = [
        e
        for e in captured
        if e.get("event") == "strategy_gated_off_embedding_unavailable"
    ]
    assert len(gated) == 1
    assert gated[0]["strategy_name"] == "extract_agent_skill"
    assert gated[0]["agent_id"] == "agent_42"


async def test_raises_when_cluster_missing_for_retry() -> None:
    """No cluster row yet — OME will retry the run."""
    with pytest.raises(_ClusterMissingError):
        await extract_agent_skill(_event(), make_deps())


# ── target case + existing skills regression (event-read, md-first) ─────


async def test_reads_target_case_from_event_no_index_probe() -> None:
    """Strategy body reads task_intent / approach / key_insight straight off
    the SkillClusterUpdated event. It never looks the target case up in the
    agent-case index — that probe was the cascade-lag corruption path: under
    sustained cascade lag the run died after ``max_retries`` and OME
    dead-lettered it, so the fresh case was never distilled into a skill.
    """
    case_index = MagicMock()
    case_index.find_by_entry_ids = AsyncMock(
        side_effect=AssertionError("the target case must not be read from the index")
    )
    deps = make_deps(indexes=make_indexes(agent_case=case_index))
    _seed_cluster(deps, _A1, cluster_id="cl1")
    event = _event(
        case_entry_id="c1",
        cluster_id="cl1",
        agent_id="a1",
        task_intent="restore replica",
        approach="stop, resync, verify",
        key_insight="watch oplog",
        quality_score=0.8,
        case_timestamp_ms=1_700_000_000_000,
        case_vector=[0.1] * 8,
    )
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract = AsyncMock(
            return_value=[_algo_skill(name="revive_replica")]
        )
        await extract_agent_skill(event, deps)

    target_arg = extractor_cls.return_value.aextract.call_args.args[0]
    assert target_arg.id == "c1"
    assert target_arg.task_intent == "restore replica"
    assert target_arg.approach == "stop, resync, verify"
    assert target_arg.key_insight == "watch oplog"
    assert target_arg.quality_score == 0.8
    assert target_arg.timestamp == 1_700_000_000_000


async def _existing_skills_seen_by_llm(deps: MemoryDeps) -> list[AlgoAgentSkill]:
    captured: list[AlgoAgentSkill] = []

    async def spy_aextract(target, *, existing_relevant_skills, supporting_cases):
        captured.extend(existing_relevant_skills)
        return []

    _seed_cluster(deps, _A1, cluster_id="cl1", members=["c0", "c1"])
    event = _event(
        cluster_id="cl1", agent_id="a1", case_entry_id="c1", case_vector=[0.1] * 8
    )
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract = spy_aextract
        await extract_agent_skill(event, deps)
    return captured


async def test_existing_skills_come_from_md_even_when_index_stale(
    tmp_path: Path,
) -> None:
    """The skill index reports nothing for the cluster, but md has the skill —
    the strategy sees it through the skill store and feeds it to the
    extractor as an existing skill. Regresses the silent-clobber bug where a
    stale index made the LLM see no existing skill and emit ``add()`` for
    one that already existed.
    """
    deps = _markdown_deps(tmp_path)
    await deps.stores.agent_skill.write(
        _A1,
        AgentSkill(
            id="a1_revive_replica",
            cluster_id="cl1",
            name="revive_replica",
            description="d",
            content="## Revive replica\nstop, resync, verify.",
            maturity_score=0.6,
            source_case_ids=["c0"],
        ),
    )

    (hydrated,) = await _existing_skills_seen_by_llm(deps)
    assert hydrated.name == "revive_replica"
    assert hydrated.source_case_ids == ["c0"]
    assert "stop, resync, verify" in hydrated.content
    assert deps.indexes.agent_skill.top_k_calls == []


async def test_existing_skills_reaches_llm_for_skill_whose_directory_has_a_space(
    tmp_path: Path,
) -> None:
    """End-to-end regression guard for the ``list_by_cluster`` drop bug: a
    ``skill_My Skill/`` directory written outside the writer (raw space,
    never sanitized) must reach ``existing_relevant_skills`` with
    non-empty ``content`` — not merely be enumerable in isolation, but
    survive the full enumeration→hydration path this strategy drives
    without a second, name-based read dropping it one layer downstream.
    """
    skill_dir = MemoryRoot(root=tmp_path).agents_dir() / "a1" / "skills"
    (skill_dir / "skill_My Skill").mkdir(parents=True)
    (skill_dir / "skill_My Skill" / "SKILL.md").write_text(
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

    (hydrated,) = await _existing_skills_seen_by_llm(_markdown_deps(tmp_path))
    assert hydrated.name == "My Skill"
    assert hydrated.content == "The real skill body."


# ── end-to-end orchestration ─────────────────────────────────────────────


async def test_extracts_and_persists_with_cluster_id_stamped() -> None:
    """Target reconstructed from event, existing skill comes from the store,
    extractor emits skills → each is written with the cluster id stamped."""
    deps = make_deps()
    _seed_cluster(deps, members=["ac_20260517_0000", "ac_20260517_0001"])
    deps.stores.agent_skill.seed(
        _AGENT,
        _skill(
            "old_skill",
            agent_id="agent_42",
            cluster_id="cl_xxxxxxxxxxx1",
            source_case_ids=["ac_20260517_0000"],
        ),
    )
    deps.indexes.agent_case.seed(_indexed_case(_AGENT, "ac_20260517_0000"))
    emitted = [_algo_skill(name="summarise_doc"), _algo_skill(name="batch_then_synth")]
    event = _event(
        task_intent="intent of ac_20260517_0001",
        approach="approach of ac_20260517_0001",
        quality_score=0.8,
        case_vector=[0.1] * 8,
    )
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract = AsyncMock(return_value=emitted)
        await extract_agent_skill(event, deps)

    extractor_call = extractor_cls.return_value.aextract.call_args
    assert extractor_call.args[0].id == "ac_20260517_0001"
    assert extractor_call.args[0].task_intent == "intent of ac_20260517_0001"
    assert [s.name for s in extractor_call.kwargs["existing_relevant_skills"]] == [
        "old_skill"
    ]
    assert [c.id for c in extractor_call.kwargs["supporting_cases"]] == [
        "ac_20260517_0000"
    ]
    assert deps.indexes.agent_skill.top_k_calls == []  # within MAX_SKILLS_IN_PROMPT

    written = deps.stores.agent_skill.skills[_AGENT]
    for expected in emitted:
        skill = written[expected.name]
        assert skill.cluster_id == "cl_xxxxxxxxxxx1"
        assert skill.confidence == expected.confidence
        assert skill.content == expected.content


# ── _select_existing_skills routing (cluster size × vector availability) ─


def _seed_md_skills(deps: MemoryDeps, count: int) -> None:
    deps.stores.agent_skill.seed(_A, *(_skill(f"s{i}") for i in range(count)))


async def test_select_existing_skills_small_cluster_returns_all_md_skills() -> None:
    """``total ≤ K`` → every md skill is used; the index is never asked to rank."""
    deps = make_deps()
    _seed_md_skills(deps, 3)

    got = await _select_existing_skills(deps, _A, "cl_x", [0.5] * 8)

    assert [s.item.name for s in got] == ["s0", "s1", "s2"]
    assert all(s.item.content == f"body of {s.item.name}" for s in got)
    assert deps.indexes.agent_skill.top_k_calls == []


async def test_select_existing_skills_large_cluster_ranks_by_index() -> None:
    """``total > K`` + ``case_vector`` present → the index ranks, md hydrates."""
    deps = make_deps()
    _seed_md_skills(deps, MAX_SKILLS_IN_PROMPT + 5)
    ranked_names = [f"s{i}" for i in reversed(range(MAX_SKILLS_IN_PROMPT))]
    deps.indexes.agent_skill.seed(*(_indexed_skill(_A, n) for n in ranked_names))

    got = await _select_existing_skills(deps, _A, "cl_x", [0.5] * 8)

    assert [s.item.name for s in got] == ranked_names
    assert all(s.item.content == f"body of {s.item.name}" for s in got)
    ((scope, cluster_id, vector, k),) = deps.indexes.agent_skill.top_k_calls
    assert (scope, cluster_id, list(vector), k) == (
        _A,
        "cl_x",
        [0.5] * 8,
        MAX_SKILLS_IN_PROMPT,
    )


async def test_select_existing_skills_no_vector_falls_back_to_md_order() -> None:
    """``total > K`` + no ``case_vector`` → md ordering, capped at K, logged."""
    deps = make_deps()
    n = MAX_SKILLS_IN_PROMPT + 5
    _seed_md_skills(deps, n)

    with structlog.testing.capture_logs() as captured:
        got = await _select_existing_skills(deps, _A, "cl_x", None)

    assert [s.item.name for s in got] == [f"s{i}" for i in range(MAX_SKILLS_IN_PROMPT)]
    assert deps.indexes.agent_skill.top_k_calls == []
    warned = [
        e
        for e in captured
        if e.get("event") == "agent_skill_topk_no_query_vector_md_fallback"
    ]
    assert len(warned) == 1
    assert warned[0]["md_count"] == n


async def test_select_existing_skills_appends_md_remainder_when_index_stale() -> None:
    """The index ranking is itself cascade-lagged: it may return a name md no
    longer has (skipped), and may omit md names it does have (appended as
    filler) — neither may silently drop a skill from the prompt."""
    deps = make_deps()
    _seed_md_skills(deps, MAX_SKILLS_IN_PROMPT + 2)
    # Ranked list omits "s0" (stale) and includes a ghost md no longer has.
    ranked_names = [f"s{i}" for i in range(1, MAX_SKILLS_IN_PROMPT)] + ["ghost_skill"]
    deps.indexes.agent_skill.seed(*(_indexed_skill(_A, n) for n in ranked_names))

    got = await _select_existing_skills(deps, _A, "cl_x", [0.1] * 8)

    names = [s.item.name for s in got]
    assert len(names) == MAX_SKILLS_IN_PROMPT
    assert "ghost_skill" not in names  # stale index row silently skipped
    assert names[:-1] == [f"s{i}" for i in range(1, MAX_SKILLS_IN_PROMPT)]
    assert names[-1] == "s0"  # md skill the index dropped is appended, not lost


# ── _select_supporting_cases ranking + cap ───────────────────────────────


def _stored_skill_with_lineage(source_case_ids: list[str]) -> Stored[AgentSkill]:
    skill = _skill("s1", source_case_ids=source_case_ids)
    return Stored(
        ref=MemoryRef(id=skill.id, entry_id="s1", md_path=""), scope=_A, item=skill
    )


async def test_select_supporting_cases_ranks_by_quality_then_timestamp() -> None:
    """Hydrated cases sort ``(quality_score desc, timestamp desc)``."""
    deps = make_deps()
    # Seeded out of order to prove the strategy sorts.
    deps.indexes.agent_case.seed(
        _indexed_case(_A, "ac_a", quality_score=0.4, timestamp=1),
        _indexed_case(_A, "ac_b", quality_score=0.9, timestamp=1),
        _indexed_case(_A, "ac_c", quality_score=0.9, timestamp=10),
    )

    got = await _select_supporting_cases(
        deps,
        _A,
        [_stored_skill_with_lineage(["ac_a", "ac_b", "ac_c"])],
        exclude_entry_id="ac_target",
    )

    assert [c.ref.entry_id for c in got] == ["ac_c", "ac_b", "ac_a"]


async def test_select_supporting_cases_caps_at_max_supporting() -> None:
    """Hydrated set is truncated to ``MAX_SUPPORTING_CASES``."""
    deps = make_deps()
    ids = [f"ac_{i:03d}" for i in range(MAX_SUPPORTING_CASES + 3)]
    deps.indexes.agent_case.seed(
        *(
            _indexed_case(_A, eid, quality_score=0.5 + 0.01 * i)
            for i, eid in enumerate(ids)
        )
    )

    got = await _select_supporting_cases(
        deps, _A, [_stored_skill_with_lineage(ids)], exclude_entry_id="ac_target"
    )

    assert len(got) == MAX_SUPPORTING_CASES


async def test_select_supporting_cases_skips_index_when_no_lineage_ids() -> None:
    """No usable source ids → ``[]`` without an index round trip."""
    case_index = MagicMock()
    case_index.find_by_entry_ids = AsyncMock()
    deps = make_deps(indexes=make_indexes(agent_case=case_index))

    got = await _select_supporting_cases(
        deps, _A, [_stored_skill_with_lineage([])], exclude_entry_id="ac_target"
    )

    assert got == []
    case_index.find_by_entry_ids.assert_not_awaited()


# ── _collect_supporting_entry_ids dedup + exclude ────────────────────────


def test_collect_supporting_entry_ids_dedups_and_excludes_target() -> None:
    """Source ids fold across skills; duplicates and the target id drop out."""
    skills = [
        _skill("a", source_case_ids=["ac_a", "ac_b", "ac_target"]),
        _skill("b", source_case_ids=["ac_b", "ac_c"]),  # ac_b duplicates a's lineage
        _skill("empty"),
    ]
    assert _collect_supporting_entry_ids(skills, exclude="ac_target") == [
        "ac_a",
        "ac_b",
        "ac_c",
    ]


def test_collect_supporting_entry_ids_handles_empty_input() -> None:
    """No skills → no supporting cases."""
    assert _collect_supporting_entry_ids([], exclude="ac_anything") == []


# ── partition lock (agent_id-level serialisation) ────────────────────────


async def _run_serialisation_probe(agent_a: str, agent_b: str) -> list[str]:
    """Drive two extract_agent_skill runs and record their critical-section order.

    The only async work inside the locked region is a tiny ``asyncio.sleep``
    masquerading as the LLM call. The returned log is the strict
    enter/leave sequence both runs go through.
    """
    log: list[str] = []

    async def slow_aextract(case, **_kwargs):
        log.append(f"enter:{case.id}")
        await asyncio.sleep(0.01)
        log.append(f"leave:{case.id}")
        return []

    deps = make_deps()
    for agent in {agent_a, agent_b}:
        scope = Scope(
            app_id="default", project_id="default", owner_id=agent, owner_type="agent"
        )
        _seed_cluster(deps, scope, members=["ac_run_a", "ac_run_b"])
    with patch(_EXTRACTOR) as extractor_cls:
        extractor_cls.return_value.aextract = slow_aextract
        await asyncio.gather(
            extract_agent_skill(
                _event(agent_id=agent_a, case_entry_id="ac_run_a"), deps
            ),
            extract_agent_skill(
                _event(agent_id=agent_b, case_entry_id="ac_run_b"), deps
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


# ── rename reconciliation (orphan directories) ──────────────────────────


async def _write_skill(store: MarkdownAgentSkillStore, name: str) -> Stored[AgentSkill]:
    skill = AgentSkill(
        id=f"agent_42_{name}",
        cluster_id="cl1",
        name=name,
        description="d",
        content="body",
        confidence=0.8,
        maturity_score=0.5,
    )
    ref = await store.write(_AGENT, skill)
    return Stored(ref=ref, scope=_AGENT, item=skill)


def _skill_dir(tmp_path: Path, name: str) -> Path:
    return (
        MemoryRoot(root=tmp_path).agents_dir() / "agent_42" / "skills" / f"skill_{name}"
    )


async def test_reap_removes_the_directory_a_rename_left_behind(
    tmp_path: Path,
) -> None:
    """An update that renames a skill must not leave its old directory.

    everalgo's ``_apply_update`` keeps ``prior.id`` while changing the
    name, so the emitted skill is written under a new directory and the
    old one would survive carrying the same ``cluster_id``. That is not a
    cosmetic leak: the next extraction's ``existing_relevant_skills`` come
    from the markdown enumeration, so the orphan returns as a duplicate of
    a skill the LLM already renamed, which is how ``add``-instead-of-
    ``update`` full-replace clobbering gets back in. Uses the real markdown
    store on a real tmp_path — the property under test is that the
    directory is gone from the filesystem.
    """
    store = MarkdownAgentSkillStore(MemoryRoot(tmp_path))
    prior = await _write_skill(store, "fix_django")
    await _write_skill(store, "fix_django_autoreload")

    await _reap_renamed_skills(
        store, _AGENT, {"agent_42_fix_django": "fix_django_autoreload"}, [prior]
    )

    assert not _skill_dir(tmp_path, "fix_django").exists()
    assert (_skill_dir(tmp_path, "fix_django_autoreload") / "SKILL.md").is_file()


async def test_reap_keeps_a_prior_name_another_emitted_skill_claimed(
    tmp_path: Path,
) -> None:
    """Never delete a directory this same batch just wrote.

    With two ops in one extraction — rename ``a`` → ``b`` while a second
    op writes ``a`` — reaping ``a`` by prior name would remove a file
    written moments earlier in the same loop. The claimed-name guard is
    what prevents the reap from undoing its own caller.
    """
    store = MarkdownAgentSkillStore(MemoryRoot(tmp_path))
    prior = await _write_skill(store, "alpha")
    await _write_skill(store, "beta")

    await _reap_renamed_skills(
        store, _AGENT, {"agent_42_alpha": "beta", "other_id": "alpha"}, [prior]
    )

    assert _skill_dir(tmp_path, "alpha").is_dir()


async def test_reap_ignores_newly_added_skills(tmp_path: Path) -> None:
    """A fresh ``add`` carries a uuid4 id absent from the enumerated set.

    Identity is the only thing that survives a rename — ``_apply_update``
    preserves ``prior.id`` while ``_apply_add`` mints a new one — so an
    id that never appeared in ``existing_skills`` cannot be a rename, and
    nothing may be deleted on its account.
    """
    store = MarkdownAgentSkillStore(MemoryRoot(tmp_path))
    prior = await _write_skill(store, "existing_skill")

    await _reap_renamed_skills(
        store, _AGENT, {"3f2a9c1e4b6d47f8a0c5e9b2d7143a6f": "brand_new_skill"}, [prior]
    )

    assert _skill_dir(tmp_path, "existing_skill").is_dir()
