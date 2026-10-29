"""The canned agent-track scripts satisfy the real operators they stand in for.

Each script under ``tests/fakes/llm_scripts/agent_track`` is fed to the real
everalgo operator through ``agent_track_llm``; a script that drifted from the
operator's strict reply model fails here instead of deep inside an
integration test. Every operator also has its empty script, which exercises
the path where it yields nothing. The case-gate scripts answer EverOS's own
judge (:class:`~everos.component.llm.LLMCaseJudge`), which the gate asks.
"""

from __future__ import annotations

import json

import numpy as np
from everalgo.agent_memory import (
    AgentCaseExtractor,
    AgentCaseGate,
    AgentSkillExtractor,
    CaseSkipReason,
)
from everalgo.boundary import adetect_boundaries
from everalgo.clustering import Cluster, acluster_by_llm
from everalgo.types import AgentCase

from everos.component.llm import LLMCaseJudge
from tests.fakes import agent_track_llm, agent_track_script, agent_trajectory

_CASE = AgentCase(
    id="ac_1",
    timestamp=1_775_116_800_000,
    task_intent="Fix the flaky date test",
    approach="reproduce, normalise to UTC, re-run",
    quality_score=0.8,
)


async def test_boundary_scripts_cut_the_trajectory_on_flush_only() -> None:
    items = agent_trajectory().items
    final = await adetect_boundaries(items, llm=agent_track_llm(), is_final=True)
    held = await adetect_boundaries(
        items, llm=agent_track_llm(boundary="boundary_empty"), is_final=False
    )
    assert [len(c.items) for c in final.cells] == [len(items)]
    assert (held.cells, len(held.tail)) == ([], len(items))


async def test_case_scripts_yield_the_compressed_case() -> None:
    result = await AgentCaseExtractor(llm=agent_track_llm()).aextract_with_reason(
        agent_trajectory()
    )
    want = json.loads(agent_track_script("case_compress"))
    (case,) = result.cases
    assert (case.task_intent, case.approach) == (want["task_intent"], want["approach"])
    assert case.quality_score == want["quality_score"]
    assert result.reason is None


async def test_empty_filter_script_rejects_the_trajectory() -> None:
    llm = agent_track_llm(case_filter="case_filter_empty")
    result = await AgentCaseExtractor(llm=llm).aextract_with_reason(agent_trajectory())
    assert (result.cases, result.reason) == ([], CaseSkipReason.FILTER_REJECTED)


async def test_empty_compress_script_yields_no_case() -> None:
    llm = agent_track_llm(case_compress="case_compress_empty")
    result = await AgentCaseExtractor(llm=llm).aextract_with_reason(agent_trajectory())
    assert (result.cases, result.reason) == ([], CaseSkipReason.COMPRESS_EMPTY_INTENT)


async def test_skill_script_adds_one_skill() -> None:
    result = await AgentSkillExtractor(llm=agent_track_llm()).aextract_with_reason(
        _CASE, existing_relevant_skills=[], supporting_cases=[]
    )
    want = json.loads(agent_track_script("skill_extract"))["operations"][0]["data"]
    (skill,) = result.skills
    assert (skill.name, skill.content) == (want["name"], want["content"])
    assert skill.source_case_ids == ["ac_1"]


async def test_empty_skill_script_proposes_nothing() -> None:
    llm = agent_track_llm(skill_extract="skill_extract_empty")
    result = await AgentSkillExtractor(llm=llm).aextract_with_reason(
        _CASE, existing_relevant_skills=[], supporting_cases=[]
    )
    assert (result.pre_reason, result.outcomes) == (None, [])


def _cluster(cluster_id: str | None, centroid: list[float]) -> Cluster:
    return Cluster(
        id=cluster_id,
        centroid=np.asarray(centroid, dtype=np.float32),
        last_ts=1,
        preview=["fix a date test"],
        members=[cluster_id or "ac_new"],
    )


async def test_cluster_scripts_merge_or_keep_apart() -> None:
    # Orthogonal centroids keep the similarity under the skip threshold, so
    # the model is asked.
    new, existing = _cluster(None, [1.0, 0.0]), [_cluster("cl_1", [0.0, 1.0])]

    merged = await acluster_by_llm(new, existing, llm=agent_track_llm())
    apart = await acluster_by_llm(
        new, existing, llm=agent_track_llm(skill_cluster="skill_cluster_empty")
    )

    assert merged is not None
    assert (merged.id, merged.members) == ("cl_1", ["cl_1", "ac_new"])
    assert apart is None


async def test_gate_scripts_admit_or_reject() -> None:
    conversation = '[{"role": "user", "content": "fix it"}]'

    admitted = await AgentCaseGate(judge=LLMCaseJudge(agent_track_llm())).aevaluate(
        conversation
    )
    rejected = await AgentCaseGate(
        judge=LLMCaseJudge(agent_track_llm(case_gate="case_gate_empty"))
    ).aevaluate(conversation)

    assert (admitted.accepted, admitted.success_probability) == (True, 0.9)
    assert (rejected.accepted, rejected.success_probability) == (False, 0.1)
