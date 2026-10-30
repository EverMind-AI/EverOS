"""Round trip between everalgo ``AgentSkill`` and the domain skill."""

from __future__ import annotations

from everalgo.types import AgentSkill as AlgoAgentSkill

from everos.memory.mappers import agent_skill


def test_algo_skill_round_trips_through_the_domain() -> None:
    algo = AlgoAgentSkill(
        id="sk1",
        cluster_id="cl_1",
        name="debug",
        description="how to debug",
        content="steps",
        confidence=0.7,
        maturity_score=0.8,
        source_case_ids=["c1", "c2"],
    )

    assert agent_skill.to_algo(agent_skill.from_algo(algo)) == algo


def test_algo_extras_do_not_leak_into_the_domain() -> None:
    algo = AlgoAgentSkill(id="sk1", embedding=[0.1])

    assert "embedding" not in agent_skill.from_algo(algo).model_dump()
