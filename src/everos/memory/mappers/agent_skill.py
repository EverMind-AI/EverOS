"""everalgo ``AgentSkill`` <-> domain :class:`~everos.memory.models.AgentSkill`."""

from __future__ import annotations

__all__ = ["from_algo", "to_algo"]

from everalgo.types import AgentSkill as AlgoAgentSkill

from everos.memory.models import AgentSkill


def from_algo(algo: AlgoAgentSkill) -> AgentSkill:
    """Domain skill with the algo skill's declared fields (extras dropped)."""
    return AgentSkill.model_validate(
        algo.model_dump(include=set(AgentSkill.model_fields))
    )


def to_algo(skill: AgentSkill) -> AlgoAgentSkill:
    return AlgoAgentSkill.model_validate(skill.model_dump())
