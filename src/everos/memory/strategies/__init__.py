"""OME business strategies — event-triggered offline tasks.

Each strategy is an async function ``fn(event, deps) -> list[Event]``
declared next to its :class:`~everos.memory.ports.StrategySpec`.
:data:`STRATEGY_SPECS` is the complete set the service registers; strategy
names are stable because ``ome.toml`` overrides and run records key on them.

External usage:
    from everos.memory.strategies import STRATEGY_SPECS, extract_atomic_facts
"""

from __future__ import annotations

from everos.memory.ports import StrategySpec

from .extract_agent_case import EXTRACT_AGENT_CASE_SPEC as EXTRACT_AGENT_CASE_SPEC
from .extract_agent_case import extract_agent_case as extract_agent_case
from .extract_agent_skill import EXTRACT_AGENT_SKILL_SPEC as EXTRACT_AGENT_SKILL_SPEC
from .extract_agent_skill import extract_agent_skill as extract_agent_skill
from .extract_atomic_facts import EXTRACT_ATOMIC_FACTS_SPEC as EXTRACT_ATOMIC_FACTS_SPEC
from .extract_atomic_facts import extract_atomic_facts as extract_atomic_facts
from .extract_foresight import EXTRACT_FORESIGHT_SPEC as EXTRACT_FORESIGHT_SPEC
from .extract_foresight import extract_foresight as extract_foresight
from .extract_user_profile import EXTRACT_USER_PROFILE_SPEC as EXTRACT_USER_PROFILE_SPEC
from .extract_user_profile import extract_user_profile as extract_user_profile
from .finish_reflection import FINISH_REFLECTION_SPEC as FINISH_REFLECTION_SPEC
from .finish_reflection import finish_reflection as finish_reflection
from .reflect_episodes import REFLECT_EPISODES_SPEC as REFLECT_EPISODES_SPEC
from .reflect_episodes import reflect_episodes as reflect_episodes
from .trigger_profile_clustering import (
    TRIGGER_PROFILE_CLUSTERING_SPEC as TRIGGER_PROFILE_CLUSTERING_SPEC,
)
from .trigger_profile_clustering import (
    trigger_profile_clustering as trigger_profile_clustering,
)
from .trigger_skill_clustering import (
    TRIGGER_SKILL_CLUSTERING_SPEC as TRIGGER_SKILL_CLUSTERING_SPEC,
)
from .trigger_skill_clustering import (
    trigger_skill_clustering as trigger_skill_clustering,
)

STRATEGY_SPECS: tuple[StrategySpec, ...] = (
    EXTRACT_ATOMIC_FACTS_SPEC,
    EXTRACT_FORESIGHT_SPEC,
    EXTRACT_AGENT_CASE_SPEC,
    EXTRACT_USER_PROFILE_SPEC,
    TRIGGER_SKILL_CLUSTERING_SPEC,
    EXTRACT_AGENT_SKILL_SPEC,
    TRIGGER_PROFILE_CLUSTERING_SPEC,
    REFLECT_EPISODES_SPEC,
    FINISH_REFLECTION_SPEC,
)
"""Every strategy the service registers, in registration order."""

__all__ = [
    "EXTRACT_AGENT_CASE_SPEC",
    "EXTRACT_AGENT_SKILL_SPEC",
    "EXTRACT_ATOMIC_FACTS_SPEC",
    "EXTRACT_FORESIGHT_SPEC",
    "EXTRACT_USER_PROFILE_SPEC",
    "FINISH_REFLECTION_SPEC",
    "REFLECT_EPISODES_SPEC",
    "STRATEGY_SPECS",
    "TRIGGER_PROFILE_CLUSTERING_SPEC",
    "TRIGGER_SKILL_CLUSTERING_SPEC",
    "extract_agent_case",
    "extract_agent_skill",
    "extract_atomic_facts",
    "extract_foresight",
    "extract_user_profile",
    "finish_reflection",
    "reflect_episodes",
    "trigger_profile_clustering",
    "trigger_skill_clustering",
]
