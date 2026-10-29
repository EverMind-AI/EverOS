"""The strategy package declares every strategy exactly once."""

from __future__ import annotations

from everos.memory.strategies import (
    STRATEGY_SPECS,
    extract_agent_case,
    extract_agent_skill,
    extract_atomic_facts,
    extract_foresight,
    extract_user_profile,
    finish_reflection,
    reflect_episodes,
    trigger_profile_clustering,
    trigger_skill_clustering,
)


def test_every_strategy_function_has_one_spec_under_its_name() -> None:
    by_name = {spec.name: spec.fn for spec in STRATEGY_SPECS}
    assert len(by_name) == len(STRATEGY_SPECS)
    assert by_name == {
        "extract_atomic_facts": extract_atomic_facts,
        "extract_foresight": extract_foresight,
        "extract_agent_case": extract_agent_case,
        "trigger_skill_clustering": trigger_skill_clustering,
        "extract_agent_skill": extract_agent_skill,
        "trigger_profile_clustering": trigger_profile_clustering,
        "extract_user_profile": extract_user_profile,
        "reflect_episodes": reflect_episodes,
        "finish_reflection": finish_reflection,
    }
