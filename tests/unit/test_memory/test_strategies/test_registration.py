"""The strategy package declares every strategy exactly once, and routes an
episode to the strategies of its space."""

from __future__ import annotations

import importlib

import pytest

from everos.config import Settings
from everos.memory.events import EpisodeExtracted
from everos.memory.ports import Idle, Immediate, StrategySpec
from everos.memory.strategies import (
    STRATEGY_SPECS,
    extract_user_profile,
    update_group_profile,
    update_group_scenes,
    update_member_profiles,
    update_user_scenes,
)
from tests.fakes import make_deps


def test_every_strategy_function_has_one_spec_under_its_name() -> None:
    by_name = {spec.name: spec.fn for spec in STRATEGY_SPECS}
    assert len(by_name) == len(STRATEGY_SPECS)
    assert by_name == {
        "extract_user_profile": extract_user_profile,
        "update_user_scenes": update_user_scenes,
        "update_member_profiles": update_member_profiles,
        "update_group_scenes": update_group_scenes,
        "update_group_profile": update_group_profile,
        "update_group_profile_idle": update_group_profile,
    }


def _episode(owner_type: str) -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id="mc_1",
        episode_entry_id="ep_1",
        episode_text="text",
        episode_timestamp_ms=1,
        owner_id="trip" if owner_type == "group" else "u_ann",
        owner_type=owner_type,  # type: ignore[arg-type]
        app_id="a",
        project_id="p",
    )


def _admits(spec: StrategySpec, event: EpisodeExtracted) -> bool:
    if not isinstance(spec.trigger, Immediate) or EpisodeExtracted not in (
        spec.trigger.on
    ):
        return False
    return not callable(spec.applies_to) or spec.applies_to(event, make_deps())


def test_an_episode_reaches_the_strategies_of_its_space_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One event for both spaces: a user's episode runs the personal
    strategies, a group's the group ones -- group content never feeds a
    personal one (docs/group-memory.md, section 7.1)."""
    base = Settings()
    settings = base.model_copy(
        update={
            "scenes": base.scenes.model_copy(update={"users": True, "groups": True})
        }
    )
    for owner in ("user", "group"):
        module = importlib.import_module(
            f"everos.memory.strategies.update_{owner}_scenes"
        )
        monkeypatch.setattr(module, "load_settings", lambda: settings)

    def runs(owner_type: str) -> set[str]:
        return {s.name for s in STRATEGY_SPECS if _admits(s, _episode(owner_type))}

    assert runs("user") == {"extract_user_profile", "update_user_scenes"}
    assert runs("group") == {
        "update_member_profiles",
        "update_group_scenes",
        "update_group_profile",
    }


def test_only_a_group_episode_starts_the_group_profile_s_idle_clock() -> None:
    idle = next(s for s in STRATEGY_SPECS if s.name == "update_group_profile_idle")
    assert isinstance(idle.trigger, Idle)
    field = idle.trigger.event_field
    assert getattr(_episode("user"), field) is None
    assert getattr(_episode("group"), field) == "a/p/trip"
