"""OME business strategies — event-triggered offline tasks.

Each strategy is an async function ``fn(event, deps) -> list[Event]``
declared next to its :class:`~everos.memory.ports.StrategySpec`.
:data:`STRATEGY_SPECS` is the complete set the service registers; strategy
names are stable because ``ome.toml`` overrides and run records key on them.

External usage:
    from everos.memory.strategies import STRATEGY_SPECS, extract_user_profile
"""

from __future__ import annotations

from everos.memory.ports import StrategySpec

from .catch_up_profiles import GROUP_PROFILES as GROUP_PROFILES
from .catch_up_profiles import CatchUpProvider as CatchUpProvider
from .catch_up_profiles import catch_up_spec as catch_up_spec
from .extract_user_profile import EXTRACT_USER_PROFILE_SPEC as EXTRACT_USER_PROFILE_SPEC
from .extract_user_profile import extract_user_profile as extract_user_profile
from .update_group_profile import (
    UPDATE_GROUP_PROFILE_IDLE_SPEC as UPDATE_GROUP_PROFILE_IDLE_SPEC,
)
from .update_group_profile import (
    UPDATE_GROUP_PROFILE_SPEC as UPDATE_GROUP_PROFILE_SPEC,
)
from .update_group_profile import update_group_profile as update_group_profile
from .update_group_scenes import (
    UPDATE_GROUP_SCENES_SPEC as UPDATE_GROUP_SCENES_SPEC,
)
from .update_group_scenes import update_group_scenes as update_group_scenes
from .update_member_profiles import (
    UPDATE_MEMBER_PROFILES_IDLE_SPEC as UPDATE_MEMBER_PROFILES_IDLE_SPEC,
)
from .update_member_profiles import (
    UPDATE_MEMBER_PROFILES_SPEC as UPDATE_MEMBER_PROFILES_SPEC,
)
from .update_member_profiles import update_member_profiles as update_member_profiles
from .update_user_scenes import UPDATE_USER_SCENES_SPEC as UPDATE_USER_SCENES_SPEC
from .update_user_scenes import update_user_scenes as update_user_scenes

CATCH_UP_PROVIDERS: tuple[CatchUpProvider, ...] = (GROUP_PROFILES,)
"""Every kind of profile ``catch_up_profiles`` brings up to date. A strategy
that keeps profiles on a cadence registers its provider here -- the groups'
today, the one-to-one profile's once it has a cadence of its own."""

CATCH_UP_PROFILES_SPEC: StrategySpec = catch_up_spec(CATCH_UP_PROVIDERS)
"""``POST /api/v2/ome/trigger {"name": "catch_up_profiles"}``."""

STRATEGY_SPECS: tuple[StrategySpec, ...] = (
    EXTRACT_USER_PROFILE_SPEC,
    UPDATE_USER_SCENES_SPEC,
    UPDATE_MEMBER_PROFILES_SPEC,
    UPDATE_MEMBER_PROFILES_IDLE_SPEC,
    UPDATE_GROUP_SCENES_SPEC,
    UPDATE_GROUP_PROFILE_SPEC,
    UPDATE_GROUP_PROFILE_IDLE_SPEC,
    CATCH_UP_PROFILES_SPEC,
)
"""Every strategy the service registers, in registration order."""

__all__ = [
    "CATCH_UP_PROFILES_SPEC",
    "CATCH_UP_PROVIDERS",
    "EXTRACT_USER_PROFILE_SPEC",
    "GROUP_PROFILES",
    "STRATEGY_SPECS",
    "UPDATE_GROUP_PROFILE_IDLE_SPEC",
    "UPDATE_GROUP_PROFILE_SPEC",
    "UPDATE_GROUP_SCENES_SPEC",
    "UPDATE_MEMBER_PROFILES_IDLE_SPEC",
    "UPDATE_MEMBER_PROFILES_SPEC",
    "UPDATE_USER_SCENES_SPEC",
    "CatchUpProvider",
    "catch_up_spec",
    "extract_user_profile",
    "update_group_profile",
    "update_group_scenes",
    "update_member_profiles",
    "update_user_scenes",
]
