"""everalgo ``Profile`` <-> domain :class:`~everos.memory.models.Profile`."""

from __future__ import annotations

__all__ = ["from_algo", "to_algo"]

from everalgo.types import Profile as AlgoProfile

from everos.memory.models import Profile


def from_algo(algo: AlgoProfile) -> Profile:
    """Domain profile carrying the algo profile's items as plain dicts."""
    data = algo.model_dump(include={"explicit_info", "implicit_traits"})
    return Profile(
        owner_id=algo.owner_id,
        summary=algo.summary,
        timestamp=algo.timestamp,
        explicit_info=data["explicit_info"],
        implicit_traits=data["implicit_traits"],
    )


def to_algo(profile: Profile) -> AlgoProfile:
    """Algo profile validated from the domain profile's fields."""
    return AlgoProfile.model_validate(profile.model_dump())
