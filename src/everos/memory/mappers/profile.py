"""everalgo ``Profile`` <-> domain :class:`~everos.memory.models.Profile`."""

from __future__ import annotations

__all__ = ["from_algo", "to_algo"]

from everalgo.types import Profile as AlgoProfile

from everos.memory.models import Profile


def from_algo(algo: AlgoProfile) -> Profile:
    """Domain profile with the algo profile's fields; item extras ride along.

    Top-level extras the algo profile carries are dropped.
    """
    return Profile.model_validate(algo.model_dump(include=set(Profile.model_fields)))


def to_algo(profile: Profile) -> AlgoProfile:
    """Algo profile validated from the domain profile's fields."""
    return AlgoProfile.model_validate(profile.model_dump())
