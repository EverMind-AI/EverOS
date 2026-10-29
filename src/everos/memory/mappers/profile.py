"""everalgo ``Profile`` <-> domain :class:`~everos.memory.models.Profile`."""

from __future__ import annotations

__all__ = ["from_algo", "to_algo"]

from everalgo.types import Profile as AlgoProfile

from everos.memory.models import Profile


def from_algo(algo: AlgoProfile, *, primary_language: str | None = None) -> Profile:
    """Domain profile with the algo profile's fields; item extras ride along.

    Top-level extras the algo profile carries are dropped; ``primary_language``
    is everos's own and comes from the caller.
    """
    data = algo.model_dump(include=set(Profile.model_fields))
    return Profile.model_validate({**data, "primary_language": primary_language})


def to_algo(profile: Profile) -> AlgoProfile:
    """Algo profile validated from the domain profile's fields.

    ``primary_language`` stays behind: it is everos bookkeeping, passed to
    the extractor as ``output_language`` instead.
    """
    return AlgoProfile.model_validate(profile.model_dump(exclude={"primary_language"}))
