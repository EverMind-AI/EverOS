"""everalgo ``Profile`` <-> domain :class:`~everos.memory.models.Profile`.

everalgo keeps two lists of its own on the profile it returns, as top-level
extras: ``do_not_record`` and ``pending``. They round-trip verbatim through
the domain profile so the next extraction reads them back; an empty one
travels as absent, which everalgo reads the same way.
"""

from __future__ import annotations

__all__ = ["from_algo", "to_algo"]

from everalgo.types import Profile as AlgoProfile

from everos.memory.models import Profile

_ALGO_LISTS = ("do_not_record", "pending")


def from_algo(algo: AlgoProfile, *, primary_language: str | None = None) -> Profile:
    """Domain profile with the algo profile's fields and its two lists.

    Other top-level extras the algo profile carries are dropped;
    ``primary_language`` is everos's own and comes from the caller.
    """
    extra = algo.model_extra or {}
    return Profile.model_validate(
        {
            **algo.model_dump(include=set(AlgoProfile.model_fields)),
            **{name: extra.get(name) or [] for name in _ALGO_LISTS},
            "primary_language": primary_language,
        }
    )


def to_algo(profile: Profile) -> AlgoProfile:
    """Algo profile validated from the domain profile's fields.

    ``primary_language`` stays behind: it is everos bookkeeping, passed to
    the extractor as ``output_language`` instead. A non-empty list goes back
    as the extra everalgo wrote it as.
    """
    data = profile.model_dump(exclude={"primary_language"})
    lists = {name: data.pop(name) for name in _ALGO_LISTS}
    return AlgoProfile.model_validate(
        {**data, **{name: value for name, value in lists.items() if value}}
    )
