"""Round trip between everalgo ``Profile`` and the domain profile."""

from __future__ import annotations

from everalgo.types import Profile as AlgoProfile

from everos.memory.mappers import profile
from everos.memory.models import Profile


def test_algo_profile_round_trips_through_the_domain() -> None:
    algo = AlgoProfile(
        owner_id="u1",
        summary="likes tea",
        timestamp=42,
        explicit_info=[{"description": "drinks tea", "category": "food"}],
        implicit_traits=[{"description": "calm", "trait": "temperament"}],
    )

    assert profile.to_algo(profile.from_algo(algo)) == algo


def test_domain_profile_items_are_plain_dicts() -> None:
    algo = AlgoProfile(
        owner_id="u1",
        summary="s",
        timestamp=1,
        explicit_info=[{"description": "d", "extra_key": 3}],
    )

    domain = profile.from_algo(algo)

    assert isinstance(domain, Profile)
    assert domain.explicit_info[0]["extra_key"] == 3
