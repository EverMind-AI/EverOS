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


def test_domain_profile_items_keep_host_extras() -> None:
    algo = AlgoProfile(
        owner_id="u1",
        summary="s",
        timestamp=1,
        explicit_info=[{"description": "d", "extra_key": 3}],
    )

    domain = profile.from_algo(algo)

    assert isinstance(domain, Profile)
    assert domain.explicit_info[0].model_extra == {"extra_key": 3}
    assert profile.to_algo(domain).explicit_info[0].model_extra == {"extra_key": 3}


_DO_NOT_RECORD = [
    {"topic": "travel budget", "evidence": "asked not to record it", "observed_at": 5}
]
_PENDING = [
    {
        "description": "Swims in the morning.",
        "category": "habit",
        "evidence": "went swimming before work",
        "bucket": "explicit_info",
        "observed_at": 5,
    }
]


def test_everalgos_lists_round_trip_verbatim() -> None:
    algo = AlgoProfile.model_validate(
        {
            "owner_id": "u1",
            "summary": "s",
            "timestamp": 5,
            "do_not_record": _DO_NOT_RECORD,
            "pending": _PENDING,
        }
    )

    domain = profile.from_algo(algo)

    assert (domain.do_not_record, domain.pending) == (_DO_NOT_RECORD, _PENDING)
    assert profile.to_algo(domain) == algo


def test_an_empty_list_travels_as_absent() -> None:
    """everalgo returns ``pending: []`` when it emptied the list; it is stored
    as nothing, and nothing goes back."""
    algo = AlgoProfile.model_validate(
        {"owner_id": "u1", "summary": "s", "timestamp": 5, "pending": []}
    )

    domain = profile.from_algo(algo)

    assert domain.pending == []
    assert profile.to_algo(domain).model_extra == {}


def test_other_top_level_extras_are_dropped() -> None:
    algo = AlgoProfile.model_validate(
        {"owner_id": "u1", "summary": "s", "timestamp": 5, "note": "x"}
    )

    assert "note" not in profile.from_algo(algo).model_dump()
