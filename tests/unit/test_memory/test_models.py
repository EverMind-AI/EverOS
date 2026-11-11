"""Unit tests for memory domain models — focused on ``from_algo`` factories.

The factories carry the load-bearing contract: algo's emitted business
fields survive, everos's engineering metadata (session_id / sender_ids
/ parent_id) gets injected, and any algo-side ``parent_id`` (smuggled
through ``extra='allow'``) is dropped in favour of the caller's value.
"""

from __future__ import annotations

import pytest
import structlog
from everalgo.types import Episode as AlgoEpisode
from pydantic import ValidationError

from everos.memory.models import (
    Episode,
    Profile,
    mint_cluster_id,
)


def test_episode_from_algo_drops_algo_side_parent_id() -> None:
    # Smuggle a parent_id through extra='allow' on the algo side.
    algo = AlgoEpisode.model_validate(
        {
            "owner_id": None,
            "summary": "s",
            "episode": "x",
            "timestamp": 1_700_000_000_000,
            "parent_id": "ALGO_STALE",
        }
    )
    episode = Episode.from_algo(
        algo,
        owner_id="u_alice",
        session_id="s1",
        sender_ids=["u_alice"],
        parent_id="mc_real",
    )
    # Caller-supplied parent_id wins; algo-side value is discarded.
    assert episode.parent_id == "mc_real"


def test_episode_from_algo_owner_id_caller_supplied() -> None:
    """Caller supplies ``owner_id``; algo's value (None or otherwise) is dropped.

    The pipeline runs the algo once with ``sender_id=None`` (generic
    EPISODE_GENERATION_PROMPT) and then fans the same algo Episode out
    to one domain Episode per user sender, each rooted at its own owner.
    """
    algo = AlgoEpisode(
        owner_id=None, summary="summary", episode="hello", timestamp=1_700_000_000_000
    )
    ep_alice = Episode.from_algo(
        algo,
        owner_id="u_alice",
        session_id="s1",
        sender_ids=["u_alice", "u_bob"],
        parent_id="mc_a",
    )
    ep_bob = Episode.from_algo(
        algo,
        owner_id="u_bob",
        session_id="s1",
        sender_ids=["u_alice", "u_bob"],
        parent_id="mc_a",
    )
    assert ep_alice.owner_id == "u_alice"
    assert ep_bob.owner_id == "u_bob"
    assert ep_alice.episode == ep_bob.episode == "hello"
    assert ep_alice.parent_id == ep_bob.parent_id == "mc_a"
    assert ep_alice.session_id == ep_bob.session_id == "s1"


def test_mint_cluster_id_shape() -> None:
    cid = mint_cluster_id()
    assert cid.startswith("cl_")
    assert len(cid) == 3 + 12  # ``cl_`` + 12 hex chars


def test_profile_from_stored_drops_items_that_are_no_longer_items() -> None:
    long_basis = "x" * 200
    with structlog.testing.capture_logs() as captured:
        profile = Profile.from_stored(
            owner_id="u1",
            summary="s",
            timestamp=1,
            explicit_info=["drinks tea", {"category": "food"}],
            implicit_traits=[{"description": "calm"}, {"basis": long_basis}],
        )

    assert [i.description for i in profile.explicit_info] == ["drinks tea"]
    assert [i.description for i in profile.implicit_traits] == ["calm"]
    dropped = [e for e in captured if e["event"] == "profile_entry_dropped"]
    assert [(e["owner_id"], e["bucket"]) for e in dropped] == [
        ("u1", "ExplicitInfoItem"),
        ("u1", "ImplicitTraitItem"),
    ]
    assert dropped[0]["entry"] == '{"category": "food"}'
    assert len(dropped[1]["entry"]) == 80


def test_profile_construction_stays_strict() -> None:
    with pytest.raises(ValidationError):
        Profile(owner_id="u1", summary="s", timestamp=1, explicit_info=[{"x": 1}])
