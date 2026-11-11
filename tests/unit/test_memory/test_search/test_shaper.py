"""Unit tests for ``memory.search.shaper``: ranked ids + stored records -> DTOs."""

from __future__ import annotations

from everalgo.types import Candidate

from everos.core.utils.datetime import from_timestamp_ms, to_display_tz
from everos.memory.search.shaper import shape_episodes

from ._records import stored_episode


def _ranked(cid: str, score: float) -> Candidate:
    """A ranked candidate: the shaper reads only its id and score."""
    return Candidate(id=cid, score=score, source="other", metadata={})


def test_shape_episodes_builds_the_dto_from_the_record() -> None:
    ep = stored_episode("ep_1")

    (item,) = shape_episodes([_ranked(ep.ref.id, 0.7)], {ep.ref.id: ep})

    assert item.id == ep.ref.id and item.score == 0.7
    assert item.user_id == "alice" and item.app_id == "default"
    assert item.episode == "body ep_1" and item.subject == "subject ep_1"
    assert item.summary == "summary ep_1" and item.session_id == "sess_a"
    assert item.timestamp == to_display_tz(from_timestamp_ms(ep.item.timestamp))


def test_the_dto_id_is_the_global_id_whatever_the_candidate_key() -> None:
    """A candidate keyed otherwise still answers with the record's global id."""
    ep = stored_episode("ep_1")
    (item,) = shape_episodes([_ranked("ep_1", 0.5)], {"ep_1": ep})
    assert item.id == ep.ref.id


def test_shape_episodes_drops_unknown_ids_and_wrong_owner_type() -> None:
    agent_owned = stored_episode("ep_2", owner_type="agent")
    items = shape_episodes(
        [_ranked("missing", 0.9), _ranked("ep_2", 0.8)], {"ep_2": agent_owned}
    )
    assert items == []
