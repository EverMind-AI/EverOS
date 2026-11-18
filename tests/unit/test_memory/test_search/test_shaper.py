"""Unit tests for ``memory.search.shaper``: ranked ids + stored records -> DTOs."""

from __future__ import annotations

from everalgo.types import Candidate, RankOutput, ScoredItem

from everos.core.utils.datetime import from_timestamp_ms, to_display_tz
from everos.memory.search.shaper import (
    shape_agent_cases,
    shape_agent_cases_ranked,
    shape_agent_skills,
    shape_episodes,
    shape_episodes_ranked,
)

from ._records import stored_case, stored_episode, stored_skill


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
    """Agentic candidates are keyed by entry id; the response is not."""
    ep = stored_episode("ep_1")
    (item,) = shape_episodes([_ranked("ep_1", 0.5)], {"ep_1": ep})
    assert item.id == ep.ref.id


def test_shape_episodes_drops_unknown_ids_and_wrong_owner_type() -> None:
    agent_owned = stored_episode("ep_2", owner_type="agent")
    items = shape_episodes(
        [_ranked("missing", 0.9), _ranked("ep_2", 0.8)], {"ep_2": agent_owned}
    )
    assert items == []


def test_shape_episodes_ranked_reads_the_rank_output_items() -> None:
    """An everalgo ``RankOutput`` shapes exactly like its items do."""
    ep = stored_episode("ep_1")
    output = RankOutput(
        items=[ScoredItem(id=ep.ref.id, score=0.9, item_type="episode")]
    )

    (item,) = shape_episodes_ranked(output, {ep.ref.id: ep})

    assert item.id == ep.ref.id and item.score == 0.9


def test_shape_agent_cases_basic_and_owner_type_check() -> None:
    case = stored_case("ac_1")
    user_owned = stored_episode("ep_1")

    items = shape_agent_cases(
        [_ranked(case.ref.id, 0.6), _ranked(user_owned.ref.id, 0.5)],
        {case.ref.id: case, user_owned.ref.id: user_owned},  # type: ignore[dict-item]
    )

    (item,) = items
    assert item.id == case.ref.id and item.agent_id == "agent_1"
    assert item.entry_id == "ac_1"
    assert item.task_intent == "fix the build" and item.quality_score == 0.8
    assert item.session_id == "sess_a" and item.score == 0.6


def test_shape_agent_cases_ranked_reads_rank_output() -> None:
    case = stored_case("ac_1")
    output = RankOutput(items=[ScoredItem(id=case.ref.id, score=0.3, item_type="case")])
    (item,) = shape_agent_cases_ranked(output, {case.ref.id: case})
    assert item.score == 0.3


def test_shape_agent_skills_basic() -> None:
    skill = stored_skill("sk_1", source_case_ids=["ac_1"])

    (item,) = shape_agent_skills([_ranked("sk_1", 0.9)], {"sk_1": skill})

    assert item.id == "sk_1" and item.agent_id == "agent_1"
    assert item.name == "debugging" and item.content == "steps"
    assert item.source_case_ids == ["ac_1"] and item.score == 0.9
