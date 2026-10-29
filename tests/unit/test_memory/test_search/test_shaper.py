"""Unit tests for ``memory.search.shaper``: ranked ids + stored records -> DTOs."""

from __future__ import annotations

from everalgo.types import Candidate, RankOutput, ScoredItem

from everos.core.utils.datetime import from_timestamp_ms, to_display_tz
from everos.memory.search.shaper import (
    reshape_hybrid_output,
    shape_agent_cases,
    shape_agent_cases_ranked,
    shape_agent_skills,
    shape_episodes,
)

from ._records import stored_case, stored_episode, stored_fact, stored_skill


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
    assert item.atomic_facts == []


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


def test_reshape_hybrid_nests_facts_under_kept_episode() -> None:
    ep = stored_episode("ep_1")
    fact = stored_fact("af_1", parent_id="ep_1")
    scored = [
        ScoredItem(id=ep.ref.id, score=0.9, item_type="episode"),
        ScoredItem(
            id=fact.ref.id,
            score=0.6,
            item_type="atomic_fact",
            parent_episode_id=ep.ref.id,
        ),
    ]

    (item,) = reshape_hybrid_output(
        scored, episodes_by_id={ep.ref.id: ep}, facts_by_id={fact.ref.id: fact}
    )

    assert item.id == ep.ref.id and item.score == 0.9
    assert [(f.id, f.content, f.score) for f in item.atomic_facts] == [
        (fact.ref.id, "fact af_1", 0.6)
    ]


def test_reshape_hybrid_backfills_evicted_episode_from_pool() -> None:
    """A fact that evicted its parent brings the parent back at its score."""
    ep = stored_episode("ep_1")
    facts = [
        stored_fact("af_1", parent_id="ep_1"),
        stored_fact("af_2", parent_id="ep_1"),
    ]
    scored = [
        ScoredItem(
            id=f.ref.id, score=s, item_type="atomic_fact", parent_episode_id=ep.ref.id
        )
        for f, s in zip(facts, (0.4, 0.8), strict=True)
    ]

    (item,) = reshape_hybrid_output(
        scored,
        episodes_by_id={ep.ref.id: ep},
        facts_by_id={f.ref.id: f for f in facts},
    )

    assert item.score == 0.8
    assert [f.score for f in item.atomic_facts] == [0.8, 0.4]


def test_reshape_hybrid_drops_orphan_facts_with_no_pool_parent() -> None:
    fact = stored_fact("af_1", parent_id="ep_gone")
    scored = [
        ScoredItem(
            id=fact.ref.id, score=0.5, item_type="atomic_fact", parent_episode_id="gone"
        )
    ]
    assert reshape_hybrid_output(scored, episodes_by_id={}, facts_by_id={}) == []


def test_reshape_hybrid_sorts_by_score() -> None:
    a, b = stored_episode("ep_a"), stored_episode("ep_b")
    scored = [
        ScoredItem(id=a.ref.id, score=0.2, item_type="episode"),
        ScoredItem(id=b.ref.id, score=0.7, item_type="episode"),
    ]
    items = reshape_hybrid_output(
        scored, episodes_by_id={a.ref.id: a, b.ref.id: b}, facts_by_id={}
    )
    assert [i.id for i in items] == [b.ref.id, a.ref.id]


def test_shape_agent_cases_basic_and_owner_type_check() -> None:
    case = stored_case("ac_1")
    user_owned = stored_episode("ep_1")

    items = shape_agent_cases(
        [_ranked(case.ref.id, 0.6), _ranked(user_owned.ref.id, 0.5)],
        {case.ref.id: case, user_owned.ref.id: user_owned},  # type: ignore[dict-item]
    )

    (item,) = items
    assert item.id == case.ref.id and item.agent_id == "agent_1"
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
