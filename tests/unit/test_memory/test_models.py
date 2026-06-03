"""Unit tests for memory domain models — focused on ``from_algo`` factories.

The factories carry the load-bearing contract: algo's emitted business
fields survive, everos's engineering metadata (session_id / sender_ids
/ parent_id) gets injected, and any algo-side ``parent_id`` (smuggled
through ``extra='allow'``) is dropped in favour of the caller's value.
"""

from __future__ import annotations

import uuid

from everalgo.types import (
    AgentCase as AlgoAgentCase,
)
from everalgo.types import (
    AtomicFact as AlgoAtomicFact,
)
from everalgo.types import (
    Episode as AlgoEpisode,
)
from everalgo.types import (
    Foresight as AlgoForesight,
)

from everos.memory.models import AgentCase, AtomicFact, Episode, Foresight


def test_atomic_fact_from_algo_carries_business_fields_and_metadata() -> None:
    algo = AlgoAtomicFact(
        owner_id="u_alice",
        content="alice likes hiking",
        timestamp=1_700_000_000_000,
    )
    fact = AtomicFact.from_algo(
        algo,
        owner_id="u_alice",
        session_id="s_42",
        parent_id="mc_abc",
    )
    assert fact.owner_id == "u_alice"
    assert fact.fact == "alice likes hiking"
    assert fact.timestamp == 1_700_000_000_000
    assert fact.session_id == "s_42"
    assert fact.parent_id == "mc_abc"
    assert not hasattr(fact, "sender_ids")


def test_atomic_fact_from_algo_drops_algo_side_parent_id() -> None:
    # Smuggle a parent_id through extra='allow' on the algo side.
    algo = AlgoAtomicFact.model_validate(
        {
            "owner_id": "u_alice",
            "content": "x",
            "timestamp": 1_700_000_000_000,
            "parent_id": "ALGO_STALE",
        }
    )
    fact = AtomicFact.from_algo(
        algo, owner_id="u_alice", session_id="s1", parent_id="mc_real"
    )
    # Caller-supplied parent_id wins; algo-side value is discarded.
    assert fact.parent_id == "mc_real"


def test_atomic_fact_from_algo_owner_id_override_for_fan_out() -> None:
    """One LLM template fans out to many owners — caller's owner_id wins."""
    algo = AlgoAtomicFact(
        owner_id="PLACEHOLDER",  # subject-agnostic prompt placeholder
        content="likes hiking",
        timestamp=1_700_000_000_000,
    )
    fact_alice = AtomicFact.from_algo(
        algo, owner_id="u_alice", session_id="s1", parent_id="mc_a"
    )
    fact_bob = AtomicFact.from_algo(
        algo, owner_id="u_bob", session_id="s1", parent_id="mc_a"
    )
    assert fact_alice.owner_id == "u_alice"
    assert fact_bob.owner_id == "u_bob"
    # Same source template body survives the fan-out.
    assert fact_alice.fact == fact_bob.fact == "likes hiking"


def test_foresight_from_algo_preserves_optional_time_window() -> None:
    algo = AlgoForesight(
        owner_id="u_alice",
        foresight="plans trip to tokyo",
        evidence="said so",
        timestamp=1_700_000_000_000,
        start_time="2026-06-01",
        duration_days=7,
    )
    fs = Foresight.from_algo(algo, session_id="s1", parent_id="mc_a")
    assert fs.foresight == "plans trip to tokyo"
    assert fs.evidence == "said so"
    assert fs.start_time == "2026-06-01"
    assert fs.duration_days == 7
    assert fs.end_time is None
    assert fs.parent_id == "mc_a"
    assert not hasattr(fs, "sender_ids")


def test_foresight_from_algo_drops_algo_side_parent_id() -> None:
    algo = AlgoForesight.model_validate(
        {
            "owner_id": "u_alice",
            "foresight": "x",
            "evidence": "y",
            "timestamp": 1_700_000_000_000,
            "parent_id": "ALGO_STALE",
        }
    )
    fs = Foresight.from_algo(algo, session_id="s1", parent_id="mc_real")
    assert fs.parent_id == "mc_real"


def test_foresight_from_algo_preserves_algo_owner_id() -> None:
    """Per-sender extraction: algo emits the correct owner_id."""
    algo = AlgoForesight(
        owner_id="u_bob",
        foresight="trip to tokyo",
        evidence="said so",
        timestamp=1_700_000_000_000,
    )
    fs = Foresight.from_algo(algo, session_id="s1", parent_id="mc_a")
    assert fs.owner_id == "u_bob"


def test_agent_case_from_algo_injects_owner_and_drops_algo_id() -> None:
    """Algo emits a uuid `id` + no owner; everos injects agent_id, drops uuid."""
    algo = AlgoAgentCase(
        id=uuid.uuid4().hex,
        timestamp=1_700_000_000_000,
        task_intent="summarise doc",
        approach="read + condense",
        quality_score=0.75,
        key_insight="batch-then-summarise",
    )
    case = AgentCase.from_algo(
        algo, owner_id="agent_42", session_id="s1", parent_id="mc_a"
    )
    assert case.owner_id == "agent_42"
    assert case.task_intent == "summarise doc"
    assert case.approach == "read + condense"
    assert case.quality_score == 0.75
    assert case.key_insight == "batch-then-summarise"
    assert case.session_id == "s1"
    assert case.parent_id == "mc_a"
    # algo's uuid `id` is not surfaced on the domain model.
    assert not hasattr(case, "id") or case.id != algo.id  # type: ignore[attr-defined]


def test_agent_case_from_algo_normalises_empty_key_insight_to_none() -> None:
    """algo emits `""` when there's nothing to insight; domain normalises to None."""
    algo = AlgoAgentCase(
        id=uuid.uuid4().hex,
        timestamp=1_700_000_000_000,
        task_intent="ti",
        approach="ap",
        quality_score=0.5,
        key_insight="",
    )
    case = AgentCase.from_algo(
        algo, owner_id="agent_42", session_id="s1", parent_id="mc_a"
    )
    assert case.key_insight is None


def test_episode_from_algo_owner_id_caller_supplied() -> None:
    """Caller supplies ``owner_id``; algo's value (None or otherwise) is dropped.

    The pipeline runs the algo once with ``sender_id=None`` (generic
    EPISODE_GENERATION_PROMPT) and then fans the same algo Episode out
    to one domain Episode per user sender, each rooted at its own owner.
    """
    algo = AlgoEpisode(owner_id=None, episode="hello", timestamp=1_700_000_000_000)
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
