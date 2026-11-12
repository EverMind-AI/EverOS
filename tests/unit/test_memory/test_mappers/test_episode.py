"""``mappers.episode.to_algo`` carries the narrative fields and drops everos context."""

from __future__ import annotations

from everos.memory.mappers import episode as episode_mapper
from everos.memory.models import Episode


def test_to_algo_keeps_the_narrative_fields_only() -> None:
    domain = Episode(
        owner_id="u_alice",
        episode="Alice plans a trip to Kyoto.",
        summary="Kyoto trip",
        subject="Trip",
        timestamp=1_700_000_000_000,
        session_id="s_1",
        sender_ids=["u_alice", "u_bob"],
        parent_id="mc_1",
        language="English",
    )

    algo = episode_mapper.to_algo(domain)

    assert algo.model_dump() == {
        "owner_id": "u_alice",
        "episode": "Alice plans a trip to Kyoto.",
        "subject": "Trip",
        "summary": "Kyoto trip",
        "timestamp": 1_700_000_000_000,
    }
