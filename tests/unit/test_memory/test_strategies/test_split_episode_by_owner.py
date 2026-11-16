"""``split_episode_by_owner``: one ``EpisodeExtractedForOwner`` per written copy.

The split makes no model call and touches no store; it only fans a pipeline
episode out so each owner's profile is a run of its own.
"""

from __future__ import annotations

import pytest

from everos.memory.events import (
    EpisodeCopy,
    EpisodeExtracted,
    EpisodeExtractedForOwner,
)
from everos.memory.strategies import (
    SPLIT_EPISODE_BY_OWNER_SPEC,
    split_episode_by_owner,
)
from tests.fakes import make_deps, make_stores
from tests.fakes.stores import InMemoryEpisodeStore


def _event(source: str = "pipeline") -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id="mc_a",
        episode_text="Alice and Bob planned a Kyoto trip.",
        episode_timestamp_ms=1_700_000_000_000,
        copies=[
            EpisodeCopy(owner_id="u_alice", entry_id="ep_1", owner_name="Alice"),
            EpisodeCopy(owner_id="u_bob", entry_id="ep_7"),
        ],
        session_id="s1",
        app_id="app",
        project_id="proj",
        source=source,
        language="English",
    )


def test_spec() -> None:
    spec = SPLIT_EPISODE_BY_OWNER_SPEC
    assert spec.name == "split_episode_by_owner"
    assert spec.trigger.on == [EpisodeExtracted]
    assert spec.emits == frozenset({EpisodeExtractedForOwner})
    assert spec.max_retries == 2


@pytest.mark.parametrize(
    ("source", "applies"), [("pipeline", True), ("reflection", False)]
)
def test_only_pipeline_episodes_are_split(source: str, applies: bool) -> None:
    applies_to = SPLIT_EPISODE_BY_OWNER_SPEC.applies_to
    assert callable(applies_to)
    assert applies_to(_event(source), make_deps()) is applies


async def test_one_event_per_copy_in_order() -> None:
    episodes = InMemoryEpisodeStore()
    deps = make_deps(stores=make_stores(episode=episodes))  # every LLM refuses

    events = await split_episode_by_owner(_event(), deps)

    assert events == [
        EpisodeExtractedForOwner(
            event_id=events[0].event_id,
            ts=events[0].ts,
            memcell_id="mc_a",
            episode_entry_id="ep_1",
            owner_id="u_alice",
            owner_name="Alice",
            app_id="app",
            project_id="proj",
            language="English",
        ),
        EpisodeExtractedForOwner(
            event_id=events[1].event_id,
            ts=events[1].ts,
            memcell_id="mc_a",
            episode_entry_id="ep_7",
            owner_id="u_bob",
            owner_name=None,
            app_id="app",
            project_id="proj",
            language="English",
        ),
    ]
    assert episodes.records == []
