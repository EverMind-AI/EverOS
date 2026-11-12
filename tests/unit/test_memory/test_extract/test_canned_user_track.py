"""The canned user-track scripts satisfy the real operators they stand in for.

Each script under ``tests/fakes/llm_scripts/user_track`` is fed to the real
everalgo operator through ``user_track_llm``; a script that drifted from the
operator's strict reply model fails here instead of deep inside an
integration test. Every operator also has its empty (or blank) script,
which exercises the path where it yields nothing. The profile scripts run
with the extractor's item checks on, against the canned episode narrative
their quotes are copied from.
"""

from __future__ import annotations

import json

import pytest
from everalgo.boundary import BoundaryDetector, Cuts
from everalgo.llm import LLMOutputError
from everalgo.types import ChatMessage, MemCell
from everalgo.types import Episode as AlgoEpisode
from everalgo.types import Profile as AlgoProfile
from everalgo.user_memory import (
    AtomicFactExtractor,
    EpisodeExtractor,
    EpisodeReflector,
    ProfileExtractor,
)

from tests.fakes import user_track_llm, user_track_script

_TS = 1_775_116_800_000

_MESSAGES = [
    ChatMessage(
        id="m1",
        role="user",
        sender_id="u_alice",
        content="Kyoto in April?",
        timestamp=_TS,
    ),
    ChatMessage(
        id="m2",
        role="assistant",
        sender_id="bot",
        content="A ryokan?",
        timestamp=_TS + 1,
    ),
    ChatMessage(
        id="m3",
        role="user",
        sender_id="u_alice",
        content="Yes, quiet.",
        timestamp=_TS + 2,
    ),
]
_CELL = MemCell(items=list(_MESSAGES), timestamp=_TS + 2)
_EPISODE = AlgoEpisode(
    owner_id="u_alice",
    episode=json.loads(user_track_script("episode"))["content"],
    summary="Alice plans a quiet Kyoto stay.",
    timestamp=_TS + 2,
)


@pytest.mark.parametrize(
    ("script", "is_final", "cuts"),
    [
        ("boundary", True, Cuts(ends=[3], tail_start=3)),
        ("boundary", False, Cuts(ends=[], tail_start=0)),
        ("boundary_empty", False, Cuts(ends=[], tail_start=0)),
        ("boundary_split", False, Cuts(ends=[2], tail_start=2)),
    ],
)
async def test_boundary_scripts(script: str, is_final: bool, cuts: Cuts) -> None:
    detector = BoundaryDetector(llm=user_track_llm(boundary=script))
    assert await detector.adetect_batch(_MESSAGES, is_final=is_final) == cuts


async def test_episode_script() -> None:
    episode = await EpisodeExtractor(llm=user_track_llm()).aextract(
        _CELL, sender_id=None
    )
    want = json.loads(user_track_script("episode"))
    assert (episode.episode, episode.subject) == (want["content"], want["title"])


async def test_blank_episode_script_is_unusable() -> None:
    with pytest.raises(LLMOutputError):
        await EpisodeExtractor(llm=user_track_llm(episode="episode_blank")).aextract(
            _CELL, sender_id=None
        )


@pytest.mark.parametrize(
    ("script", "count"), [("atomic_facts", 3), ("atomic_facts_empty", 0)]
)
async def test_atomic_fact_scripts(script: str, count: int) -> None:
    facts = await AtomicFactExtractor(
        llm=user_track_llm(atomic_facts=script)
    ).aextract_from_text("Alice plans a Kyoto trip.", timestamp=_TS)
    assert len(facts) == count


@pytest.mark.parametrize(
    ("script", "counts"),
    [
        # The trait is a pattern seen once: it waits on ``pending``, and no new
        # trait is written by default.
        (
            "profile_init",
            {
                "explicit_info": 2,
                "implicit_traits": 0,
                "pending": 1,
                "do_not_record": 1,
            },
        ),
        (
            "profile_init_empty",
            {
                "explicit_info": 0,
                "implicit_traits": 0,
                "pending": 0,
                "do_not_record": 0,
            },
        ),
    ],
)
async def test_profile_init_scripts(script: str, counts: dict[str, int]) -> None:
    profile = await ProfileExtractor(
        llm=user_track_llm(profile_init=script)
    ).aextract_from_episodes([_EPISODE], owner_id="u_alice", owner_name="Alice")
    extra = profile.model_extra or {}
    assert {
        "explicit_info": len(profile.explicit_info),
        "implicit_traits": len(profile.implicit_traits),
        "pending": len(extra.get("pending", [])),
        "do_not_record": len(extra.get("do_not_record", [])),
    } == counts


@pytest.mark.parametrize(
    ("script", "explicit"), [("profile_update", 2), ("profile_update_empty", 1)]
)
async def test_profile_update_scripts(script: str, explicit: int) -> None:
    old = AlgoProfile(
        owner_id="u_alice",
        summary="Alice travels.",
        timestamp=_TS - 1,
        explicit_info=[{"description": "Plans a trip to Kyoto.", "category": "travel"}],
    )
    profile = await ProfileExtractor(
        llm=user_track_llm(profile_update=script)
    ).aextract_from_episodes(
        [_EPISODE], owner_id="u_alice", owner_name="Alice", old_profile=old
    )
    assert len(profile.explicit_info) == explicit


async def test_reflect_script() -> None:
    episodes = [
        AlgoEpisode(
            owner_id="u_alice", episode=f"e{i}", summary=f"s{i}", timestamp=_TS + i
        )
        for i in range(2)
    ]
    merged = await EpisodeReflector(llm=user_track_llm()).areflect(episodes)
    assert merged.summary == json.loads(user_track_script("reflect"))["summary"]
