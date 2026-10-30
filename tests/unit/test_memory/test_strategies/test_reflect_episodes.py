"""Tests for the two Reflection strategies.

``reflect_episodes`` (Cron) merges and returns the merged episodes;
``finish_reflection`` deprecates once their facts exist. Both are thin
entry points — the Reflection logic itself is tested in
``test_reflection/test_orchestrator.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import structlog.testing

from everos.memory.events import (
    AtomicFactsExtracted,
    EpisodeExtracted,
    ReflectedEpisode,
    ReflectionContext,
)
from everos.memory.models import Cluster
from everos.memory.ports import Cron, CronTick, Scope
from everos.memory.strategies import (
    FINISH_REFLECTION_SPEC,
    REFLECT_EPISODES_SPEC,
    finish_reflection,
    reflect_episodes,
)
from tests.fakes import make_deps

_SCOPE = Scope(
    app_id="default", project_id="default", owner_id="u_alice", owner_type="user"
)
_TICK = CronTick(strategy_name="reflect_episodes")


def _context() -> ReflectionContext:
    return ReflectionContext(
        cluster_id="cl_1",
        mode="init",
        source_members=[("ep_1", "episode"), ("ep_2", "episode")],
        episodes=[
            ReflectedEpisode(entry_id="ep_1", md_path="a.md", parent_id="mc_1"),
            ReflectedEpisode(entry_id="ep_2", md_path="a.md", parent_id="mc_2"),
        ],
        merged_episode="merged narrative",
        last_ts=1_700_000_000_000,
    )


def _facts_event(**overrides: object) -> AtomicFactsExtracted:
    fields: dict[str, object] = {
        "episode_entry_id": "ep_merged",
        "owner_id": "u_alice",
        "source": "reflection",
        "reflection": _context(),
    }
    return AtomicFactsExtracted(**{**fields, **overrides})


# ── reflect_episodes ─────────────────────────────────────────────────────


def test_reflect_spec() -> None:
    spec = REFLECT_EPISODES_SPEC
    assert spec.name == "reflect_episodes"
    assert isinstance(spec.trigger, Cron)
    assert spec.trigger.expr == "0 2 * * 1"
    assert spec.emits == frozenset({EpisodeExtracted})
    assert spec.max_retries == 1
    assert spec.enabled is False


async def test_reflect_returns_without_side_effects_when_embedding_unavailable() -> (
    None
):
    """No embedder → early return; no owner enumeration, no orchestrator.

    Reflection re-embeds merged narratives, so it cannot run without an
    embedder; a Tier-1 cron tick simply no-ops.
    """
    deps = make_deps(embedding=None)
    with (
        patch(
            "everos.memory.strategies.reflect_episodes.ReflectionOrchestrator",
            side_effect=AssertionError("orchestrator must not be built"),
        ),
        structlog.testing.capture_logs() as captured,
    ):
        events = await reflect_episodes(_TICK, deps)

    assert events == []
    gated = [
        e
        for e in captured
        if e.get("event") == "strategy_gated_off_embedding_unavailable"
    ]
    assert len(gated) == 1
    assert gated[0]["strategy_name"] == "reflect_episodes"


async def test_reflect_returns_every_owner_merge() -> None:
    """One orchestrator run per owner scope; their events are concatenated."""
    deps = make_deps()
    bob = Scope(
        app_id="default", project_id="default", owner_id="u_bob", owner_type="user"
    )
    for scope, cid in ((_SCOPE, "cl_a"), (bob, "cl_b")):
        deps.stores.cluster.seed(
            scope, "user_memory", Cluster(id=cid, centroid=(0.0,), last_ts=1)
        )

    async def merged_for(*, scope: Scope) -> list[EpisodeExtracted]:
        return [
            EpisodeExtracted(
                memcell_id=f"m_{scope.owner_id}",
                episode_entry_id=f"m_{scope.owner_id}",
                episode_text="merged",
                episode_timestamp_ms=1,
                owner_id=scope.owner_id,
                source="reflection",
            )
        ]

    with patch(
        "everos.memory.strategies.reflect_episodes.ReflectionOrchestrator"
    ) as orchestrator_cls:
        orchestrator_cls.return_value.run = AsyncMock(side_effect=merged_for)
        events = await reflect_episodes(_TICK, deps)

    assert sorted(e.owner_id for e in events) == ["u_alice", "u_bob"]


# ── finish_reflection ────────────────────────────────────────────────────


def test_finish_spec() -> None:
    spec = FINISH_REFLECTION_SPEC
    assert spec.name == "finish_reflection"
    assert spec.trigger.on == [AtomicFactsExtracted]
    assert spec.emits == frozenset()
    assert spec.max_retries == 1
    assert spec.enabled is True


def test_finish_applies_only_to_reflection_facts() -> None:
    applies = FINISH_REFLECTION_SPEC.applies_to
    assert callable(applies)
    deps = make_deps()
    assert applies(_facts_event(), deps) is True
    assert applies(_facts_event(source="pipeline", reflection=None), deps) is False


async def test_finish_hands_the_context_to_the_finisher() -> None:
    deps = make_deps()
    with patch(
        "everos.memory.strategies.finish_reflection.ReflectionFinisher"
    ) as finisher_cls:
        finisher_cls.return_value.finish = AsyncMock(return_value=None)
        events = await finish_reflection(_facts_event(), deps)

    assert events == []
    finisher_cls.return_value.finish.assert_awaited_once_with(
        scope=_SCOPE, merged_entry_id="ep_merged", context=_context()
    )


async def test_finish_returns_without_side_effects_when_embedding_unavailable() -> None:
    with patch(
        "everos.memory.strategies.finish_reflection.ReflectionFinisher",
        side_effect=AssertionError("finisher must not be built"),
    ):
        assert await finish_reflection(_facts_event(), make_deps(embedding=None)) == []
