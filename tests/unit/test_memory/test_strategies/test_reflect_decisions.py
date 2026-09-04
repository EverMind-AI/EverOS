"""Tests for the ``reflect_decisions`` Cron strategy.

Verifies decorator metadata (name, trigger type, emits, enabled flag).
The strategy body is a thin entry point — orchestrator logic is tested
separately in ``test_reflection/test_decision_orchestrator.py``.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, patch

import structlog.testing

from everos.component.embedding import EmbeddingCapability
from everos.infra.ome.testing import FakeStrategyContext
from everos.infra.ome.triggers import Cron
from everos.memory.events import DecisionExtracted
from everos.memory.strategies.reflect_decisions import reflect_decisions
from everos.service.memorize import _STRATEGIES_ALWAYS, _STRATEGIES_REQUIRE_EMBED


async def test_strategy_meta_is_attached() -> None:
    """Decorator stamps the expected StrategyMeta on the function."""
    meta = reflect_decisions.meta
    assert meta.name == "reflect_decisions"
    assert isinstance(meta.trigger, Cron)
    assert meta.trigger.expr == "0 2 * * 1"
    assert meta.emits == frozenset({DecisionExtracted})
    assert meta.max_retries == 1
    assert meta.enabled is False
    assert reflect_decisions in _STRATEGIES_REQUIRE_EMBED
    assert reflect_decisions not in _STRATEGIES_ALWAYS


async def test_strategy_is_callable() -> None:
    """The Strategy wrapper must be callable (delegates to async func)."""
    assert callable(reflect_decisions)
    assert inspect.iscoroutinefunction(reflect_decisions.meta.func)


async def test_returns_without_side_effects_when_embedding_unavailable() -> None:
    """Capability unavailable → early return; no cluster iteration.

    Reflection re-embeds merged decision text, so it cannot run without
    an embedder. Registration is unconditional so a runtime tier upgrade
    (Tier 1 → Tier 2) picks up on the next scheduled tick; a Tier-1
    cron tick simply no-ops.
    """
    from everos.infra.ome.events import CronTick

    with (
        patch(
            "everos.memory.strategies.reflect_decisions.get_embedding_capability",
            return_value=EmbeddingCapability(provider=None),
        ),
        patch(
            "everos.memory.strategies.reflect_decisions.cluster_repo"
        ) as mock_cluster_repo,
        structlog.testing.capture_logs() as captured,
    ):
        mock_cluster_repo.list_distinct_owners = AsyncMock(
            side_effect=AssertionError("cluster_repo must not be touched"),
        )
        await reflect_decisions(
            CronTick(strategy_name="reflect_decisions"), FakeStrategyContext()
        )

    gated = [
        e
        for e in captured
        if e.get("event") == "strategy_gated_off_embedding_unavailable"
    ]
    assert len(gated) == 1
    assert gated[0]["strategy_name"] == "reflect_decisions"
