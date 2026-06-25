"""Tests for the ``reflect_episodes`` Cron strategy.

Verifies decorator metadata (name, trigger type, emits, enabled flag).
The strategy body is a thin entry point — orchestrator logic is tested
separately in ``test_reflection/test_orchestrator.py``.
"""

from __future__ import annotations

import inspect

from everos.infra.ome.triggers import Cron
from everos.memory.events import EpisodeExtracted
from everos.memory.strategies.reflect_episodes import reflect_episodes


async def test_strategy_meta_is_attached() -> None:
    """Decorator stamps the expected StrategyMeta on the function."""
    meta = reflect_episodes.meta
    assert meta.name == "reflect_episodes"
    assert isinstance(meta.trigger, Cron)
    assert meta.trigger.expr == "0 2 * * 1"
    assert meta.emits == frozenset({EpisodeExtracted})
    assert meta.max_retries == 1
    assert meta.enabled is False


async def test_strategy_is_callable() -> None:
    """The Strategy wrapper must be callable (delegates to async func)."""
    assert callable(reflect_episodes)
    assert inspect.iscoroutinefunction(reflect_episodes.meta.func)
