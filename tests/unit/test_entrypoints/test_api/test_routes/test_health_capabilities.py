"""``/health`` — ``capabilities.embed`` reports the probe, not the constructor.

This is the half of issue #434 that lied. Whether an embedding provider was
configured only answers "could a provider be built", and building one makes
no network call, so a complete configuration pointing at a dead service read
``embed: true`` while every write failed deep in the index layer. The field
now reports whether the startup probe actually got an answer.

It is deliberately a snapshot: a service that comes back after startup keeps
reading ``false`` until the next restart, because re-probing on an endpoint
designed for frequent polling would put a network round trip in its path.

Every case calls the route. An earlier version of this module rebuilt the
route's expression in a helper and asserted on the copy, which stayed green
when the route itself was changed to ``embed=False``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from everos.entrypoints.api.routes import health as health_route
from tests.fakes import make_application
from tests.fakes.application import _ALL_CAPABILITIES

# Everything the route reads except the one field under test is pinned off.
_NEUTRAL = dataclasses.replace(_ALL_CAPABILITIES, multimodal_llm=False, parser=False)


async def _embed_from_the_route(*, probed: bool, embedding: bool = True) -> bool:
    """Call the handler and read ``capabilities.embed`` out of its response."""
    application = make_application(
        capabilities=dataclasses.replace(_NEUTRAL, embedding=embedding)
    )
    application.ops.probed = probed

    response: Any = await health_route.health(application)
    return response.capabilities.embed


async def test_embed_is_false_when_the_probe_never_answered() -> None:
    """A complete config pointing at a dead service must not read true.

    This is the state the old field got wrong: the provider builds fine, so
    ``available`` was ``True`` while nothing could actually be embedded.
    """
    assert await _embed_from_the_route(probed=False) is False


async def test_embed_is_true_once_the_probe_measured_a_width() -> None:
    assert await _embed_from_the_route(probed=True) is True


async def test_embed_follows_the_probe_not_the_capability() -> None:
    """The two disagree exactly in the case that matters.

    A constructible provider pointing at an unreachable service: the old
    source says yes, the probe says no, and the route has to follow the probe.
    """
    assert await _embed_from_the_route(probed=False, embedding=True) is False


# ── the decider (the model every search round asks) ─────────────────────────


class _DeciderEndpoint:
    """A decider reached through ``achat_raw``; fails with ``error`` when set."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    async def achat_raw(self, messages: Any, **params: Any) -> Any:
        from everos.component.llm.openai_provider import RawReply

        self.calls += 1
        if self.error is not None:
            raise self.error
        return RawReply(text="O", finish_reason="length", usage=None)


def _decider_status(endpoint: _DeciderEndpoint) -> Any:
    from everos.memory.search import DeciderStatus
    from tests.fakes import static_tuning

    return DeciderStatus(
        endpoint,
        model="qwen3.5-9b-decider",
        base_url="http://127.0.0.1:9999/v1",
        tuning_source=static_tuning(),
    )


async def test_health_lists_the_decider_and_its_probe() -> None:
    status = _decider_status(_DeciderEndpoint())
    await status.probe()
    application = make_application(capabilities=_NEUTRAL)
    application.ops.decider = status

    response: Any = await health_route.health(application)

    assert response.capabilities.decider is True
    assert response.decider.model == "qwen3.5-9b-decider"
    assert response.decider.base_url == "http://127.0.0.1:9999/v1"
    assert response.decider.reachable is True
    assert response.decider.probe_error is None


async def test_an_unreachable_decider_reads_false_with_the_reason() -> None:
    status = _decider_status(_DeciderEndpoint(RuntimeError("Connection refused")))
    await status.probe()
    application = make_application(capabilities=_NEUTRAL)
    application.ops.decider = status

    response: Any = await health_route.health(application)

    assert response.capabilities.decider is False
    assert response.decider.reachable is False
    assert response.decider.probe_error == "RuntimeError: Connection refused"


async def test_ops_start_probes_the_decider_in_the_background() -> None:
    import asyncio

    from tests.fakes.application import make_ops

    endpoint = _DeciderEndpoint()
    ops = make_ops(decider=_decider_status(endpoint))
    await ops.start()
    for _ in range(50):
        if (await ops.health()).decider.reachable is not None:
            break
        await asyncio.sleep(0.01)
    report = await ops.health()
    await ops.aclose()

    assert endpoint.calls == 1
    assert report.providers.decider is True
    assert report.decider.reachable is True


# ── the offline engine's failed runs ────────────────────────────────────────


async def test_health_counts_the_engines_failed_runs() -> None:
    from datetime import UTC, datetime

    from everos.service.ops import StrategyFailures

    at = datetime(2026, 11, 13, tzinfo=UTC)
    failures = StrategyFailures()
    failures.record(
        "update_group_profile",
        "DeferredFailure: group profile update of 'trip' failed (invalid_reply): "
        "its batch of 2 segments is kept and read again in 600 s (failure 1 of 3)",
        at,
    )
    failures.record("extract_user_profile", "RuntimeError: boom\nTraceback ...", at)
    application = make_application(capabilities=_NEUTRAL)
    application.ops.strategy_failures = failures

    response: Any = await health_route.health(application)

    assert response.ome.failed_runs == 2
    assert response.ome.by_strategy == {
        "update_group_profile": 1,
        "extract_user_profile": 1,
    }
    assert response.ome.last_failure == "extract_user_profile: RuntimeError: boom"
    assert response.ome.last_failure_at == at


async def test_health_has_no_engine_block_without_a_counter() -> None:
    response: Any = await health_route.health(make_application(capabilities=_NEUTRAL))
    assert response.ome is None
