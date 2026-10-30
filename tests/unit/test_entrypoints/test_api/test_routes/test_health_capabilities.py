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
_NEUTRAL = dataclasses.replace(
    _ALL_CAPABILITIES, rerank=False, multimodal_llm=False, parser=False
)


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
