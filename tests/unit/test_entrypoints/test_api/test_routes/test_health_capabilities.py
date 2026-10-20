"""``/health`` — ``capabilities.embed`` reports the probe, not the constructor.

This is the half of issue #434 that lied. ``get_embedding_capability()`` only
answers "could a provider be built", and building one makes no network call,
so a complete configuration pointing at a dead service read ``embed: true``
while every write failed deep in the index layer. The field now reports
whether the startup probe actually got an answer.

It is deliberately a snapshot: a service that comes back after startup keeps
reading ``false`` until the next restart, because re-probing on an endpoint
designed for frequent polling would put a network round trip in its path.

Every case calls the route. An earlier version of this module rebuilt the
route's expression in a helper and asserted on the copy, which stayed green
when the route itself was changed to ``embed=False``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from everos.entrypoints.api.routes import health as health_route

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _neutral_capabilities(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin everything the route reads except the one field under test."""

    class _Cap:
        available = False

    monkeypatch.setattr(health_route, "get_rerank_capability", lambda: _Cap())
    monkeypatch.setattr(health_route, "get_multimodal_llm_capability", lambda: _Cap())
    monkeypatch.setattr(health_route, "parser_available", lambda: False)
    yield


async def _embed_from_the_route() -> bool:
    """Call the handler and read ``capabilities.embed`` out of its response."""

    class _App:
        state = type("S", (), {})()

    class _Request:
        app = _App()

    response: Any = await health_route.health(_Request())  # type: ignore[arg-type]
    return response.capabilities.embed


async def test_embed_is_false_when_the_probe_never_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete config pointing at a dead service must not read true.

    This is the state the old field got wrong: the provider builds fine, so
    ``available`` was ``True`` while nothing could actually be embedded.
    """
    monkeypatch.setattr(health_route, "last_probe", lambda: None)
    assert await _embed_from_the_route() is False


async def test_embed_is_true_once_the_probe_measured_a_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(health_route, "last_probe", lambda: 768)
    assert await _embed_from_the_route() is True


async def test_embed_follows_the_probe_not_the_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two disagree exactly in the case that matters.

    A constructible provider pointing at an unreachable service: the old
    source says yes, the probe says no, and the route has to follow the probe.
    """

    class _Available:
        available = True

    monkeypatch.setattr(
        health_route, "get_embedding_capability", lambda: _Available(), raising=False
    )
    monkeypatch.setattr(health_route, "last_probe", lambda: None)
    assert await _embed_from_the_route() is False
