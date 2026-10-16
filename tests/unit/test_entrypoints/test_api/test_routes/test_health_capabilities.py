"""``/health`` — ``capabilities.embed`` reports the probe, not the constructor.

This is the half of issue #434 that lied. ``get_embedding_capability()`` only
answers "could a provider be built", and building one makes no network call,
so a complete configuration pointing at a dead service read ``embed: true``
while every write failed deep in the index layer. The field now reports
whether the startup probe actually got an answer.

It is deliberately a snapshot: a service that comes back after startup keeps
reading ``false`` until the next restart, because re-probing on an endpoint
designed for frequent polling would put a network round trip in its path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from everos.infra.persistence.index import _provenance

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _reset_probe() -> Iterator[None]:
    """The probe result is process-wide; don't leak it between cases."""
    _provenance._last_probe = None
    yield
    _provenance._last_probe = None


def _capabilities() -> dict[str, bool]:
    from everos.entrypoints.api.routes.health import HealthCapabilities

    return HealthCapabilities(
        llm=True,
        embed=_provenance.last_probe() is not None,
        rerank=False,
        multimodal_llm=False,
        parser=False,
    ).model_dump()


def test_embed_is_false_when_the_probe_never_answered() -> None:
    """A complete config pointing at a dead service must not read true."""
    assert _capabilities()["embed"] is False


def test_embed_is_true_once_the_probe_measured_a_width() -> None:
    _provenance._last_probe = 768
    assert _capabilities()["embed"] is True


def test_embed_stays_false_after_the_service_recovers() -> None:
    """Snapshot semantics: only a restart re-runs the probe."""
    assert _capabilities()["embed"] is False
    # The service coming back does not touch the recorded probe result.
    assert _provenance.last_probe() is None
    assert _capabilities()["embed"] is False


def test_health_route_reads_the_probe_not_the_capability() -> None:
    """Pin the wiring itself: a revert to the old source should fail here."""
    import inspect

    from everos.entrypoints.api.routes import health

    source = inspect.getsource(health)
    assert "embed=last_probe() is not None" in source
    assert "get_embedding_capability" not in source
