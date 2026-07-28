"""``TracingLifespanProvider`` — reads [observability] and manages the
tracer provider over the app lifespan.

Startup returns whether tracing was enabled; shutdown always flushes and
tears down without raising. Span-capture behavior itself is covered in
``test_observability/test_tracing.py``; here we assert the lifespan wiring
contract only.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI

from everos.config import load_settings
from everos.core.lifespan import TracingLifespanProvider
from everos.core.observability.tracing import shutdown_tracing


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith("EVEROS_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    load_settings.cache_clear()
    shutdown_tracing()
    yield
    shutdown_tracing()


async def test_startup_returns_true_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_OBSERVABILITY__ENABLED", "true")
    monkeypatch.setenv("EVEROS_OBSERVABILITY__ENDPOINT", "http://collector.invalid")
    load_settings.cache_clear()
    provider = TracingLifespanProvider()
    app = FastAPI()
    result = await provider.startup(app)
    assert result is True
    await provider.shutdown(app)  # must not raise


async def test_startup_returns_false_when_disabled() -> None:
    provider = TracingLifespanProvider()
    app = FastAPI()
    result = await provider.startup(app)
    assert result is False
    await provider.shutdown(app)  # must not raise


def test_provider_has_low_order_to_start_first() -> None:
    # Tracer must be live before other providers start.
    assert TracingLifespanProvider().order < 5  # MetricsLifespanProvider is 5


async def test_startup_installs_score_sink_with_creds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With Langfuse creds + emit_recall_scores, startup installs the score
    sink; shutdown drains and tears it down."""
    from pydantic import SecretStr

    import everos.core.observability.tracing.scores as scores_mod
    from everos.config import Settings
    from everos.config.settings import ObservabilitySettings

    obs = ObservabilitySettings(
        enabled=True,
        endpoint="http://collector.invalid",
        langfuse_public_key="pk-lf",
        langfuse_secret_key=SecretStr("sk-lf"),
        langfuse_host="https://lf.example",
    )
    monkeypatch.setattr(
        "everos.core.lifespan.tracing_lifespan.load_settings",
        lambda: Settings(observability=obs),
    )
    provider = TracingLifespanProvider()
    app = FastAPI()
    await provider.startup(app)
    try:
        assert scores_mod._sink is not None
    finally:
        await provider.shutdown(app)
    assert scores_mod._sink is None
