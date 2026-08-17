"""``CascadeLifespanProvider`` boot contract under soft-embedding.

Two variants of the same startup path — one with the embedding provider
unconfigured (Tier 1), one with it configured (Tier 2+). Both must
build a working :class:`CascadeOrchestrator`, expose it on the FastAPI
app state, and shut it down cleanly. Round-2 review finding M12 turned
the tests' bodies from ``# Should NOT raise`` (pytest passes on any
non-raising path) into real assertions on the object the provider
actually built and stored — so a regression that silently no-ops the
startup would break the test rather than slip through green.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from sqlmodel import SQLModel

from everos.entrypoints.api.lifespans.cascade import CascadeLifespanProvider
from everos.infra.persistence.sqlite import dispose_engine, get_engine
from everos.memory.cascade import CascadeOrchestrator


@pytest.fixture(autouse=True)
def _clear_capability_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test starts with a fresh capability lookup."""
    import everos.component.embedding.accessor as acc

    monkeypatch.setattr(acc, "_capability", None)


@pytest.fixture
async def sqlite_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[None]:
    """Create the SQLite schema ``CascadeOrchestrator.start()`` reads/writes.

    Production wiring runs :class:`SqliteLifespanProvider` before
    :class:`CascadeLifespanProvider` (see ``cascade.py`` module
    docstring); these tests exercise :class:`CascadeLifespanProvider`
    standalone, so the schema has to be created explicitly — same as
    ``test_orchestrator.py``'s ``runtime`` fixture.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    await dispose_engine()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield
    await dispose_engine()


async def test_startup_succeeds_when_embedding_missing(
    sqlite_schema: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tier 1 (LLM only): cascade must boot with no embedding provider.

    Assertions replace the pre-M12 ``# Should NOT raise`` comment: the
    provider's :meth:`startup` returns the :class:`CascadeOrchestrator`
    it just built, the internal handle survives the round-trip, and
    :meth:`shutdown` clears it so a second cycle wouldn't double-stop.
    """
    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "")
    monkeypatch.setenv("EVEROS_EMBEDDING__API_KEY", "")
    provider = CascadeLifespanProvider()
    app = FastAPI()

    result = await provider.startup(app)

    assert isinstance(result, CascadeOrchestrator), (
        "startup must return the orchestrator it built, not None"
    )
    assert provider._orchestrator is result, (
        "provider must keep an internal reference so shutdown can stop it"
    )
    assert result._started is True, "orchestrator.start() must have run"

    await provider.shutdown(app)

    assert provider._orchestrator is None, (
        "shutdown must clear the internal reference so a second cycle is clean"
    )
    assert result._started is False, "shutdown must stop the orchestrator"


async def test_startup_still_works_when_embedding_present(
    sqlite_schema: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tier 2+: cascade must boot identically when embedding IS configured.

    Same assertions as the Tier 1 variant — the soft-dependency refactor
    must not have introduced a divergence in the "provider present" boot
    path, and the orchestrator instance shape is what downstream lifespan
    consumers rely on regardless of tier.
    """
    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setenv("EVEROS_EMBEDDING__API_KEY", "test-key")
    monkeypatch.setenv("EVEROS_EMBEDDING__BASE_URL", "https://api.example.com/v1")
    provider = CascadeLifespanProvider()
    app = FastAPI()

    result = await provider.startup(app)

    assert isinstance(result, CascadeOrchestrator)
    assert provider._orchestrator is result
    assert result._started is True

    await provider.shutdown(app)

    assert provider._orchestrator is None
    assert result._started is False
