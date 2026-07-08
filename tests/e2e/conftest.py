"""Shared fixtures for ``tests/e2e/``.

Provides:

- ``core_pipeline_runtime``: tmp memory root + reset memorize singletons.
  Uses the **real** LLM / embedding / rerank creds from ``.env`` per the
  project test policy.
- ``async_client``: ``httpx.AsyncClient`` wired into ``create_app()`` with
  the full lifespan stack (SQLite + LanceDB + Cascade + OME).
- ``cascade_done_poll``: wait until ``md_change_state`` queue is fully
  drained (``pending`` rows == 0; includes the internal ``processing``).
- ``pipeline_done_poll``: composite drain — waits until OME strategy runs AND
  ``md_change_state`` queue both drain (use for tests that exercise the full
  OME → md → cascade pipeline).
- ``buffer_count`` / ``memcell_count``: raw counts for buffer-delta and
  memcell-growth assertions.

The ``long_conversation`` fixture (LoCoMo conv_0) lives in
:mod:`tests.conftest` so both ``tests/e2e/`` and
``tests/integration/search/`` can depend on it.

Conventions:

- ``.env`` is loaded at import time (before any everos module reads
  settings) — overrides for ``EVEROS_ROOT`` happen per-test.
- This file does **not** define ``cascade_runtime`` — that name belongs
  to ``tests/integration/test_cascade_integration.py``'s local fixture.
  The pipeline test uses ``core_pipeline_runtime`` to avoid name
  collision.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import text

# Load real .env creds before any everos import touches load_settings().
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

_FIXTURE_DIR = _PROJECT_ROOT / "tests" / "fixtures"
_SEARCH_SEED_DIR = _FIXTURE_DIR / "search_seed"

# Memorize service module-level singletons that survive across tests; we
# null them out so each test rebuilds against its own ``tmp_path``.
_MEMORIZE_SINGLETONS: tuple[str, ...] = (
    "_episode_writer",
    "_prompt_loader",
    "_user_pipeline",
    "_agent_pipeline",
    "_ome_engine",
)

# OME strategy modules carry module-level lazy singletons (``_writer`` /
# ``_reader``) that capture ``MemoryRoot.default()`` at first call. They
# survive across tests, so the second test writes its output to the
# **first test's** tmp_path. Reset all of them per-test.
_STRATEGY_SINGLETONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("everos.memory.strategies.extract_atomic_facts", ("_writer",)),
    ("everos.memory.strategies.extract_foresight", ("_writer",)),
    ("everos.memory.strategies.extract_user_profile", ("_writer", "_reader")),
    ("everos.memory.strategies.extract_agent_case", ("_writer",)),
    ("everos.memory.strategies.extract_agent_skill", ("_writer",)),
)


def _reset_strategy_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Null every strategy ``_writer`` / ``_reader`` so the next test
    rebuilds against its own ``MemoryRoot.default()`` (driven by the
    fresh ``EVEROS_ROOT`` env var set by the calling fixture).
    """
    for mod_name, attrs in _STRATEGY_SINGLETONS:
        mod = importlib.import_module(mod_name)
        for attr in attrs:
            monkeypatch.setattr(mod, attr, None, raising=False)


# ---------------------------------------------------------------------------
# Data fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def search_seed() -> dict[str, list[dict]]:
    """Load the search seed slice produced by ``_dump_search_seed.py``.

    Returns a dict with four keys (``episode`` / ``atomic_fact`` /
    ``foresight`` / ``user_profile``); each value is a list of raw row
    dicts ready to be fed into ``Model.model_validate`` for LanceDB.

    Tests pick the subset they need and may mutate per-row fields
    (e.g. set distinct ``session_id`` values to exercise filter DSL)
    before instantiating the pydantic model.
    """
    return {
        name: json.loads((_SEARCH_SEED_DIR / f"{name}.json").read_text())
        for name in ("episode", "atomic_fact", "foresight", "user_profile")
    }


# ---------------------------------------------------------------------------
# Runtime fixture: tmp memory root + singleton reset (no app lifespan)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def core_pipeline_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Path]:
    """Prepare clean memory root + reset memorize singletons.

    Keeps real LLM / embedding settings from ``.env`` (do NOT overwrite
    ``EVEROS_LLM__*`` or ``EVEROS_EMBEDDING__*``).
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))

    from everos.config import load_settings

    load_settings.cache_clear()

    svc = importlib.import_module("everos.service.memorize")
    client_mod = importlib.import_module("everos.component.llm.client")

    for attr in _MEMORIZE_SINGLETONS:
        monkeypatch.setattr(svc, attr, None, raising=False)
    monkeypatch.setattr(client_mod, "_llm_client", None, raising=False)
    _reset_strategy_singletons(monkeypatch)

    yield tmp_path


# ---------------------------------------------------------------------------
# Async client fixture (full app lifespan)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def async_client(
    core_pipeline_runtime: Path,
) -> AsyncIterator[httpx.AsyncClient]:
    """Bring up the full everos app with lifespan, return an httpx client.

    The lifespan starts: SQLite engine, LanceDB connection + business
    indexes, Cascade orchestrator (watcher + scanner + worker), OME
    engine. Teardown stops everything in reverse.
    """
    from everos.entrypoints.api.app import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)

    # Drive starlette's lifespan_context explicitly — httpx.ASGITransport
    # does not run startup / shutdown on its own.
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        yield client


# ---------------------------------------------------------------------------
# Poll helpers
# ---------------------------------------------------------------------------


async def _poll(
    condition: Callable[[], Awaitable[bool]],
    *,
    deadline_seconds: float,
    interval: float = 0.5,
) -> None:
    """Poll an async predicate until truthy; ``TimeoutError`` on deadline."""
    async with asyncio.timeout(deadline_seconds):
        while True:
            if await condition():
                return
            await asyncio.sleep(interval)


@pytest.fixture
def cascade_done_poll() -> Callable[..., Awaitable[None]]:
    """Wait until ``md_change_state`` queue is drained (no pending/processing)."""

    async def _wait(*, deadline_seconds: float = 180.0) -> None:
        from everos.infra.persistence.sqlite import md_change_state_repo

        async def _drained() -> bool:
            summary = await md_change_state_repo.queue_summary()
            # `pending` includes the internal `processing` rows (see QueueSummary).
            return summary.pending == 0

        await _poll(_drained, deadline_seconds=deadline_seconds)

    return _wait


@pytest.fixture
def pipeline_done_poll() -> Callable[..., Awaitable[None]]:
    """Wait until OME strategy runs AND ``md_change_state`` queue both drain.

    Composite drain — fixes the trap where :func:`cascade_done_poll`
    alone returns immediately while a slow LLM-driven strategy is still
    in flight (the strategy has not written md yet, so the cascade queue
    is momentarily empty). Pipeline tests that touch the full async
    chain (OME -> md -> cascade -> LanceDB) must use this instead of
    ``cascade_done_poll``.
    """

    async def _wait(*, deadline_seconds: float = 180.0) -> None:
        from everos.infra.persistence.sqlite import md_change_state_repo
        from everos.service.memorize import _get_engine

        engine = _get_engine()

        async def _drained() -> bool:
            # OME side first: cascade can only fire after a strategy
            # writes md, so an in-flight run means the queue check below
            # is premature.
            if not await engine.wait_idle(timeout=0.5):
                return False
            # `pending` includes the internal `processing` rows (see
            # QueueSummary).
            summary = await md_change_state_repo.queue_summary()
            return summary.pending == 0

        await _poll(_drained, deadline_seconds=deadline_seconds)

    return _wait


# ---------------------------------------------------------------------------
# Count helpers (used directly by tests for buffer-delta assertions)
# ---------------------------------------------------------------------------


@pytest.fixture
def buffer_count() -> Callable[[str], Awaitable[int]]:
    """Return an async callable: ``await buffer_count(session_id) -> int``."""

    async def _count(session_id: str) -> int:
        from everos.infra.persistence.sqlite import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT COUNT(*) FROM unprocessed_buffer WHERE session_id = :sid"),
                {"sid": session_id},
            )
            return int(result.scalar() or 0)

    return _count


@pytest.fixture
def memcell_count() -> Callable[[str], Awaitable[int]]:
    """Return an async callable: ``await memcell_count(user_id_or_session) -> int``.

    Counts memcell rows; pass session_id to count by session, or omit to
    count all.
    """

    async def _count(session_id: str | None = None) -> int:
        from everos.infra.persistence.sqlite import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            if session_id is None:
                result = await conn.execute(text("SELECT COUNT(*) FROM memcell"))
            else:
                result = await conn.execute(
                    text("SELECT COUNT(*) FROM memcell WHERE session_id = :sid"),
                    {"sid": session_id},
                )
            return int(result.scalar() or 0)

    return _count
