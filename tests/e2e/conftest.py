"""Shared fixtures for ``tests/e2e/``.

Provides:

- ``core_pipeline_runtime``: tmp memory root + packaged ``ome.toml``.
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
import json
import os
import shutil
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
import pytest_asyncio
from dotenv import load_dotenv
from sqlalchemy import text

if TYPE_CHECKING:
    from everos.service import Application
    from tests.fakes import StorageView

# Load real .env creds before any everos import touches load_settings().
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env", override=False)

_FIXTURE_DIR = _PROJECT_ROOT / "tests" / "fixtures"
_SEARCH_SEED_DIR = _FIXTURE_DIR / "search_seed"


@pytest.fixture(autouse=True)
def _strongly_consistent_lancedb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every LanceDB read see the latest committed version.

    The application and the test reach the same tables through separate
    connections (see ``storage_view``). LanceDB's default never refreshes an
    open table, so rows one side writes after the other opened the table
    would stay invisible to it.
    """
    monkeypatch.setenv("EVEROS_LANCEDB__READ_CONSISTENCY_SECONDS", "0")


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
# Runtime fixture: tmp memory root (each test builds its own application)
# ---------------------------------------------------------------------------


def _index_backends() -> list[str]:
    """Backends the e2e suite runs against.

    LanceDB is always exercised. Milvus joins only when a real server is
    configured — it is a remote service with no embedded fallback, so a run
    without ``EVEROS_TEST_MILVUS_URI`` proves nothing about it and should not
    collect cases that suggest otherwise.
    """
    backends = ["lancedb"]
    if os.environ.get("EVEROS_TEST_MILVUS_URI"):
        backends.append("milvus")
    return backends


@pytest_asyncio.fixture(params=_index_backends(), ids=lambda b: f"index={b}")
async def index_backend(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[str]:
    """Point the derived index at one backend for the duration of a test."""
    backend = request.param
    monkeypatch.setenv("EVEROS_INDEX__BACKEND", backend)
    if backend == "milvus":
        monkeypatch.setenv("EVEROS_MILVUS__URI", os.environ["EVEROS_TEST_MILVUS_URI"])
        monkeypatch.setenv(
            "EVEROS_MILVUS__TOKEN", os.environ.get("EVEROS_TEST_MILVUS_TOKEN", "")
        )
        monkeypatch.setenv(
            "EVEROS_MILVUS__DB_NAME", os.environ.get("EVEROS_TEST_MILVUS_DB_NAME", "")
        )
        # Collections are remote and outlive the process, so each test needs
        # its own namespace or state leaks between them.
        monkeypatch.setenv(
            "EVEROS_MILVUS__COLLECTION_PREFIX", f"everos_e2e_{uuid.uuid4().hex}"
        )

    yield backend

    if backend == "milvus":
        from everos.config import load_settings
        from everos.infra.persistence import MemoryRoot
        from everos.infra.persistence.factories import build_derived_index

        load_settings.cache_clear()
        index = build_derived_index(load_settings(), MemoryRoot.resolve())
        try:
            await index.drop_business_tables()
        finally:
            await index.shutdown()


@pytest_asyncio.fixture
async def core_pipeline_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index_backend: str,
) -> AsyncIterator[Path]:
    """Prepare a clean memory root (each test builds its own application).

    Keeps real LLM / embedding settings from ``.env`` (do NOT overwrite
    ``EVEROS_LLM__*`` or ``EVEROS_EMBEDDING__*``).
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))

    from everos.config import load_settings

    load_settings.cache_clear()

    # The full-app lifespan starts OME, whose config reloader requires an
    # ``ome.toml`` in the memory root (normally created by ``everos init``).
    # Provision the packaged default so the pipeline e2e can boot; OME
    # strategies are code-registered (see service/bootstrap.py), so the
    # default config is sufficient for extraction.
    default_ome = files("everos.config") / "default_ome.toml"
    shutil.copyfile(str(default_ome), str(tmp_path / "ome.toml"))

    yield tmp_path


# ---------------------------------------------------------------------------
# Async client fixture (full app lifespan)
# ---------------------------------------------------------------------------


@pytest.fixture
def e2e_application(core_pipeline_runtime: Path) -> Application:
    """The application one test's app runs, built over its memory root."""
    from everos.config import load_settings
    from everos.service import build_application

    return build_application(load_settings())


@pytest_asyncio.fixture
async def async_client(
    e2e_application: Application,
) -> AsyncIterator[httpx.AsyncClient]:
    """Bring up the full everos app with lifespan, return an httpx client.

    The lifespan starts: SQLite engine, LanceDB connection + business
    indexes, Cascade orchestrator (watcher + scanner + worker), OME
    engine. Teardown stops everything in reverse.
    """
    from everos.entrypoints.api.app import create_app

    app = create_app(application=e2e_application)
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
def cascade_done_poll(
    core_pipeline_runtime: Path, storage_view: StorageView
) -> Callable[..., Awaitable[None]]:
    """Wait until ``md_change_state`` queue is drained (no pending/processing)."""

    async def _wait(*, deadline_seconds: float = 180.0) -> None:
        async def _drained() -> bool:
            summary = await storage_view.sqlite.md_change_state.queue_summary()
            # `pending` includes the internal `processing` rows (see QueueSummary).
            return summary.pending == 0

        await _poll(_drained, deadline_seconds=deadline_seconds)

    return _wait


@pytest.fixture
def pipeline_done_poll(
    e2e_application: Application, storage_view: StorageView
) -> Callable[..., Awaitable[None]]:
    """Wait until OME strategy runs AND ``md_change_state`` queue both drain.

    Composite drain — fixes the trap where :func:`cascade_done_poll`
    alone returns immediately while a slow LLM-driven strategy is still
    in flight (the strategy has not written md yet, so the cascade queue
    is momentarily empty). Pipeline tests that touch the full async
    chain (OME -> md -> cascade -> LanceDB) must use this instead of
    ``cascade_done_poll``.
    """

    async def _wait(*, deadline_seconds: float = 180.0) -> None:
        engine = e2e_application._engine

        async def _drained() -> bool:
            # OME side first: cascade can only fire after a strategy
            # writes md, so an in-flight run means the queue check below
            # is premature.
            if not await engine.wait_idle(timeout=0.5):
                return False
            # `pending` includes the internal `processing` rows (see
            # QueueSummary).
            summary = await storage_view.sqlite.md_change_state.queue_summary()
            return summary.pending == 0

        await _poll(_drained, deadline_seconds=deadline_seconds)

    return _wait


# ---------------------------------------------------------------------------
# Count helpers (used directly by tests for buffer-delta assertions)
# ---------------------------------------------------------------------------


@pytest.fixture
def buffer_count(
    core_pipeline_runtime: Path, storage_view: StorageView
) -> Callable[[str], Awaitable[int]]:
    """Return an async callable: ``await buffer_count(session_id) -> int``."""

    async def _count(session_id: str) -> int:
        async with storage_view.database.engine.connect() as conn:
            result = await conn.execute(
                text("SELECT COUNT(*) FROM unprocessed_buffer WHERE session_id = :sid"),
                {"sid": session_id},
            )
            return int(result.scalar() or 0)

    return _count


@pytest.fixture
def memcell_count(
    core_pipeline_runtime: Path, storage_view: StorageView
) -> Callable[[str], Awaitable[int]]:
    """Return an async callable: ``await memcell_count(user_id_or_session) -> int``.

    Counts memcell rows; pass session_id to count by session, or omit to
    count all.
    """

    async def _count(session_id: str | None = None) -> int:
        async with storage_view.database.engine.connect() as conn:
            if session_id is None:
                result = await conn.execute(text("SELECT COUNT(*) FROM memcell"))
            else:
                result = await conn.execute(
                    text("SELECT COUNT(*) FROM memcell WHERE session_id = :sid"),
                    {"sid": session_id},
                )
            return int(result.scalar() or 0)

    return _count
