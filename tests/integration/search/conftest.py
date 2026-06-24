"""Session-scoped corpus fixture for ``tests/integration/search/``.

The pipeline that produces the search corpus (`/add` × 19 + `/flush` +
cascade drain) is the same one exercised by
``tests/integration/test_add_flush_pipeline_e2e.py`` — and it costs
~10 minutes against real LLMs. To keep the search test suite usable
in CI we run that pipeline **once per session** here, persist the
resulting memory_root to a session ``tmp_path``, and let every test
re-attach a fresh FastAPI lifespan against the on-disk corpus.

Layout::

    _ingested_memory_root  (session-scoped)
        └── ingests LoCoMo conv_0 via the HTTP API, then tears
            lifespan down. Returns the memory_root path with md +
            sqlite + lancedb populated on disk.

    search_client  (function-scoped)
        └── per-test ``httpx.AsyncClient`` wired to a freshly built
            FastAPI app, ``EVEROS_ROOT`` pointed at the
            session corpus. Singletons are reset so each test starts
            with cold caches and the lifespan is the only thing
            constructing them.

This is intentionally separate from ``tests/integration/conftest.py``
fixtures (which are function-scoped). Cross-suite isolation: tests
under ``search/`` cannot poison or be poisoned by the ones above.

All tests in this folder are marked ``slow`` via the module-level
``pytestmark`` in ``test_search_e2e.py`` — a non-``-m slow`` run skips
the whole suite cleanly without paying the ingest cost.
"""

from __future__ import annotations

import asyncio
import importlib
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text

# Set ``EVEROS_REUSE_CORPUS=<path>`` to skip ingest and point the
# session fixture at an existing memory_root (md + lancedb already
# populated). Search is a read-only path, so no copy is needed — the
# fixture just sets ``EVEROS_ROOT`` to that directory.
_REUSE_ENV = "EVEROS_REUSE_CORPUS"

# Memorize-service module-level lazy singletons; reset between phases so
# stale clients / engines don't leak from ingest into per-test lifespans.
_MEMORIZE_SINGLETONS: tuple[str, ...] = (
    "_episode_writer",
    "_prompt_loader",
    "_user_pipeline",
    "_agent_pipeline",
    "_ome_engine",
)


# ── Session-scoped MonkeyPatch ─────────────────────────────────────────


@pytest.fixture(scope="session")
def _session_monkeypatch() -> Generator[pytest.MonkeyPatch, None, None]:
    """A ``MonkeyPatch`` instance with session lifetime.

    Pytest's default ``monkeypatch`` is function-scoped. The ingest
    fixture below has to set env vars and null singletons before the
    lifespan even starts — those changes have to live for the whole
    session, so we open our own ``MonkeyPatch`` and undo it at session
    end.
    """
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


# ── Singleton reset helper ─────────────────────────────────────────────


def _reset_memorize_singletons(mp: pytest.MonkeyPatch) -> None:
    """Null out memorize/strategy/LLM-client lazy singletons.

    Called once before ingest (so the freshly-set ``EVEROS_ROOT``
    actually wins) and once per test (so the session corpus's lifespan
    sees clean caches).
    """
    from everos.config import load_settings

    load_settings.cache_clear()

    svc = importlib.import_module("everos.service.memorize")
    client_mod = importlib.import_module("everos.component.llm.client")
    af_mod = importlib.import_module("everos.memory.strategies.extract_atomic_facts")
    fs_mod = importlib.import_module("everos.memory.strategies.extract_foresight")

    for attr in _MEMORIZE_SINGLETONS:
        mp.setattr(svc, attr, None, raising=False)
    mp.setattr(client_mod, "_llm_client", None, raising=False)
    mp.setattr(af_mod, "_writer", None, raising=False)
    mp.setattr(fs_mod, "_writer", None, raising=False)


# ── Session corpus: ingest once ────────────────────────────────────────


@pytest.fixture(scope="session")
def _ingested_memory_root(
    tmp_path_factory: pytest.TempPathFactory,
    _session_monkeypatch: pytest.MonkeyPatch,
    long_conversation: dict,
) -> Path:
    """Run /add × 19 + /flush + cascade drain once; return the memory_root.

    All on-disk artifacts (md files + sqlite system.db + lancedb
    tables) survive lifespan teardown, so per-test fixtures can
    re-attach a fresh app against the populated root and exercise
    only the read path.

    Marked **slow** transitively via ``pytestmark`` in
    ``test_search_e2e.py`` — without ``-m slow`` the test module is
    deselected and this fixture is never instantiated.
    """
    reuse = os.environ.get(_REUSE_ENV)
    if reuse:
        memory_root = Path(reuse).expanduser().resolve()
        users_dir = memory_root / "default_app" / "default_project" / "users"
        if not users_dir.is_dir():
            raise AssertionError(
                f"{_REUSE_ENV}={memory_root} has no "
                "default_app/default_project/users/ subdir — point it at a "
                "fully-ingested memory_root or unset to rebuild from scratch"
            )
    else:
        memory_root = tmp_path_factory.mktemp("search_corpus")

    _session_monkeypatch.setenv("EVEROS_ROOT", str(memory_root))
    _reset_memorize_singletons(_session_monkeypatch)
    (memory_root / "ome.toml").write_text("# test\n")

    if reuse:
        # Search is read-only; the corpus is consumed in place, no copy.
        return memory_root

    # Drive the ingest in its own event loop. The lifespan inside
    # ``_ingest`` properly closes LanceDB / SQLite handles on exit so
    # the per-test lifespans can re-open them.
    asyncio.run(_ingest(memory_root, long_conversation))
    return memory_root


async def _ingest(memory_root: Path, long_conversation: dict) -> None:
    """Bring up the app once, push the LoCoMo fixture through /add+/flush."""
    from everos.entrypoints.api.app import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        session_id = long_conversation["everos_session_id"]
        for batch in long_conversation["batches"]:
            messages = [
                {
                    "sender_id": m["sender_id"],
                    "role": m["role"],
                    "timestamp": m["timestamp"],
                    "content": m["content"],
                }
                for m in batch["messages"]
            ]
            resp = await client.post(
                "/api/v1/memory/add",
                json={"session_id": session_id, "messages": messages},
                timeout=600.0,
            )
            resp.raise_for_status()

        resp = await client.post(
            "/api/v1/memory/flush",
            json={"session_id": session_id},
            timeout=600.0,
        )
        resp.raise_for_status()

        await _poll_cascade_drained(deadline_seconds=600.0)


async def _poll_cascade_drained(*, deadline_seconds: float) -> None:
    """Block until ``md_change_state.pending == 0`` or deadline."""
    from everos.infra.persistence.sqlite import md_change_state_repo

    async with asyncio.timeout(deadline_seconds):
        while True:
            summary = await md_change_state_repo.queue_summary()
            if summary.pending == 0:
                return
            await asyncio.sleep(0.5)


# ── Per-test client against the session corpus ─────────────────────────


@pytest_asyncio.fixture
async def search_client(
    _ingested_memory_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[httpx.AsyncClient]:
    """Per-test ``AsyncClient`` reading from the session corpus.

    Singletons are reset before the lifespan starts so the search
    manager builds a fresh embedding / rerank / LLM client per test —
    we don't want cross-test client state to mask a regression.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(_ingested_memory_root))
    _reset_memorize_singletons(monkeypatch)

    # The search service has its own module-level singletons; reset
    # those too so re-attach is clean.
    search_svc = importlib.import_module("everos.service.search")
    for attr in (
        "_manager",
        "_embedding",
        "_reranker",
        "_llm_client",
        "_embedding_resolved",
        "_rerank_resolved",
        "_llm_resolved",
    ):
        if hasattr(search_svc, attr):
            monkeypatch.setattr(
                search_svc,
                attr,
                None if not attr.endswith("_resolved") else False,
                raising=False,
            )

    from everos.entrypoints.api.app import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        yield client


# ── Diagnostic helpers (handy for tests that probe SQLite directly) ───


@pytest.fixture
def memcell_count() -> Callable[[], Awaitable[int]]:
    """Return an async callable: ``await memcell_count() -> int``."""

    async def _count() -> int:
        from everos.infra.persistence.sqlite import get_engine

        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT COUNT(*) FROM memcell"))
            return int(result.scalar() or 0)

    return _count
