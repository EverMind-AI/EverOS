"""Shared fixtures for ``tests/integration/test_tiers/``.

Three runtime fixtures build the real FastAPI app (``create_app()`` with
the full lifespan stack: LLM -> SQLite -> LanceDB -> Cascade -> OME)
against a per-test ``EVEROS_ROOT``, differing only in which capability
singletons (embedding / rerank) are wired to a working stub provider:

- ``tier1_runtime`` — LLM only. embed/rerank both unavailable.
- ``tier2_runtime`` — LLM + embed. rerank unavailable.
- ``tier3_runtime`` — LLM + embed + rerank, all available.

Every fixture yields a ready ``httpx.AsyncClient`` wired to the app via
``ASGITransport`` with the lifespan already started (mirrors
``tests/e2e/conftest.py``'s ``async_client`` fixture, but swaps the real
``.env`` credentials for deterministic stubs so the suite runs under
``make integration`` without live LLM/embed/rerank credentials).

LLM stubbing notes:

- ``everos.component.llm.client._llm_client`` is the process-wide
  singleton every OME strategy + the memorize pipeline read through
  ``get_llm_client()``. Pre-seeding it with a ``FakeLLMClient`` before
  the lifespan starts means ``LLMLifespanProvider.startup`` (which calls
  ``get_llm_client()``) sees the fake immediately and never touches
  settings.
- ``everos.service.search`` builds its **own** LLM client straight from
  settings (bypassing the singleton above), only to satisfy the
  ``enable_llm_rerank=True`` non-None guard in
  ``SearchManager._validate_components``. That LLM is never actually
  invoked in this suite: ``everalgo.rank.rerank._basic_arank`` only
  calls the LLM when ``enable_rerank and scored`` — and these tests
  never seed agent_case/agent_skill data, so ``scored`` is always empty.
  A placeholder ``FakeLLMClient(responses=[])`` is therefore sufficient;
  if that assumption ever breaks, the fake raises loudly instead of
  silently degrading the test.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient

from everos.component.embedding import EmbeddingCapability, EmbeddingProvider
from everos.component.rerank import RerankCapability, RerankResult

# ---------------------------------------------------------------------------
# Stub providers
# ---------------------------------------------------------------------------

_DIM = 1024


class StubEmbedder(EmbeddingProvider):
    """Deterministic 1024-dim vector; no network calls."""

    dim = _DIM

    async def embed(self, text: str) -> list[float]:
        return [0.1] * self.dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class StubReranker:
    """Deterministic reranker — returns candidates in original order."""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        instruction: str | None = None,
    ) -> list[RerankResult]:
        return [
            RerankResult(index=i, score=1.0 - i * 0.01) for i in range(len(documents))
        ]


# ---------------------------------------------------------------------------
# Fake LLM (boundary detection + episode extraction)
# ---------------------------------------------------------------------------


def _boundary_response(boundaries: list[int]) -> str:
    payload = {"reasoning": "test", "boundaries": boundaries, "should_wait": False}
    return json.dumps(payload)


def _episode_response(title: str = "Test Subject", content: str = "Test body") -> str:
    return json.dumps({"title": title, "content": content})


def make_fake_llm(
    boundary_responses: list[list[int]] | None = None,
    *,
    episode_title: str = "Hiking",
    episode_content: str = "Alice loves hiking in the mountains every weekend.",
) -> FakeLLMClient:
    """Build a ``FakeLLMClient`` that dispatches by prompt fingerprint.

    Mirrors ``tests/integration/test_memorize_integration.py``'s
    ``_make_fake_llm``: pops one ``boundaries=...`` entry per boundary
    prompt seen, every episode prompt gets the same canned
    ``{title, content}``. Any other prompt (atomic facts / foresight /
    profile / agent case background strategies) also falls through to
    the episode-shaped response; those strategies run as OME background
    jobs that log-and-continue on a parse failure, so they never affect
    the HTTP response under test.
    """
    boundary_queue: list[list[int]] = list(boundary_responses or [])

    def handler(messages: list[LLMChatMessage], **_: Any) -> ChatResponse:
        prompt = messages[0].content
        if "boundaries" in prompt.lower() or "memcell" in prompt.lower():
            cuts = boundary_queue.pop(0) if boundary_queue else []
            return ChatResponse(content=_boundary_response(cuts), model="fake")
        return ChatResponse(
            content=_episode_response(episode_title, episode_content),
            model="fake",
        )

    return FakeLLMClient(handler=handler)


# ---------------------------------------------------------------------------
# Module-level singleton reset (mirrors tests/e2e/conftest.py)
# ---------------------------------------------------------------------------

_MEMORIZE_SINGLETONS: tuple[str, ...] = (
    "_episode_writer",
    "_prompt_loader",
    "_user_pipeline",
    "_agent_pipeline",
    "_ome_engine",
)

_STRATEGY_SINGLETONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("everos.memory.strategies.extract_atomic_facts", ("_writer",)),
    ("everos.memory.strategies.extract_foresight", ("_writer",)),
    ("everos.memory.strategies.extract_user_profile", ("_writer", "_reader")),
    ("everos.memory.strategies.extract_agent_case", ("_writer",)),
    ("everos.memory.strategies.extract_agent_skill", ("_writer",)),
)


def _reset_strategy_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    for mod_name, attrs in _STRATEGY_SINGLETONS:
        mod = importlib.import_module(mod_name)
        for attr in attrs:
            monkeypatch.setattr(mod, attr, None, raising=False)


def _reset_search_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    search_svc = importlib.import_module("everos.service.search")
    monkeypatch.setattr(search_svc, "_manager", None, raising=False)
    monkeypatch.setattr(search_svc, "_llm_client", None, raising=False)
    monkeypatch.setattr(search_svc, "_llm_resolved", False, raising=False)


# ---------------------------------------------------------------------------
# Tier runtime builder
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _tier_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    embed_available: bool,
    rerank_available: bool,
) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_MEMORIZE__MODE", "chat")

    from everos.config import load_settings

    load_settings.cache_clear()

    (tmp_path / ".index" / "sqlite").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ome.toml").write_text("# test\n")

    svc = importlib.import_module("everos.service.memorize")
    for attr in _MEMORIZE_SINGLETONS:
        monkeypatch.setattr(svc, attr, None, raising=False)
    _reset_strategy_singletons(monkeypatch)
    _reset_search_singletons(monkeypatch)

    client_mod = importlib.import_module("everos.component.llm.client")
    monkeypatch.setattr(client_mod, "_llm_client", make_fake_llm(), raising=False)

    embed_acc = importlib.import_module("everos.component.embedding.accessor")
    rerank_acc = importlib.import_module("everos.component.rerank.accessor")
    stub_embedder = StubEmbedder() if embed_available else None
    stub_reranker = StubReranker() if rerank_available else None
    monkeypatch.setattr(
        embed_acc, "_capability", EmbeddingCapability(provider=stub_embedder)
    )
    monkeypatch.setattr(
        rerank_acc, "_capability", RerankCapability(provider=stub_reranker)
    )

    # service.search's independent LLM lazy-build -- see module docstring.
    search_svc = importlib.import_module("everos.service.search")
    monkeypatch.setattr(
        search_svc, "_llm_client", FakeLLMClient(responses=[]), raising=False
    )
    monkeypatch.setattr(search_svc, "_llm_resolved", True, raising=False)

    from everos.entrypoints.api.app import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        yield client


@pytest_asyncio.fixture
async def tier1_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[httpx.AsyncClient]:
    """Tier 1: LLM only. embed/rerank/multimodal all unavailable."""
    async with _tier_client(
        tmp_path, monkeypatch, embed_available=False, rerank_available=False
    ) as client:
        yield client


@pytest_asyncio.fixture
async def tier2_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[httpx.AsyncClient]:
    """Tier 2: LLM + embed. rerank unavailable."""
    async with _tier_client(
        tmp_path, monkeypatch, embed_available=True, rerank_available=False
    ) as client:
        yield client


@pytest_asyncio.fixture
async def tier3_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[httpx.AsyncClient]:
    """Tier 3: LLM + embed + rerank, all available."""
    async with _tier_client(
        tmp_path, monkeypatch, embed_available=True, rerank_available=True
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Poll helpers
# ---------------------------------------------------------------------------


async def _poll(
    condition: Callable[[], Awaitable[bool]],
    *,
    deadline_seconds: float,
    interval: float = 0.2,
) -> None:
    async with asyncio.timeout(deadline_seconds):
        while True:
            if await condition():
                return
            await asyncio.sleep(interval)


@asynccontextmanager
async def cascade_progress() -> AsyncIterator[Callable[..., Awaitable[None]]]:
    """Track cascade queue progress across a block of writes.

    Snapshots ``md_change_state_repo.queue_summary()`` on entry and
    yields a ``wait(deadline_seconds=...)`` callable that blocks until
    the queue has both drained (``pending == 0``) *and* actually
    advanced past the snapshot (``done + failed`` grew by at least one
    row) since it was taken.

    A plain ``pending == 0`` poll (no baseline) is a false-negative trap
    here: the cascade watcher enqueues asynchronously off a filesystem
    event, so a poll that starts before the watcher has picked up the
    write sees an empty queue and returns immediately, before the row
    was ever processed. Requiring forward progress past a pre-write
    snapshot closes that race.
    """
    from everos.infra.persistence.sqlite import md_change_state_repo

    baseline = await md_change_state_repo.queue_summary()
    baseline_total = (
        baseline.done + baseline.failed_retryable + baseline.failed_permanent
    )

    async def _wait(*, deadline_seconds: float = 40.0, min_processed: int = 1) -> None:
        async def _progressed() -> bool:
            summary = await md_change_state_repo.queue_summary()
            total = summary.done + summary.failed_retryable + summary.failed_permanent
            return summary.pending == 0 and (total - baseline_total) >= min_processed

        await _poll(_progressed, deadline_seconds=deadline_seconds)

    yield _wait


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------


async def add_and_flush(
    client: httpx.AsyncClient,
    *,
    session_id: str,
    sender_id: str = "u_alice",
    content: str = "Alice loves hiking in the mountains every weekend.",
    deadline_seconds: float = 40.0,
) -> dict[str, Any]:
    """Add one message, force extraction via ``/flush``, wait for cascade.

    A single ``/add`` call only accumulates into the boundary-detection
    buffer (the fake LLM's boundary detector returns no cuts by
    default); ``/flush`` forces ``is_final=True`` so the pipeline always
    extracts -- mirrors real client usage (flush at session end).
    Returns the flush response body once the resulting episode has been
    cascaded into LanceDB.
    """
    async with cascade_progress() as wait_drained:
        resp = await client.post(
            "/api/v1/memory/add",
            json=add_payload(
                session_id=session_id, sender_id=sender_id, content=content
            ),
        )
        assert resp.status_code == 200, resp.text

        flush_resp = await client.post(
            "/api/v1/memory/flush", json={"session_id": session_id}
        )
        assert flush_resp.status_code == 200, flush_resp.text
        assert flush_resp.json()["data"]["status"] == "extracted"

        await wait_drained(deadline_seconds=deadline_seconds)

    return flush_resp.json()


async def seed_atomic_fact_for_episode(
    episode_row: dict[str, Any],
    *,
    vector: list[float],
    fact: str = "Alice enjoys hiking in the mountains.",
) -> None:
    """Seed one real ``AtomicFact`` row linked to an already-cascaded episode.

    The manager's VECTOR method (``_maxsim_atomic_recall``) does not ANN
    -search the episode table directly -- it scans ``atomic_fact``
    (denser, finer-grained) and max-pools back to the parent episode via
    ``AtomicFact.parent_id == Episode.entry_id`` (see
    ``extract_atomic_facts.py``: ``parent_id=event.episode_entry_id``).
    A Tier 2/3 test that adds one episode and expects ``method="vector"``
    to find it therefore also needs at least one embedded atomic fact --
    normally produced by the ``extract_atomic_facts`` OME strategy, which
    this suite's fake LLM does not attempt to satisfy (its canned
    response only matches the boundary/episode JSON contract). Seeding
    the fact row directly exercises the same recall code path without
    depending on a second, unrelated LLM JSON contract.
    """
    import hashlib

    from everos.component.utils.datetime import get_utc_now
    from everos.infra.persistence.lancedb import AtomicFact, atomic_fact_repo

    entry_id = f"af_seed_{episode_row['entry_id']}"
    owner_id = episode_row["owner_id"]
    await atomic_fact_repo.add(
        [
            AtomicFact(
                id=f"{owner_id}_{entry_id}",
                entry_id=entry_id,
                owner_id=owner_id,
                owner_type=episode_row["owner_type"],
                app_id=episode_row["app_id"],
                project_id=episode_row["project_id"],
                session_id=episode_row.get("session_id"),
                timestamp=get_utc_now(),
                parent_id=episode_row["entry_id"],
                sender_ids=episode_row["sender_ids"],
                fact=fact,
                fact_tokens=fact.lower(),
                md_path=f"users/{owner_id}/.atomic_facts/atomic_fact-seed.md",
                content_sha256=hashlib.sha256(entry_id.encode()).hexdigest(),
                vector=vector,
            )
        ]
    )


def add_payload(
    *,
    session_id: str,
    sender_id: str = "u_alice",
    content: str = "I love hiking in the mountains every weekend.",
    timestamp: int = 1_700_000_000_000,
    app_id: str = "default",
    project_id: str = "default",
) -> dict[str, Any]:
    """Build a minimal ``POST /api/v1/memory/add`` request body."""
    return {
        "session_id": session_id,
        "app_id": app_id,
        "project_id": project_id,
        "messages": [
            {
                "sender_id": sender_id,
                "role": "user",
                "timestamp": timestamp,
                "content": content,
            }
        ],
    }
