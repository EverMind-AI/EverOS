"""The ordering rule through real parts: route -> ``has_pending`` -> worker.

A synchronous ``/add`` on a session whose earlier ``async_mode`` batch is still
queued must itself be queued (202) and run after it; a session without a
backlog still runs inline (200). Only ``memorize()`` is faked (no LLM); the
``memorize_inbox`` table, the worker and the route are real. The worker is
not started: the test drains it by hand so the interleaving is deterministic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from everos.service import MemorizeResult, MemoryService, build_application
from everos.service.memory import _queue as mq
from tests.fakes.storage import Storage


class _FakeMemorize:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []  # (session_id, first message text)

    async def __call__(
        self,
        payload: dict[str, Any],
        *,
        is_final: bool = False,
        defer_extraction: bool = False,
    ) -> MemorizeResult:
        text = payload["messages"][0]["content"] if payload["messages"] else ""
        self.calls.append((payload["session_id"], text))
        return MemorizeResult(
            message_count=len(payload["messages"]), status="extracted"
        )


@pytest.fixture
async def setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> AsyncIterator[tuple[AsyncClient, mq.MemorizeQueueWorker, _FakeMemorize]]:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_MEMORIZE__ASYNC_QUEUE__RETRY_BACKOFF_SECONDS", "0")
    load_settings.cache_clear()
    await storage.create_schema()

    fake = _FakeMemorize()
    # Set on the class before the application is built, so both the worker
    # (which binds ``_memorize`` when it is built) and the inline route path
    # reach the fake.
    monkeypatch.setattr(MemoryService, "_memorize", fake)

    application = build_application(load_settings())
    worker = application.memory._queue
    app = create_app(application=application)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, worker, fake
    load_settings.cache_clear()


def _add(session_id: str, text: str, *, async_mode: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {
        "session_id": session_id,
        "messages": [
            {
                "sender_id": "alice",
                "role": "user",
                "timestamp": 1_700_000_000_000,
                "content": text,
            }
        ],
    }
    if async_mode:
        body["async_mode"] = True
    return body


async def test_sync_add_queues_behind_async_batch_and_runs_after_it(
    setup: tuple[AsyncClient, mq.MemorizeQueueWorker, _FakeMemorize], storage: Storage
) -> None:
    client, worker, fake = setup

    first = await client.post(
        "/api/v2/memory/add", json=_add("s1", "first", async_mode=True)
    )
    second = await client.post("/api/v2/memory/add", json=_add("s1", "second"))
    other = await client.post("/api/v2/memory/add", json=_add("s2", "other"))

    assert (first.status_code, second.status_code, other.status_code) == (
        202,
        202,
        200,
    )
    assert second.json()["data"]["pending"] == 2
    assert fake.calls == [("s2", "other")]  # only the backlog-free session ran inline

    delivered = await worker.drain_once()

    assert delivered == 2
    assert fake.calls == [("s2", "other"), ("s1", "first"), ("s1", "second")]
    statuses = {
        r.request_id: r.status for r in await storage.sqlite.memorize_inbox.list_all()
    }
    assert statuses == {
        first.json()["request_id"]: "success",
        second.json()["request_id"]: "success",
    }
    assert (
        await worker.has_pending(session_id="s1") is False
    )  # a later sync /add on s1 runs inline again
