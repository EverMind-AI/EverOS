"""``async_mode`` on ``/add``, the backlog rule on ``/flush``, and ``GET /tasks``.

The route has one rule: a request is queued (202) when it asks for
``async_mode`` **or** when its session already has unfinished queued rows;
otherwise it runs ``memorize()`` inline exactly as before (200). These
tests pin that rule and the two response shapes against a stub worker
stashed where the lifespan would put it, with ``memorize`` faked out.

The sync 200 body must stay byte-for-byte what it was before ``async_mode``
existed: every existing caller depends on it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from everos.config import load_settings
from everos.core.errors import QueueSaturatedError
from everos.entrypoints.api.app import create_app
from everos.entrypoints.api.routes import memorize as route_mod
from everos.entrypoints.api.routes import tasks as tasks_mod
from everos.service.memorize import MemorizeResult
from everos.service.memorize_queue import (
    MemorizeQueueWorker,
    QueueHealth,
    TaskItem,
    TaskPage,
)


class _StubQueue(MemorizeQueueWorker):
    """Records enqueues; ``pending`` seeds per-session backlog."""

    def __init__(self, pending: dict[str, int] | None = None) -> None:
        self.pending = dict(pending or {})
        self.enqueued: list[tuple[dict[str, Any], str, bool, bool]] = []
        self.saturated = False
        self.accepting = True
        self.health_calls = 0

    async def has_pending(
        self,
        *,
        session_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> bool:
        return self.pending.get(session_id, 0) > 0

    async def health(self) -> QueueHealth:
        self.health_calls += 1
        return QueueHealth(
            healthy=True,
            reasons=[],
            loop_alive=True,
            accepting=self.accepting,
            pending=sum(self.pending.values()),
            oldest_pending_age_s=1.5 if self.pending else None,
            failed=0,
            loop_restarts=0,
            last_loop_error_at=None,
        )

    async def enqueue(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        is_final: bool,
        defer_extraction: bool,
    ) -> int:
        if self.saturated:
            raise QueueSaturatedError("memorize queue saturated for session s1")
        self.enqueued.append((payload, request_id, is_final, defer_extraction))
        n = self.pending.get(payload["session_id"], 0) + 1
        self.pending[payload["session_id"]] = n
        return n


class _FakeMemorize:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], bool, bool]] = []

    async def __call__(
        self,
        payload: dict[str, Any],
        *,
        is_final: bool = False,
        defer_extraction: bool = False,
    ) -> MemorizeResult:
        self.calls.append((payload, is_final, defer_extraction))
        return MemorizeResult(
            message_count=len(payload["messages"]), status="extracted"
        )


@pytest.fixture
def fake_memorize(monkeypatch: pytest.MonkeyPatch) -> _FakeMemorize:
    f = _FakeMemorize()
    monkeypatch.setattr(route_mod, "memorize", f)
    return f


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    application = create_app(lifespan_providers=[])
    yield application
    load_settings.cache_clear()


@pytest.fixture
def queue(app: FastAPI) -> _StubQueue:
    q = _StubQueue()
    app.state.lifespan_data = {"memorize_queue": q}
    return q


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


def _add_body(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "session_id": "s1",
        "messages": [
            {
                "sender_id": "alice",
                "role": "user",
                "timestamp": 1_700_000_000_000,
                "content": "hi",
            }
        ],
    }
    body.update(overrides)
    return body


# ── /add ───────────────────────────────────────────────────────────────────


async def test_sync_add_without_backlog_runs_inline_and_keeps_old_body(
    client: AsyncClient, queue: _StubQueue, fake_memorize: _FakeMemorize
) -> None:
    resp = await client.post("/api/v2/memory/add", json=_add_body())

    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"request_id", "data"}
    assert data["data"] == {"message_count": 1, "status": "extracted"}
    assert queue.enqueued == []
    ((payload, is_final, defer),) = fake_memorize.calls
    assert (is_final, defer) == (False, False)
    assert "async_mode" not in payload and "defer_extraction" not in payload


async def test_async_add_is_queued_and_answers_202(
    client: AsyncClient, queue: _StubQueue, fake_memorize: _FakeMemorize
) -> None:
    resp = await client.post("/api/v2/memory/add", json=_add_body(async_mode=True))

    assert resp.status_code == 202
    data = resp.json()
    assert data["data"] == {"status": "queued", "message_count": 1, "pending": 1}
    assert data["request_id"] == resp.headers["X-Request-Id"]
    assert fake_memorize.calls == []
    ((payload, request_id, is_final, defer),) = queue.enqueued
    assert request_id == data["request_id"]
    assert (is_final, defer) == (False, False)
    assert payload["session_id"] == "s1" and len(payload["messages"]) == 1
    assert "async_mode" not in payload and "defer_extraction" not in payload


async def test_sync_add_with_backlog_joins_the_queue(
    client: AsyncClient, queue: _StubQueue, fake_memorize: _FakeMemorize
) -> None:
    queue.pending["s1"] = 2

    resp = await client.post("/api/v2/memory/add", json=_add_body())

    assert resp.status_code == 202
    assert resp.json()["data"]["pending"] == 3
    assert fake_memorize.calls == []
    assert len(queue.enqueued) == 1


async def test_sync_add_on_another_session_is_not_blocked_by_s1_backlog(
    client: AsyncClient, queue: _StubQueue, fake_memorize: _FakeMemorize
) -> None:
    queue.pending["s1"] = 2

    resp = await client.post("/api/v2/memory/add", json=_add_body(session_id="s2"))

    assert resp.status_code == 200
    assert len(fake_memorize.calls) == 1


async def test_defer_extraction_rides_on_the_queued_row(
    client: AsyncClient, queue: _StubQueue, fake_memorize: _FakeMemorize
) -> None:
    resp = await client.post(
        "/api/v2/memory/add", json=_add_body(async_mode=True, defer_extraction=True)
    )

    assert resp.status_code == 202
    ((_, _, is_final, defer),) = queue.enqueued
    assert (is_final, defer) == (False, True)


async def test_async_mode_without_worker_runs_synchronously(
    client: AsyncClient, app: FastAPI, fake_memorize: _FakeMemorize
) -> None:
    app.state.lifespan_data = {}  # queue disabled / lifespan absent

    resp = await client.post("/api/v2/memory/add", json=_add_body(async_mode=True))

    assert resp.status_code == 200
    assert resp.json()["data"] == {"message_count": 1, "status": "extracted"}
    assert len(fake_memorize.calls) == 1


async def test_saturated_queue_maps_to_429(
    client: AsyncClient, queue: _StubQueue, fake_memorize: _FakeMemorize
) -> None:
    queue.saturated = True

    resp = await client.post("/api/v2/memory/add", json=_add_body(async_mode=True))

    assert resp.status_code == 429
    assert resp.json()["error"]["code"] == "QUEUE_SATURATED"


# ── /flush ─────────────────────────────────────────────────────────────────


async def test_sync_flush_without_backlog_keeps_old_body(
    client: AsyncClient, queue: _StubQueue, fake_memorize: _FakeMemorize
) -> None:
    resp = await client.post("/api/v2/memory/flush", json={"session_id": "s1"})

    assert resp.status_code == 200
    assert resp.json()["data"] == {"status": "extracted"}
    ((_, is_final, _),) = fake_memorize.calls
    assert is_final is True


async def test_sync_flush_with_backlog_joins_the_queue(
    client: AsyncClient, queue: _StubQueue, fake_memorize: _FakeMemorize
) -> None:
    queue.pending["s1"] = 1

    resp = await client.post("/api/v2/memory/flush", json={"session_id": "s1"})

    assert resp.status_code == 202
    assert fake_memorize.calls == []
    ((_, _, is_final, _),) = queue.enqueued
    assert is_final is True


# ── GET /tasks ─────────────────────────────────────────────────────────────


async def test_task_by_id(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    async def fake_get(task_id: str) -> TaskItem | None:
        seen.append(task_id)
        if task_id != "r1":
            return None
        return TaskItem(
            id="r1",
            status="processing",
            task_type="memory_add",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

    monkeypatch.setattr(tasks_mod, "get_task", fake_get)

    resp = await client.get("/api/v2/tasks/r1")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert (data["id"], data["status"], data["task_type"]) == (
        "r1",
        "processing",
        "memory_add",
    )
    assert data["finished_at"] is None
    assert data["created_at"].startswith("2026-01-01T00:00:00")

    missing = await client.get("/api/v2/tasks/zz")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "NOT_FOUND"
    assert seen == ["r1", "zz"]


async def test_task_list_passes_filters_and_caps_page_size(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict[str, Any]] = []

    async def fake_list(**kwargs: Any) -> TaskPage:
        seen.append(kwargs)
        return TaskPage(
            total=0, page=kwargs["page"], page_size=kwargs["page_size"], items=[]
        )

    monkeypatch.setattr(tasks_mod, "list_tasks", fake_list)

    resp = await client.get(
        "/api/v2/tasks",
        params={
            "session_id": "s1",
            "status": "queued",
            "page": 2,
            "page_size": 500,
            "app_id": "a",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["data"] == {"total": 0, "page": 2, "page_size": 100, "items": []}
    assert seen == [
        {
            "session_id": "s1",
            "status": "queued",
            "page": 2,
            "page_size": 100,
            "app_id": "a",
            "project_id": "default",
        }
    ]


async def test_task_list_rejects_unknown_status(client: AsyncClient) -> None:
    resp = await client.get("/api/v2/tasks", params={"status": "done"})

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "INVALID_INPUT"


# ── enabled = false: the door is closed, the backlog rule still holds ─────


async def test_async_mode_is_ignored_when_queue_is_not_accepting(
    client: AsyncClient, queue: _StubQueue, fake_memorize: _FakeMemorize
) -> None:
    queue.accepting = False

    resp = await client.post("/api/v2/memory/add", json=_add_body(async_mode=True))

    assert resp.status_code == 200
    assert resp.json()["data"] == {"message_count": 1, "status": "extracted"}
    assert queue.enqueued == []


async def test_backlog_still_queues_sync_calls_when_not_accepting(
    client: AsyncClient, queue: _StubQueue, fake_memorize: _FakeMemorize
) -> None:
    queue.accepting = False
    queue.pending["s1"] = 1

    resp = await client.post("/api/v2/memory/add", json=_add_body())

    assert resp.status_code == 202
    assert fake_memorize.calls == []


# ── GET /health carries the queue block ────────────────────────────────────


async def test_health_reports_the_memorize_queue_block(
    client: AsyncClient, queue: _StubQueue
) -> None:
    queue.pending = {"s1": 2}

    resp = await client.get("/health")

    assert resp.status_code == 200
    block = resp.json()["memorize_queue"]
    assert block["healthy"] is True and block["loop_alive"] is True
    assert (block["pending"], block["oldest_pending_age_s"], block["failed"]) == (
        2,
        1.5,
        0,
    )
    assert block["loop_restarts"] == 0 and block["last_loop_error_at"] is None
    assert queue.health_calls == 1


async def test_health_without_the_queue_lifespan_has_no_block(app: FastAPI) -> None:
    app.state.lifespan_data = {}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["memorize_queue"] is None
