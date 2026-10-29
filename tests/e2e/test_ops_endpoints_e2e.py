"""The operator endpoints, end to end over a real started application.

Pins the parity rows the route-level unit tests could not: ``GET /tasks``
(and ``/tasks/{task_id}``) reports an ``async_mode`` request through to its
delivery, ``POST /cascade/quiesce`` drains and stops a running cascade (and
answers 503 when there is none), and ``GET /metrics`` serves the Prometheus
exposition. Hermetic: the extraction model is canned, no embedding or rerank
is configured.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from tests.fakes import make_providers, user_track_llm

_TS = 1_775_116_800_000  # 2026-04-02 08:00 UTC


@asynccontextmanager
async def _client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[httpx.AsyncClient]:
    """A started application over the test's memory root, with a canned LLM."""
    from everos.config import load_settings
    from everos.entrypoints.api.app import create_app
    from everos.service import build_application

    monkeypatch.setenv("EVEROS_MEMORIZE__MODE", "chat")
    load_settings.cache_clear()
    settings = load_settings()
    providers = make_providers(llm=user_track_llm(), embedding=None, rerank=None)
    app = create_app(application=build_application(settings, providers=providers))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        yield client


async def test_an_async_add_is_reported_by_the_task_endpoints(
    core_pipeline_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _client(monkeypatch) as client:
        add = await client.post(
            "/api/v2/memory/add",
            json={
                "session_id": "s_tasks",
                "async_mode": True,
                "messages": [
                    {
                        "sender_id": "u_alice",
                        "role": "user",
                        "timestamp": _TS,
                        "content": "I am planning a trip to Kyoto.",
                    }
                ],
            },
        )
        assert add.status_code == 202, add.text
        task_id = add.json()["request_id"]
        assert add.json()["data"]["status"] == "queued"

        async with asyncio.timeout(30):
            while True:
                task = await client.get(f"/api/v2/tasks/{task_id}")
                assert task.status_code == 200, task.text
                if task.json()["data"]["status"] in ("success", "failed"):
                    break
                await asyncio.sleep(0.1)
        assert task.json()["data"]["status"] == "success", task.text
        assert task.json()["data"]["task_type"] == "memory_add"

        listed = await client.get(
            "/api/v1/tasks", params={"session_id": "s_tasks", "page_size": 500}
        )
        assert listed.status_code == 200, listed.text
        page = listed.json()["data"]
        assert [item["id"] for item in page["items"]] == [task_id]
        assert page["page_size"] == 100

        missing = await client.get("/api/v2/tasks/does-not-exist")
        assert missing.status_code == 404
        bad_status = await client.get("/api/v2/tasks", params={"status": "done"})
        assert bad_status.status_code == 422
        assert bad_status.json()["error"]["code"] == "INVALID_INPUT"


async def test_quiesce_drains_and_stops_a_running_cascade(
    core_pipeline_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _client(monkeypatch) as client:
        resp = await client.post("/api/v1/cascade/quiesce")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["quiesced"] is True
        assert body["pending_after"] == 0

        health = await client.get("/health")
        assert health.status_code == 200


async def test_quiesce_without_a_running_cascade_is_503(
    core_pipeline_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVEROS_DISABLE_CASCADE", "1")
    async with _client(monkeypatch) as client:
        resp = await client.post("/api/v2/cascade/quiesce")
    assert resp.status_code == 503


async def test_metrics_serves_the_prometheus_exposition(
    core_pipeline_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _client(monkeypatch) as client:
        await client.get("/health")
        resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "# HELP" in resp.text
