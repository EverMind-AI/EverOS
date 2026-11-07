"""``/add`` and ``/flush`` say ``degraded``, not ``extracted``, when indexing fails."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from everos.entrypoints.api.routes.memorize import router
from everos.service import FlushResult, MemorizeResult
from tests.fakes import StubMemoryService, make_application


class _DegradedMemory(StubMemoryService):
    async def add(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        async_mode: bool,
        defer_extraction: bool,
    ) -> MemorizeResult:
        return MemorizeResult(
            message_count=len(payload["messages"]),
            status="degraded",
            failed_permanent=1,
            failed_retryable=0,
        )

    async def flush(self, **_kwargs: Any) -> FlushResult:
        return FlushResult(status="degraded", failed_permanent=1, failed_retryable=2)


def _client() -> AsyncClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.application = make_application(memory=_DegradedMemory())
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


_MESSAGE = {
    "sender_id": "default",
    "role": "user",
    "timestamp": 1_700_000_000_000,
    "content": "probe",
}


async def test_flush_does_not_claim_extracted_when_the_index_write_failed() -> None:
    async with _client() as client:
        resp = await client.post("/api/v1/memory/flush", json={"session_id": "probe"})
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status"] == "degraded"
    assert data["status"] != "extracted"
    assert data["failed_permanent"] == 1
    assert data["failed_retryable"] == 2


async def test_add_does_not_claim_extracted_when_the_index_write_failed() -> None:
    async with _client() as client:
        resp = await client.post(
            "/api/v1/memory/add",
            json={"session_id": "probe", "messages": [_MESSAGE]},
        )
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status"] == "degraded"
    assert data["failed_permanent"] == 1
    assert data["failed_retryable"] == 0
