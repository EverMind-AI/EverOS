"""``global_exception_handler`` — uniform error envelope per v1 API §1.

We mount the handler on a minimal FastAPI app with three error-emitting
routes (HTTPException 4xx / 5xx, RequestValidationError, raw exception)
and assert the envelope shape + status code each route produces.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from everos.core.middleware.global_exception import global_exception_handler


class _Body(BaseModel):
    name: str


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_exception_handler(HTTPException, global_exception_handler)
    app.add_exception_handler(RequestValidationError, global_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)

    @app.get("/raise-400")
    async def raise_400() -> None:
        raise HTTPException(status_code=400, detail="bad input")

    @app.get("/raise-500-http")
    async def raise_500_http() -> None:
        raise HTTPException(status_code=503, detail="upstream dead")

    @app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("hidden internals")

    @app.post("/validate")
    async def validate(_body: _Body) -> dict[str, str]:
        return {"ok": "yes"}

    return app


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = _build_app()
    # raise_app_exceptions=False — let the registered handler convert the
    # RuntimeError into a 500 response instead of re-raising into the test.
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _assert_envelope(body: dict[str, object], *, code: str, path: str) -> None:
    """Wiki §1 envelope: ``{request_id, error: {code, message, timestamp, path}}``."""
    assert isinstance(body["request_id"], str) and body["request_id"]
    error = body["error"]
    assert isinstance(error, dict)
    assert error["code"] == code
    assert isinstance(error["message"], str) and error["message"]
    assert isinstance(error["timestamp"], str) and "T" in error["timestamp"]
    assert error["path"] == path


async def test_http_exception_4xx(client: AsyncClient) -> None:
    resp = await client.get("/raise-400")
    assert resp.status_code == 400
    body = resp.json()
    _assert_envelope(body, code="HTTP_ERROR", path="/raise-400")
    assert body["error"]["message"] == "bad input"


async def test_http_exception_5xx_uses_system_error(client: AsyncClient) -> None:
    """5xx routed through HTTPException still produces SYSTEM_ERROR + generic msg."""
    resp = await client.get("/raise-500-http")
    assert resp.status_code == 503
    body = resp.json()
    _assert_envelope(body, code="SYSTEM_ERROR", path="/raise-500-http")
    # Internal detail "upstream dead" is suppressed in 5xx envelopes.
    assert body["error"]["message"] == "Internal server error"


async def test_unhandled_exception_5xx(client: AsyncClient) -> None:
    """RuntimeError → 500 with generic ``SYSTEM_ERROR`` envelope; details hidden."""
    resp = await client.get("/boom")
    assert resp.status_code == 500
    body = resp.json()
    _assert_envelope(body, code="SYSTEM_ERROR", path="/boom")
    assert body["error"]["message"] == "Internal server error"
    # Must not leak the internal exception message.
    assert "hidden internals" not in resp.text


async def test_validation_error_returns_422(client: AsyncClient) -> None:
    resp = await client.post("/validate", json={})  # missing ``name``
    assert resp.status_code == 422
    body = resp.json()
    _assert_envelope(body, code="HTTP_ERROR", path="/validate")
    # First-error message includes the offending field somewhere.
    assert "name" in body["error"]["message"].lower()
