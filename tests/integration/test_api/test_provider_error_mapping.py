"""Verify ProviderNotConfiguredError maps to HTTP 422 via the FastAPI handler.

Deviates from a naive flat `{"error_code": ..., "message": ...}` envelope: the
project already has a canonical error envelope (`ErrorResponse` /
`_error_response` in `exception_handlers.py`) shared by every domain
exception, so `ProviderNotConfiguredError` reuses it rather than inventing a
parallel shape. `ProviderNotConfiguredError` is a `ConfigurationError`
subclass; the assertions below also pin that the more-specific handler wins
over the parent `ConfigurationError` -> 500 mapping.

Uses `httpx.AsyncClient` + `ASGITransport` (not `fastapi.testclient.TestClient`)
to match the project's existing exception-handler test convention -- the
installed starlette/httpx pair deprecates `TestClient`.
"""

from __future__ import annotations

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from everos.core.errors import ProviderNotConfiguredError
from everos.entrypoints.api.exception_handlers import register_handlers


def _build_app_with_probes() -> FastAPI:
    app = FastAPI()
    register_handlers(app)

    @app.get("/probe_embed")
    async def probe_embed() -> None:
        raise ProviderNotConfiguredError(provider="embedding")

    @app.get("/probe_rerank_with_alt")
    async def probe_rerank() -> None:
        raise ProviderNotConfiguredError(
            provider="rerank",
            feature="agent_hybrid",
            alternative_hint="Set enable_llm_rerank=true.",
        )

    return app


async def _get(path: str) -> tuple[int, dict]:
    app = _build_app_with_probes()
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(path)
    return resp.status_code, resp.json()


async def test_maps_to_422():
    status_code, _ = await _get("/probe_embed")
    assert status_code == 422


async def test_error_code_field():
    _, body = await _get("/probe_embed")
    assert body["error"]["code"] == "PROVIDER_NOT_CONFIGURED"


async def test_message_field_carries_toml_path():
    _, body = await _get("/probe_embed")
    assert "everos.toml" in body["error"]["message"]


async def test_alternative_hint_present_in_message():
    _, body = await _get("/probe_rerank_with_alt")
    assert "Alternative:" in body["error"]["message"]
    assert "enable_llm_rerank=true" in body["error"]["message"]
