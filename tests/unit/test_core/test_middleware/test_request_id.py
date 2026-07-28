"""``RequestIdMiddleware`` — mints a W3C-compatible request id per request,
propagates it to the endpoint via both ``request.state`` and the
``core.context`` contextvar, and echoes it on the ``X-Request-Id``
response header.

The contextvar assertion is the load-bearing one: it proves the id set in
the middleware crosses Starlette's ``BaseHTTPMiddleware`` task boundary and
is visible to downstream handlers / loggers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from everos.core.context import get_request_id
from everos.core.middleware import RequestIdMiddleware


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/echo")
    async def echo(request: Request) -> dict[str, str | None]:
        return {
            "from_state": getattr(request.state, "request_id", None),
            "from_contextvar": get_request_id(),
        }

    return app


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = _build_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_sets_request_id_on_state_and_response_header(
    client: AsyncClient,
) -> None:
    resp = await client.get("/echo")
    assert resp.status_code == 200
    rid = resp.headers["x-request-id"]
    assert len(rid) == 32  # W3C trace-id shape (gen_request_id)
    assert resp.json()["from_state"] == rid


async def test_request_id_visible_to_endpoint_via_contextvar(
    client: AsyncClient,
) -> None:
    resp = await client.get("/echo")
    body = resp.json()
    assert body["from_contextvar"] is not None
    assert body["from_contextvar"] == resp.headers["x-request-id"]


async def test_each_request_gets_distinct_id(client: AsyncClient) -> None:
    r1 = await client.get("/echo")
    r2 = await client.get("/echo")
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


async def test_inbound_traceparent_continues_upstream_trace() -> None:
    """When the request carries a W3C traceparent header, our first span
    continues that upstream trace (distributed tracing). Absent → own root."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from everos.config.settings import ObservabilitySettings
    from everos.core.observability.tracing import (
        force_flush,
        init_tracing,
        memory_span,
        shutdown_tracing,
    )

    exporter = InMemorySpanExporter()
    shutdown_tracing()
    init_tracing(
        ObservabilitySettings(enabled=True, endpoint="http://collector.invalid"),
        span_processor=SimpleSpanProcessor(exporter),
    )
    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)

    @app.get("/s")
    async def s() -> dict[str, str]:
        with memory_span("everos.memory.search", observation_type="retriever"):
            pass
        return {"ok": "1"}

    upstream = "1234567890abcdef1234567890abcdef"
    tp = f"00-{upstream}-1111111111111111-01"
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            await c.get("/s", headers={"traceparent": tp})
        force_flush()
        span = exporter.get_finished_spans()[0]
        assert format(span.context.trace_id, "032x") == upstream
    finally:
        shutdown_tracing()
