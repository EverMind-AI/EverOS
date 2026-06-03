"""``GET /metrics`` — Prometheus exposition + middleware integration.

Verifies three contracts of the metrics path:

1. The route renders ``prometheus_client``-parseable exposition format.
2. The ``PrometheusMiddleware`` actually bumps the per-route counter
   on a real round-trip (verified via before/after delta to avoid
   coupling to the global registry's cross-test accumulation).
3. The ``_SKIP_PATHS`` set (``/metrics``, ``/health``) is honoured —
   those endpoints never appear in ``everos_http_requests_total``.

No lifespan / no LanceDB / no LLM needed — middleware lives at the ASGI
layer above any of that.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from prometheus_client.parser import text_string_to_metric_families

from everos.config import load_settings
from everos.entrypoints.api.app import create_app

# ``prometheus_client.parser`` strips the ``_total`` counter suffix from
# the *family* name but leaves *sample* names intact.
_REQUESTS_FAMILY = "everos_http_requests"
_REQUESTS_TOTAL = "everos_http_requests_total"


@pytest.fixture
async def client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[AsyncClient]:
    """FastAPI app with no lifespan; middleware stack is wired by ``create_app``."""
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    load_settings.cache_clear()

    app = create_app(lifespan_providers=[])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    load_settings.cache_clear()


# ── Helpers ────────────────────────────────────────────────────────────


def _counter_value(text: str, path: str, status: str) -> float:
    """Sum ``everos_http_requests_total`` samples matching path + status."""
    total = 0.0
    for fam in text_string_to_metric_families(text):
        if fam.name != _REQUESTS_FAMILY:
            continue
        for s in fam.samples:
            if s.name != _REQUESTS_TOTAL:
                continue
            if s.labels.get("path") == path and s.labels.get("status") == status:
                total += s.value
    return total


def _all_recorded_paths(text: str) -> set[str]:
    """Set of ``path`` label values present in ``everos_http_requests_total``."""
    paths: set[str] = set()
    for fam in text_string_to_metric_families(text):
        if fam.name != _REQUESTS_FAMILY:
            continue
        for s in fam.samples:
            if s.name == _REQUESTS_TOTAL:
                paths.add(s.labels.get("path", ""))
    return paths


# ── Tests ──────────────────────────────────────────────────────────────


async def test_metrics_endpoint_renders_prometheus_format(
    client: AsyncClient,
) -> None:
    """``GET /metrics`` returns parsable Prometheus exposition format."""
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")

    # Must parse cleanly + expose the request counter family.
    families = {f.name for f in text_string_to_metric_families(resp.text)}
    assert _REQUESTS_FAMILY in families


async def test_metrics_counter_increments_on_request(client: AsyncClient) -> None:
    """A real route hit bumps ``everos_http_requests_total`` for that label triple.

    Uses a 422 to avoid needing LanceDB — Pydantic rejects the empty
    body before the route handler runs, but the middleware still sees
    a completed request/response with ``status=422``.
    """
    before_resp = await client.get("/metrics")
    before = _counter_value(before_resp.text, "/api/v1/memory/get", "422")

    bad = await client.post("/api/v1/memory/get", json={})
    assert bad.status_code == 422

    after_resp = await client.get("/metrics")
    after = _counter_value(after_resp.text, "/api/v1/memory/get", "422")

    assert after - before == 1.0, f"counter not bumped: {before} → {after}"


async def test_metrics_skip_paths_not_recorded(client: AsyncClient) -> None:
    """``_SKIP_PATHS`` (``/metrics``, ``/health``) never appear in the counter."""
    # Hit both endpoints. If they were *not* skipped, they'd show up in
    # the next /metrics dump.
    await client.get("/health")
    await client.get("/metrics")

    resp = await client.get("/metrics")
    recorded = _all_recorded_paths(resp.text)
    assert "/metrics" not in recorded, recorded
    assert "/health" not in recorded, recorded
