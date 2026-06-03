"""``PrometheusMiddleware`` — increments counters / histograms, skips /metrics.

We isolate the test from the production registry by overriding it with a
fresh :class:`prometheus_client.CollectorRegistry` for the duration of
the test. The middleware was already imported with module-level Counter /
Histogram bound to whatever the registry was at import time — those
metric objects continue to record to the real registry. The test
therefore reads via ``_http_requests_total`` directly rather than via
``generate_metrics_response()``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from everos.core.middleware import prometheus as prom_mod


def _sample_value(metric: object, **labels: str) -> float:
    """Read the current value of a labeled prometheus metric (test helper)."""
    labeled = metric.labels(**labels)._labeled  # type: ignore[attr-defined]
    for sample in labeled.collect()[0].samples:
        if sample.name.endswith("_total"):
            return float(sample.value)
    return float("nan")


def _histogram_count(metric: object, **labels: str) -> float:
    labeled = metric.labels(**labels)._labeled  # type: ignore[attr-defined]
    for sample in labeled.collect()[0].samples:
        if sample.name.endswith("_count"):
            return float(sample.value)
    return float("nan")


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(prom_mod.PrometheusMiddleware)

    @app.get("/hello")
    async def hello() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/users/{user_id}")
    async def get_user(user_id: str) -> dict[str, str]:
        return {"user": user_id}

    return app


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = _build_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_increments_counter_on_200(client: AsyncClient) -> None:
    before = _sample_value(
        prom_mod._http_requests_total, method="GET", path="/hello", status="200"
    )
    resp = await client.get("/hello")
    assert resp.status_code == 200
    after = _sample_value(
        prom_mod._http_requests_total, method="GET", path="/hello", status="200"
    )
    assert after == before + 1


async def test_observes_duration_histogram(client: AsyncClient) -> None:
    before = _histogram_count(
        prom_mod._http_request_duration_seconds, method="GET", path="/hello"
    )
    await client.get("/hello")
    after = _histogram_count(
        prom_mod._http_request_duration_seconds, method="GET", path="/hello"
    )
    assert after == before + 1


def test_skip_paths_constant_contains_known_endpoints() -> None:
    """Skip set is the contract — assert membership directly to avoid

    polluting the global registry by ``.labels(path='/metrics')``-ing it
    (that creates a zero-valued sample which then leaks into the
    exposition format that test_metrics_route inspects).
    """
    assert "/metrics" in prom_mod._SKIP_PATHS
    assert "/health" in prom_mod._SKIP_PATHS
    assert "/healthz" in prom_mod._SKIP_PATHS
    assert "/favicon.ico" in prom_mod._SKIP_PATHS


async def test_path_params_normalized(client: AsyncClient) -> None:
    """``/users/abc`` should record against the route template ``/users/{user_id}``."""
    before = _sample_value(
        prom_mod._http_requests_total,
        method="GET",
        path="/users/{user_id}",
        status="200",
    )
    resp = await client.get("/users/abc")
    assert resp.status_code == 200
    after = _sample_value(
        prom_mod._http_requests_total,
        method="GET",
        path="/users/{user_id}",
        status="200",
    )
    assert after == before + 1


# ── _normalize_path direct tests (defensive fallback branches) ─────────


def test_normalize_path_uses_path_params_fallback() -> None:
    """When scope has no ``route`` but ``path_params`` is set, substitute names."""
    from types import SimpleNamespace

    from everos.core.middleware.prometheus import _normalize_path

    fake_req = SimpleNamespace(
        scope={},
        url=SimpleNamespace(path="/x/abc/y"),
        path_params={"id": "abc"},
    )
    # type: ignore[arg-type] — helper accepts anything duck-typed.
    assert _normalize_path(fake_req) == "/x/{id}/y"  # type: ignore[arg-type]


def test_normalize_path_unmatched_fallback() -> None:
    """No route, no path_params → ``{unmatched}`` sentinel."""
    from types import SimpleNamespace

    from everos.core.middleware.prometheus import _normalize_path

    fake_req = SimpleNamespace(
        scope={},
        url=SimpleNamespace(path="/x"),
        path_params={},
    )
    assert _normalize_path(fake_req) == "{unmatched}"  # type: ignore[arg-type]


def test_normalize_path_non_dict_scope_falls_through() -> None:
    """Defensive: a non-dict ``scope`` skips the route lookup entirely."""
    from types import SimpleNamespace

    from everos.core.middleware.prometheus import _normalize_path

    fake_req = SimpleNamespace(
        scope="not-a-dict",
        url=SimpleNamespace(path="/x"),
        path_params={},
    )
    assert _normalize_path(fake_req) == "{unmatched}"  # type: ignore[arg-type]
