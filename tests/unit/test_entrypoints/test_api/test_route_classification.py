"""Every API route is classified once as read or write for the purge gate.

The gate lets reads through and refuses writes while a purge runs; a route
in neither table would slip through as a read. The route set is taken from
the app itself (v1 and v2 alike, prefix removed).
"""

from __future__ import annotations

from everos.entrypoints.api.app import create_app
from everos.entrypoints.api.middleware.purge_gate import (
    READ_ROUTES,
    WRITE_ROUTES,
    classify,
)
from tests.fakes import make_application

_DOCS = {
    ("GET", "/docs"),
    ("GET", "/docs/oauth2-redirect"),
    ("GET", "/redoc"),
    ("GET", "/openapi.json"),
}


def _app_routes() -> set[tuple[str, str]]:
    app = create_app(application=make_application())
    routes: set[tuple[str, str]] = set()
    for path, operations in app.openapi()["paths"].items():
        bare = path.removeprefix("/api/v1").removeprefix("/api/v2")
        routes |= {(method.upper(), bare) for method in operations}
    return routes


def test_every_route_classified_once() -> None:
    routes = _app_routes()
    assert not READ_ROUTES & WRITE_ROUTES
    assert routes - (READ_ROUTES | WRITE_ROUTES) == set(), "unclassified route"
    assert (READ_ROUTES | WRITE_ROUTES) - routes - _DOCS == set(), "stale entry"


def test_both_versions_classify_alike() -> None:
    for method, path in READ_ROUTES | WRITE_ROUTES:
        concrete = path.replace("{doc_id}", "d_abcdef123456").replace(
            "{topic_id}", "d_abcdef123456_1"
        )
        concrete = concrete.replace("{task_id}", "t1").replace("{job_id}", "pj_1")
        expected = (method, path) in READ_ROUTES
        assert classify(method, concrete) is expected
        if path.startswith(("/memory", "/knowledge", "/tasks", "/ome", "/cascade")):
            assert classify(method, f"/api/v1{concrete}") is expected
            assert classify(method, f"/api/v2{concrete}") is expected
