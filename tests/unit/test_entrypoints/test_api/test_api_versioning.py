"""API version aliasing — every business route is served under v1 and v2.

The ``/api/v2`` prefix is the canonical, cloud-aligned name; ``/api/v1`` is a
legacy compatibility alias pointing to the *same* endpoint. These tests are
the completeness guard: they fail if any versioned route is exposed under one
prefix but not the other, or if the two prefixes ever diverge to different
handlers. Infrastructure endpoints (``/health``, ``/metrics``) are
deliberately unversioned and must NOT be mirrored.

Assertions run against ``app.openapi()["paths"]`` — the authoritative,
fully-resolved public surface — rather than ``app.routes`` (which FastAPI keeps
as lazy ``_IncludedRouter`` wrappers, so leaf paths are not directly readable).
Same-handler identity is proved via the operationId, which FastAPI derives from
the endpoint function name + path: twin routes must share an operationId that
differs only by the version segment.
"""

from __future__ import annotations

from everos.entrypoints.api.app import create_app

_V1 = "/api/v1/"
_V2 = "/api/v2/"


def _openapi_paths() -> dict[str, dict]:
    app = create_app(lifespan_providers=[])
    return app.openapi()["paths"]


def test_every_v1_route_has_identical_v2_twin() -> None:
    paths = _openapi_paths()
    v1 = {p: ops for p, ops in paths.items() if p.startswith(_V1)}
    assert v1, "expected at least one /api/v1 route"

    for path, ops in v1.items():
        twin = _V2 + path[len(_V1) :]
        assert twin in paths, f"{path} has no v2 twin at {twin}"
        assert set(paths[twin]) == set(ops), f"{twin} verbs differ from {path}"
        # Same handler: operationId differs only by the version token.
        for method, op in ops.items():
            v1_id = op["operationId"]
            v2_id = paths[twin][method]["operationId"]
            assert v1_id.replace("_v1_", "_v2_") == v2_id, (
                f"{twin} [{method}] resolves to a different handler: "
                f"{v1_id!r} vs {v2_id!r}"
            )


def test_every_v2_route_has_v1_twin() -> None:
    paths = _openapi_paths()
    v2 = {p for p in paths if p.startswith(_V2)}
    assert v2, "expected at least one /api/v2 route"

    for path in v2:
        twin = _V1 + path[len(_V2) :]
        assert twin in paths, f"{path} has no v1 twin at {twin}"


def test_infra_endpoints_are_not_versioned() -> None:
    paths = set(_openapi_paths())
    assert "/health" in paths
    assert "/metrics" in paths
    # No accidental versioned mirror of infra endpoints.
    assert "/api/v1/health" not in paths
    assert "/api/v2/health" not in paths
    assert "/api/v1/metrics" not in paths
    assert "/api/v2/metrics" not in paths
