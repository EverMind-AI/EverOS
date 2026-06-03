"""Belt-and-braces gate: dev-mode ``GET /openapi.json`` ≡ ``docs/openapi.json``.

The lint-time ``make check-openapi`` already diffs ``app.openapi()``
against the committed ``docs/openapi.json``. This e2e test closes the
remaining theoretical gap: if anyone ever adds a *lifespan-mutated*
OpenAPI schema (e.g. ``app.openapi_schema = ...`` inside a startup
handler), the in-memory ``app.openapi()`` and the runtime
``GET /openapi.json`` response would diverge — the lint gate would
miss it, but this test wouldn't.

How:

1. Force ``ENV=DEV`` so the ``openapi_url`` route is enabled.
2. Construct the app via ``create_app(lifespan_providers=[])`` to skip
   SQLite / LanceDB / OME (the schema is route-driven, not state-
   driven) — but *do* run the lifespan context, so any startup hook
   that mutates ``app.openapi_schema`` is exercised.
3. ``GET /openapi.json`` through ``httpx.AsyncClient``.
4. Diff against ``docs/openapi.json`` byte-for-byte (after JSON
   normalisation to defeat ordering nondeterminism).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMITTED_OPENAPI = _REPO_ROOT / "docs" / "openapi.json"


async def test_dev_mode_openapi_endpoint_matches_committed_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime ``GET /openapi.json`` (dev mode) must equal ``docs/openapi.json``."""
    # The gate's own committed snapshot must exist — otherwise the dev
    # workflow ``make openapi`` has been skipped.
    assert _COMMITTED_OPENAPI.is_file(), (
        f"{_COMMITTED_OPENAPI} not found — run `make openapi`"
    )

    # Force dev-mode so ``openapi_url="/openapi.json"`` is registered.
    monkeypatch.setenv("ENV", "DEV")

    from everos.entrypoints.api.app import create_app

    app = create_app(lifespan_providers=[])
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        resp = await client.get("/openapi.json")
    assert resp.status_code == 200, resp.text
    runtime_schema = resp.json()

    committed_schema = json.loads(_COMMITTED_OPENAPI.read_text(encoding="utf-8"))

    if runtime_schema != committed_schema:
        # Emit a concise diff to help locate the drift cause.
        import difflib

        runtime_rendered = json.dumps(runtime_schema, indent=2, ensure_ascii=False)
        committed_rendered = json.dumps(committed_schema, indent=2, ensure_ascii=False)
        diff = "\n".join(
            list(
                difflib.unified_diff(
                    committed_rendered.splitlines(),
                    runtime_rendered.splitlines(),
                    fromfile="docs/openapi.json (committed)",
                    tofile="GET /openapi.json (runtime)",
                    lineterm="",
                )
            )[:120]
        )
        raise AssertionError(
            "runtime /openapi.json drifts from docs/openapi.json; "
            "run `make openapi` and commit the result.\n\n" + diff
        )


# Keep ``os`` legit in case future scenarios need direct env reads.
_ = os
