"""Routes resolve the application and the request id through ``Depends``.

That is what lets a test swap either one with ``app.dependency_overrides``
without touching ``app.state``: the override must reach the route, and the
application the app was built with must not be called at all.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from everos.entrypoints.api.dependencies import application, extract_request_id
from everos.service import Application
from tests.fakes import make_application

_ADD_BODY: dict[str, Any] = {
    "session_id": "s1",
    "messages": [
        {
            "sender_id": "alice",
            "role": "user",
            "timestamp": 1_700_000_000_000,
            "content": "hi",
        }
    ],
}


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    yield create_app(application=make_application())
    load_settings.cache_clear()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


async def test_overridden_application_receives_the_call(
    app: FastAPI, client: AsyncClient
) -> None:
    installed: Application = app.state.application
    stand_in = make_application()
    app.dependency_overrides[application] = lambda: stand_in

    resp = await client.post("/api/v1/memory/add", json=_ADD_BODY)

    assert resp.status_code == 200
    assert len(stand_in.memory.calls) == 1
    assert installed.memory.calls == []


async def test_overridden_request_id_reaches_the_envelope(
    app: FastAPI, client: AsyncClient
) -> None:
    app.dependency_overrides[extract_request_id] = lambda: "rid-override"

    resp = await client.post("/api/v1/memory/add", json=_ADD_BODY)

    assert resp.status_code == 200
    assert resp.json()["request_id"] == "rid-override"
