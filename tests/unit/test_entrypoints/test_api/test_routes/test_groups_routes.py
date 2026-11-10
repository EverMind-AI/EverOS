"""The group routes over HTTP, and the group fields of the other routes.

A real memory service (``tests/helpers/groups.py``) behind the real app. What
these pin (docs/group-memory.md, sections 5.1-5.6 and 14 task 2):

- ``PUT /groups/{id}`` creates a group with the asker as owner; ``GET`` serves
  current members only (403 ``FORBIDDEN`` otherwise); ``POST .../roster``
  appends events (403 for a member changing someone else, 409 on a conflict);
- the identity header, when configured, wins and a body that disagrees is 403;
- group-scoped delete and forget answer 422;
- the OpenAPI document lists the new routes and fields.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from tests.fakes import make_application
from tests.helpers.groups import GroupWorld

_T0 = 1_772_323_200_000


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[AsyncClient]:
    async with GroupWorld.open(tmp_path) as world:
        app = create_app(application=make_application(memory=world.memory))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            yield c


def _put(asker: str | None = "u_chen", **fields: Any) -> dict[str, Any]:
    body = {
        "project_id": "p",
        "name": "Trip team",
        "created_at": _T0,
        "roster": [
            {"op": "join", "member_id": "u_betty", "display_name": "Betty", "at": _T0}
        ],
        **fields,
    }
    if asker is not None:
        body["asker_id"] = asker
    return body


async def test_create_read_and_change_a_group(client: AsyncClient) -> None:
    created = await client.put("/api/v2/groups/trip", json=_put())
    assert created.status_code == 200, created.text
    data = created.json()["data"]
    assert data["created"] is True
    roles = {m["member_id"]: m["role"] for m in data["group"]["roster"]}
    assert roles == {"u_chen": "owner", "u_betty": "member"}

    read = await client.get(
        "/api/v2/groups/trip", params={"project_id": "p", "asker_id": "u_betty"}
    )
    assert read.status_code == 200
    assert read.json()["data"]["name"] == "Trip team"

    joined = await client.post(
        "/api/v2/groups/trip/roster",
        json={
            "project_id": "p",
            "asker_id": "u_chen",
            "events": [
                {
                    "op": "join",
                    "member_id": "u_evan",
                    "title": "Designer",
                    "at": _T0 + 5,
                }
            ],
        },
    )
    assert joined.status_code == 200, joined.text
    assert joined.json()["data"]["events_applied"] == 1

    history = await client.get(
        "/api/v2/groups/trip",
        params={"project_id": "p", "asker_id": "u_evan", "include_history": True},
    )
    assert {m["member_id"] for m in history.json()["data"]["roster"]} == {
        "u_chen",
        "u_betty",
        "u_evan",
    }


async def test_errors_are_403_409_and_422_as_specified(client: AsyncClient) -> None:
    await client.put("/api/v2/groups/trip", json=_put())

    stranger = await client.get(
        "/api/v2/groups/trip", params={"project_id": "p", "asker_id": "u_zoe"}
    )
    assert stranger.status_code == 403
    assert stranger.json()["error"]["code"] == "FORBIDDEN"

    not_manager = await client.post(
        "/api/v2/groups/trip/roster",
        json={
            "project_id": "p",
            "asker_id": "u_betty",
            "events": [{"op": "join", "member_id": "u_eve", "at": _T0 + 1}],
        },
    )
    assert not_manager.status_code == 403

    conflict = await client.post(
        "/api/v2/groups/trip/roster",
        json={
            "project_id": "p",
            "asker_id": "u_chen",
            "events": [{"op": "join", "member_id": "u_betty", "at": _T0 + 1}],
        },
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "CONFLICT"

    no_asker = await client.put("/api/v2/groups/other", json=_put(asker=None))
    assert no_asker.status_code == 422

    for path, body in (
        ("/api/v2/memory/delete", {"group_id": "trip", "session_id": "s"}),
        (
            "/api/v2/memory/forget",
            {"user_id": "u_chen", "group_id": "trip", "image_id": "ab" * 32},
        ),
        (
            "/api/v2/memory/forget/session",
            {"user_id": "u_chen", "group_id": "trip", "session_id": "s"},
        ),
    ):
        refused = await client.post(path, json=body)
        assert refused.status_code == 422, path


async def test_the_identity_header_wins_and_must_agree(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await client.put("/api/v2/groups/trip", json=_put())
    monkeypatch.setenv("EVEROS_ACCESS__IDENTITY_HEADER", "X-Asker")
    load_settings.cache_clear()

    by_header = await client.get(
        "/api/v2/groups/trip",
        params={"project_id": "p"},
        headers={"X-Asker": "u_betty"},
    )
    assert by_header.status_code == 200
    mismatch = await client.get(
        "/api/v2/groups/trip",
        params={"project_id": "p", "asker_id": "u_chen"},
        headers={"X-Asker": "u_betty"},
    )
    assert mismatch.status_code == 403
    personal = await client.post(
        "/api/v2/memory/search",
        json={"user_id": "u_chen", "query": "x", "method": "keyword"},
        headers={"X-Asker": "u_betty"},
    )
    assert personal.status_code == 403


def test_the_openapi_document_lists_the_group_surface() -> None:
    spec = create_app(application=make_application()).openapi()

    assert "/api/v2/groups/{group_id}" in spec["paths"]
    assert {"get", "put"} <= set(spec["paths"]["/api/v2/groups/{group_id}"])
    assert "post" in spec["paths"]["/api/v2/groups/{group_id}/roster"]
    add = spec["components"]["schemas"]["MemorizeAddRequest"]["properties"]
    assert {"group_id", "roster", "asker_id"} <= set(add)
    assert "message_id" in spec["components"]["schemas"]["MessageItemDTO"]["properties"]
    search = spec["components"]["schemas"]["SearchRequest"]["properties"]
    assert {"group_id", "group_ids", "asker_id", "audience"} <= set(search)
