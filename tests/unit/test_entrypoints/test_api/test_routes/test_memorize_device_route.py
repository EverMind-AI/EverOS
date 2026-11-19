"""``POST /api/v2/memory/device/add``: a device batch reaches ``/add`` per person.

Pinned against the stub ``MemoryService``: what the pipeline receives is the
``/add`` payload with one sender per person (``<device_id>@<slot>``), the
device id for unidentified utterances, and nothing from a speaker who did not
consent.
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
from tests.fakes import StubMemoryService, make_application

_TS = 1_700_000_000_000


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    application = create_app(application=make_application())
    yield application
    load_settings.cache_clear()


@pytest.fixture
def memory(app: FastAPI) -> StubMemoryService:
    return app.state.application.memory


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c


def _says(slot: str | None, text: str, **speaker: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"text": text, "timestamp": _TS}
    if slot is not None:
        row["speaker"] = {"slot": slot, **speaker}
    return row


async def test_each_speaker_reaches_add_as_their_own_sender(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    resp = await client.post(
        "/api/v2/memory/device/add",
        json={
            "device_id": "xiaoai01",
            "session_id": "s_kitchen",
            "utterances": [
                _says("spk_a", "remind me to buy milk", display_name="Zhang San"),
                _says("spk_b", "and bread for me"),
                _says("spk_c", "do not keep this", consented=False),
                _says(None, "who left this here"),
            ],
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["data"] == {
        "message_count": 3,
        "status": "extracted",
        "owners": ["xiaoai01@spk_a", "xiaoai01@spk_b"],
    }
    ((payload, _, _),) = memory.calls
    assert payload["session_id"] == "s_kitchen"
    # The slot rides as a fallback label, so a name stored earlier still wins.
    assert [
        (m["sender_id"], m["sender_name"], m.get("sender_name_fallback"), m["role"])
        for m in payload["messages"]
    ] == [
        ("xiaoai01@spk_a", "Zhang San", "spk_a", "user"),
        ("xiaoai01@spk_b", None, "spk_b", "user"),
        ("xiaoai01", None, None, "user"),
    ]


async def test_a_batch_nobody_consented_to_writes_nothing(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    resp = await client.post(
        "/api/v2/memory/device/add",
        json={
            "device_id": "xiaoai01",
            "session_id": "s_kitchen",
            "utterances": [_says("spk_c", "do not keep this", consented=False)],
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["data"] == {
        "message_count": 0,
        "status": "accumulated",
        "owners": [],
    }
    assert memory.calls == []


@pytest.mark.parametrize(
    ("device_id", "slot"),
    [("xiaoai@01", "spk_a"), ("xiaoai01", "spk@a"), ("d" * 120, "s" * 20)],
)
async def test_an_owner_that_would_be_ambiguous_or_too_long_is_a_422(
    client: AsyncClient, memory: StubMemoryService, device_id: str, slot: str
) -> None:
    resp = await client.post(
        "/api/v2/memory/device/add",
        json={
            "device_id": device_id,
            "session_id": "s",
            "utterances": [_says(slot, "hi")],
        },
    )

    assert resp.status_code == 422
    assert memory.calls == []


async def test_async_mode_queues_the_batch_and_still_names_its_owners(
    client: AsyncClient, memory: StubMemoryService
) -> None:
    resp = await client.post(
        "/api/v2/memory/device/add",
        json={
            "device_id": "xiaoai01",
            "session_id": "s_kitchen",
            "async_mode": True,
            "utterances": [
                _says("spk_a", "remind me to buy milk"),
                _says(None, "who left this here"),
            ],
        },
    )

    assert resp.status_code == 202, resp.text
    data = resp.json()
    assert data["data"] == {
        "status": "queued",
        "message_count": 2,
        "pending": 1,
        "owners": ["xiaoai01@spk_a"],
    }
    assert memory.calls == []
    ((payload, request_id, _, _),) = memory.enqueued
    assert request_id == data["request_id"]
    assert [m["sender_id"] for m in payload["messages"]] == [
        "xiaoai01@spk_a",
        "xiaoai01",
    ]
