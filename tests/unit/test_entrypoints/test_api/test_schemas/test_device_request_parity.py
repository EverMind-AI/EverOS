"""The hardware-assistant request surface, and the existing flows' parity.

The models in :mod:`everos.entrypoints.api.schemas.device` add a second,
device-first way to describe the same three things ``/memory/add``,
``/memory/search`` and ``/memory/get`` already take. These tests pin both
halves of the design:

* the device request shapes and how they project onto the existing ones
  (:meth:`DeviceAddRequest.speaker_owners` is the fan-out the pipeline does);
* **backwards compatibility** — the existing flows' request bodies are
  unchanged, byte for byte, because nothing here alters a field of
  :class:`~.memorize.MemorizeAddRequest`, :class:`~.memorize.MemorizeFlushRequest`,
  :class:`~.delete.DeleteRequest`, :class:`~.delete.DeleteByIdsRequest`,
  :class:`~everos.memory.get.dto.GetRequest` or
  :class:`~everos.memory.search.dto.SearchRequest`.
"""

from __future__ import annotations

import json

from everos.entrypoints.api.schemas import (
    DeleteByIdsRequest,
    DeleteRequest,
    DeviceAddRequest,
    MemorizeAddRequest,
    MemorizeFlushRequest,
)
from everos.memory.get.dto import GetRequest
from everos.memory.search.dto import SearchRequest

_TS = 1_700_000_000_000

_EXISTING_BODIES: dict[str, tuple[type, dict[str, object]]] = {
    "memorize.add": (
        MemorizeAddRequest,
        {
            "session_id": "s_kyoto",
            "messages": [
                {
                    "sender_id": "u_alice",
                    "sender_name": "Alice",
                    "role": "user",
                    "timestamp": _TS,
                    "content": "I climbed Yosemite last spring.",
                },
                {
                    "sender_id": "agent",
                    "role": "assistant",
                    "timestamp": _TS + 1000,
                    "content": "Noted.",
                },
            ],
        },
    ),
    "memorize.flush": (MemorizeFlushRequest, {"session_id": "s_kyoto"}),
    "delete.one": (
        DeleteRequest,
        {"user_id": "u_alice", "session_id": "s_kyoto"},
    ),
    "delete.by_ids": (
        DeleteByIdsRequest,
        {
            "memory_type": "episode",
            "memory_ids": ["u_alice_ep_20250514_00000001"],
        },
    ),
    "get.episode": (
        GetRequest,
        {"user_id": "u_alice", "memory_type": "episode", "page": 2},
    ),
    "search.hybrid": (
        SearchRequest,
        {"user_id": "u_alice", "query": "where did I climb", "method": "hybrid"},
    ),
}


def test_every_existing_request_body_round_trips_unchanged() -> None:
    """The frozen bodies validate, and re-serialize to the same values.

    Every value a client sent comes back unchanged: a field renamed or given a
    validator that rewrites it shows up here before it can reach a client.
    Fields the body left unset are defaults the server fills, so they appear
    on the way back — that is the wire contract, not a change to the request.
    """
    for name, (model, body) in _EXISTING_BODIES.items():
        parsed = model.model_validate(body)
        dumped = json.loads(parsed.model_dump_json())
        for key, value in body.items():
            if isinstance(value, list):
                assert len(dumped[key]) == len(value), f"{name}: {key} resized"
                for sent, got in zip(value, dumped[key], strict=True):
                    if isinstance(sent, dict):
                        assert set(sent) <= set(got), f"{name}: {key} lost a field"
                        for field, sent_value in sent.items():
                            assert got[field] == sent_value, (
                                f"{name}: {key}.{field} came back changed"
                            )
                    else:
                        assert got == sent, f"{name}: {key} came back changed"
            else:
                assert dumped[key] == value, f"{name}: {key} came back changed"


def test_the_device_request_projects_onto_the_existing_add_body() -> None:
    """A device batch is an add batch: one message per utterance, one sender
    per person, the device as the scope of everything unattributed."""
    request = DeviceAddRequest.model_validate(
        {
            "device_id": "dev_kitchen",
            "session_id": "s_kitchen",
            "utterances": [
                {
                    "speaker": {"slot": "spk_a", "display_name": "Alice"},
                    "text": "we need milk",
                    "timestamp": _TS,
                },
                {"text": "unintelligible", "timestamp": _TS + 1000},
            ],
        }
    )

    messages = request.projected_messages()

    assert [m["sender_id"] for m in messages] == [
        "dev_kitchen@spk_a",
        "dev_kitchen",
    ]
    assert [m["sender_name"] for m in messages] == ["Alice", None]
    assert [m["content"] for m in messages] == ["we need milk", "unintelligible"]
    assert request.speaker_owners() == ["dev_kitchen@spk_a"]


def test_the_projection_is_a_valid_add_body_and_owner() -> None:
    """``/memory/device/add`` hands the projection to ``/memory/add``'s model,
    and the person is then read and deleted by the same id on the existing
    routes — so the composed owner must pass every one of them unchanged."""
    request = DeviceAddRequest.model_validate(
        {
            "device_id": "dev_kitchen",
            "session_id": "s_kitchen",
            "utterances": [
                {"speaker": {"slot": "spk_a"}, "text": "milk", "timestamp": _TS}
            ],
        }
    )
    add = MemorizeAddRequest.model_validate(
        {"session_id": "s_kitchen", "messages": request.projected_messages()}
    )
    assert [m.sender_id for m in add.messages] == ["dev_kitchen@spk_a"]
    assert DeleteRequest(user_id="dev_kitchen@spk_a").user_id == "dev_kitchen@spk_a"
    search = SearchRequest.model_validate(
        {"user_id": "dev_kitchen@spk_a", "query": "milk", "method": "keyword"}
    )
    assert search.owner_id == "dev_kitchen@spk_a"


def test_the_device_scope_carries_the_device_id_not_a_person() -> None:
    """Searching the device is searching ``user_id = device_id``."""
    request = DeviceAddRequest.model_validate(
        {
            "device_id": "dev_kitchen",
            "session_id": "s_kitchen",
            "utterances": [{"text": "turn the lights off", "timestamp": _TS}],
        }
    )
    search = SearchRequest.model_validate(
        {"user_id": request.device_id, "query": "lights", "method": "keyword"}
    )
    assert search.owner_id == "dev_kitchen"
    assert search.owner_type == "user"
