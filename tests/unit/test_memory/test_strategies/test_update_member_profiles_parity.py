"""``per_member`` mode sends what EverOS sent until 2026-11-14, request for request.

``tests/fixtures/member_profiles_per_member_ea7d2570.json`` holds every model
request and every written member document of one scripted group, recorded
with ea7d2570 -- the commit before ``[groups] member_profile_mode`` existed --
through the same scenario (scratch tool ``per_member_scenario.py``: three
members and an assistant, five segments, a stored profile with a frozen
language, ``member_profile_every = 2``, then a flush). With
``member_profile_mode = "per_member"`` and ``profile_on_flush = true`` this
tree must reproduce it exactly: the same requests in the same order, the
same documents.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.memory.events import EpisodeExtracted, GroupFlushed
from everos.memory.models import ExplicitInfoItem, Group, Profile, RosterEvent
from everos.memory.ports import Scope
from everos.memory.strategies import update_member_profiles
from tests.fakes import make_deps

_GOLDEN = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "member_profiles_per_member_ea7d2570.json"
)
_SCOPE = Scope(app_id="default", project_id="p", owner_id="trip", owner_type="group")
_T0 = 1_772_323_200_000
_SEGMENTS = [
    (
        "mc_1",
        [
            ("u_chen", "I moved to Lisbon last month."),
            ("u_betty", "I'm vegetarian, pick a place with options."),
        ],
    ),
    ("mc_2", [("u_chen", "Let's book Friday; I always want refundable tickets.")]),
    (
        "mc_3",
        [("u_ann", "I can only travel on weekends."), ("bot", "Noted, weekends only.")],
    ),
    (
        "mc_4",
        [("u_betty", "I'll handle the hotel."), ("u_chen", "Great, thanks Betty.")],
    ),
    ("mc_5", [("u_ann", "Count me in for Saturday.")]),
]


def _handler(messages: Any, *, response_format: type, **_: Any) -> str:
    name = response_format.__name__
    if name == "_ProfileUpdateReply":
        item = {
            "evidence": "2026-03-01: 'x'",
            "description": "A fact",
            "category": "plans",
        }
        op = {
            "action": "add",
            "type": "explicit_info",
            "data": item,
            "sources": ["s1"],
            "pictures": [],
        }
        return json.dumps({"operations": [op]})
    if name == "_ConsolidateReply":
        return json.dumps({"changes": [], "dropped": []})
    raise AssertionError(name)


def _plain(message: Any) -> dict[str, Any]:
    content = (
        message.content
        if isinstance(message.content, str)
        else [getattr(p, "text", type(p).__name__) for p in message.content]
    )
    return {"role": message.role, "content": content}


async def _scenario() -> dict[str, Any]:
    llm = FakeLLMClient(handler=_handler)
    deps = make_deps(llm=llm)
    key = ("default", "p", "trip")
    deps.stores.groups.groups[key] = Group(group_id="trip", created_at=0)
    deps.stores.groups.events[key].extend(
        [
            RosterEvent(op="join", member_id=m, at=0, display_name=n)
            for m, n in (("u_chen", "Chen"), ("u_betty", "Betty"), ("u_ann", "Ann"))
        ]
        + [
            RosterEvent(
                op="join", member_id="bot", at=0, display_name="Ava", kind="assistant"
            )
        ]
    )
    stored = Profile(
        owner_id="u_betty",
        summary="",
        timestamp=0,
        primary_language="English",
        explicit_info=[
            ExplicitInfoItem(
                evidence="2026-02-01: 'I live in Porto'",
                description="Lives in Porto",
                category="Home",
            )
        ],
    )
    await deps.stores.profile.write(_SCOPE, stored)
    for n, (memcell_id, lines) in enumerate(_SEGMENTS):
        at = _T0 + n * 600_000
        items = [
            ChatMessage(
                id=f"{memcell_id}_{i}",
                role="assistant" if s == "bot" else "user",
                sender_id=s,
                sender_name=s,
                content=text,
                timestamp=at + i * 1000,
            )
            for i, (s, text) in enumerate(lines)
        ]
        deps.stores.memcell.seed(
            memcell_id, MemCell(items=items, timestamp=at + len(items) * 1000)
        )
        speakers = list(dict.fromkeys(s for s, _ in lines if s != "bot"))
        event = EpisodeExtracted(
            memcell_id=memcell_id,
            episode_entry_id=f"gep_{memcell_id}",
            episode_text="t",
            episode_timestamp_ms=at,
            owner_id="trip",
            owner_type="group",
            app_id="default",
            project_id="p",
            speaker_ids=speakers,
            language="English",
        )
        await update_member_profiles(event, deps)
    flushed = GroupFlushed(
        group_id="trip", app_id="default", project_id="p", session_id="s"
    )
    await update_member_profiles(flushed, deps)
    requests = [
        {
            "messages": [_plain(m) for m in call.messages],
            "response_format": call.response_format.__name__,
        }
        for call in llm.calls
    ]
    docs: dict[str, Any] = {}
    for member in ("u_ann", "u_betty", "u_chen"):
        p = await deps.stores.profile.read(_SCOPE, member)
        docs[member] = (
            None
            if p is None
            else {
                "items": [
                    i.description for i in (*p.explicit_info, *p.implicit_traits)
                ],
                "sources": [
                    (i.model_extra or {}).get("source_memcell_ids")
                    for i in p.explicit_info
                ],
                "pending": p.pending_memcell_ids,
                "language": p.primary_language,
            }
        )
    return {"requests": requests, "documents": docs}


@pytest.mark.usefixtures("per_member_profiles")
async def test_per_member_mode_sends_the_requests_of_ea7d2570(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from everos.config import load_settings

    monkeypatch.setenv("EVEROS_GROUPS__MEMBER_PROFILE_EVERY", "2")
    load_settings.cache_clear()
    got = json.loads(json.dumps(await _scenario(), ensure_ascii=False, sort_keys=True))

    want = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    assert len(got["requests"]) == len(want["requests"]) == 4
    assert got == want
