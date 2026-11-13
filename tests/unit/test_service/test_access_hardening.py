"""Access hardening: who may delete, join, manage and read (review of 2026-11-13).

Real markdown, ``system.db`` and LanceDB behind the bootstrap's memory service
(``tests/helpers/groups.py``), and the real app over HTTP where the wire shape
matters. What these pin (docs/group-memory.md, sections 4.3, 5.1, 5.2, 9.1 and
15; docs/api.md, "Identity"):

- ``/memory/delete`` and ``/delete_by_ids`` take the asker as every other
  personal route does: a personal delete reaches the asker's own memory only
  (403 for someone else's, before anything is forgotten); a group delete is
  an owner's, an admin's or the deliverer's.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from httpx import ASGITransport, AsyncClient

from everos.config import load_settings
from everos.entrypoints.api.app import create_app
from everos.memory.models import Episode
from everos.memory.ports import NoSourceCheck
from everos.service import GetRequest
from tests.fakes import make_application
from tests.helpers.groups import APP, PROJECT, GroupWorld

T0 = 1_772_323_200_000
DAY = 86_400_000
BOT = "gw_trip"
HEADER = "X-Asker"


def _model(messages: Any, *, response_format: type, **_kw: Any) -> str:
    """Boundary never cuts (a flush closes the segment); episodes repeat the
    words they were shown; every other update writes nothing."""
    name = response_format.__name__
    prompt = "\n".join(_text(getattr(m, "content", "")) for m in messages)
    said = " ".join(w for w in prompt.split() if w.startswith("mk_")) or "nothing"
    if name == "_BatchReply":
        return json.dumps({"reasoning": "s", "boundaries": [], "tail_complete": False})
    if name == "EpisodeReply":
        return json.dumps({"title": "Plan", "summary": said, "content": said})
    if name == "ImageEpisodeReply":
        return json.dumps(
            {
                "title": "Plan",
                "summary": said,
                "content": said,
                "images": [{"ref": "img-1", "description": "mk_photo of a plan"}],
            }
        )
    if name == "_DecisionReply":
        return json.dumps({"core": list(range(10)), "next_queries": []})
    return json.dumps({"operations": []})


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        str(p.get("text") if isinstance(p, dict) else getattr(p, "text", "") or "")
        for p in content or []
    )


@pytest.fixture
async def world(tmp_path: Path) -> AsyncIterator[GroupWorld]:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_model)) as w:
        yield w


@pytest.fixture
async def client(world: GroupWorld) -> AsyncIterator[AsyncClient]:
    app = create_app(
        application=make_application(memory=world.memory, images=world.stores.images)
    )
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        yield c


def _gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """Askers come from the gateway's header from now on."""
    monkeypatch.setenv("EVEROS_ACCESS__IDENTITY_HEADER", HEADER)
    load_settings.cache_clear()


def _as(user: str) -> dict[str, str]:
    return {HEADER: user}


async def _diary(
    world: GroupWorld, user: str, text: str, *, session: str, at: int
) -> str:
    """One episode of ``user``'s own memory, from ``session``; its memory id."""
    ref = await world.stores.episode.append(
        world.personal(user),
        Episode(
            owner_id=user,
            episode=text,
            summary=text,
            subject=text,
            timestamp=at,
            session_id=session,
            sender_ids=[user],
            parent_id=f"mc_{user}_{at}",
        ),
        only_if=NoSourceCheck(),
    )
    assert ref is not None
    return ref.id


async def _episodes(world: GroupWorld, user: str) -> list[str]:
    """What ``user``'s default listing still holds."""
    await world.project()
    listed = await world.memory.get(
        GetRequest(
            app_id=APP,
            project_id=PROJECT,
            user_id=user,
            memory_type="episode",
            page_size=100,
        )
    )
    return sorted(e.episode for e in listed.data.episodes)


async def _two_diaries(world: GroupWorld) -> dict[str, str]:
    """Amy and Bob each hold a copy of one conversation, and a diary entry."""
    ids = {
        "amy_pair": await _diary(world, "amy", "amy pair", session="s_pair", at=T0),
        "bob_pair": await _diary(world, "bob", "bob pair", session="s_pair", at=T0),
        "amy_own": await _diary(world, "amy", "amy diary", session="s_amy", at=T0 + 1),
    }
    await world.project()
    return ids


# -- C-1: the delete aliases take the asker ---------------------------------


async def test_a_personal_delete_is_the_owners_alone(
    world: GroupWorld, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = await _two_diaries(world)
    _gateway(monkeypatch)
    for path, request in (
        ("/api/v2/memory/delete", {"project_id": PROJECT, "user_id": "amy"}),
        (
            "/api/v2/memory/delete",
            {"project_id": PROJECT, "user_id": "amy", "session_id": "s_pair"},
        ),
        (
            "/api/v2/memory/delete_by_ids",
            {
                "project_id": PROJECT,
                "memory_type": "episode",
                "memory_ids": [ids["bob_pair"], ids["amy_own"]],
            },
        ),
    ):
        answer = await client.post(path, json=request, headers=_as("bob"))
        assert answer.status_code == 403, (path, answer.text)
    assert await _episodes(world, "amy") == ["amy diary", "amy pair"]
    assert await _episodes(world, "bob") == ["bob pair"]

    # The owner deletes in their own memory: by id, then all of it.
    by_ids = await client.post(
        "/api/v2/memory/delete_by_ids",
        json={
            "project_id": PROJECT,
            "memory_type": "episode",
            "memory_ids": [ids["amy_own"]],
        },
        headers=_as("amy"),
    )
    assert by_ids.status_code == 200, by_ids.text
    assert await _episodes(world, "amy") == ["amy pair"]
    whole = await client.post(
        "/api/v2/memory/delete",
        json={"project_id": PROJECT, "user_id": "amy"},
        headers=_as("amy"),
    )
    assert whole.status_code == 200, whole.text
    assert await _episodes(world, "amy") == []
    assert await _episodes(world, "bob") == ["bob pair"]


async def test_a_session_delete_with_an_asker_forgets_the_askers_copy(
    world: GroupWorld, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _two_diaries(world)
    _gateway(monkeypatch)
    answer = await client.post(
        "/api/v2/memory/delete",
        json={"project_id": PROJECT, "session_id": "s_pair"},
        headers=_as("bob"),
    )
    assert answer.status_code == 200, answer.text
    assert len(answer.json()["data"]["forget_ids"]) == 1
    assert await _episodes(world, "bob") == []
    assert await _episodes(world, "amy") == ["amy diary", "amy pair"]


async def test_a_session_delete_without_an_asker_reaches_every_copy(
    world: GroupWorld,
) -> None:
    """A trusted caller (no gateway, no asker) deletes as before."""
    await _two_diaries(world)
    result = await world.memory.delete(
        app_id=APP, project_id=PROJECT, user_id=None, session_id="s_pair"
    )
    assert len(result.forget_ids) == 2
    assert await _episodes(world, "amy") == ["amy diary"]
    assert await _episodes(world, "bob") == []


async def _managed_group(
    world: GroupWorld, *, deliverer: str | None = BOT, owner: str = "u_chen"
) -> None:
    """``trip``: an owner and Betty from the start; Evan from ``T0 + 5000``."""
    await world.memory.upsert_group(
        app_id=APP,
        project_id=PROJECT,
        group_id="trip",
        asker_id=owner,
        asserted_asker=None,
        name="Trip",
        description=None,
        created_at=T0,
        history_visible_from=None,
        extra=None,
        roster=[
            world.roster_input(op="join", member_id=owner, role="owner", at=T0),
            world.roster_input(op="join", member_id="u_betty", at=T0),
            world.roster_input(op="join", member_id="u_evan", at=T0 + 5000),
        ],
        deliverer_id=deliverer,
    )


async def test_a_group_delete_is_a_managers_or_the_deliverers(
    world: GroupWorld, client: AsyncClient
) -> None:
    await _managed_group(world)
    await world.episode("trip", "mk_early plan", start=T0 + 1000, end=T0 + 2000)
    await world.stores.groups.bind(world.session("s_group"), "trip", at=T0)
    await world.project()
    body = {"project_id": PROJECT, "group_id": "trip", "session_id": "s_group"}
    for asker, status in (("u_betty", 403), ("u_zoe", 403), (BOT, 200)):
        answer = await client.post(
            "/api/v2/memory/delete", json={**body, "asker_id": asker}
        )
        assert answer.status_code == status, (asker, answer.text)
