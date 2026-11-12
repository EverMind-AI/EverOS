"""The leak suite: nothing an asker may not see reaches them, on any read path.

docs/group-memory.md, section 13.5 (and 14 task 8). Real markdown,
``system.db`` and LanceDB with group memory seeded the way the group write
path writes it (``tests/helpers/groups.py``). One roster holds every shape the
section names:

- ``u_chen`` -- in the group from the start (sees everything);
- ``u_evan`` -- a late joiner (``[5000, ...)``);
- ``u_dan``  -- left and rejoined (``[0, 2500)`` and ``[7500, ...)``);
- ``u_betty`` -- a member who left (``[0, 6000)``): no group read at all;
- ``u_zoe``  -- never a member.

and records placed around those times: episodes inside, outside and
straddling windows, a picture memory from before the late joiner, a later
episode that refers to it, buffered messages before and after the join, and a
personal episode and profile of ``u_chen``. Cases that need more seed their
own: member-in-group profile items on both sides of a window, a scene spanning
a member's absence, agent cases learned in this and another group, and the
group write path with its strategies.

Every case asserts the record is absent from search (the multi-round
retrieval's results and its decider's very input), get, attached pictures,
attached profiles, the session buffer and picture bytes. Zero leaks is the
pass mark. Read paths of section 9.3 that do not exist yet are listed in
:data:`NOT_YET` with the reason, so the table is accounted for row by row.

Two people talking without a ``group_id`` are an implicit group (section
7.9), read by its people only: a stranger reads nothing of it, and a third
person who joins later -- refused in the pair's session, a group of their own
with the pair in a session of the three -- reads nothing of the pair's.

A personal search with ``include_groups`` reads the user's groups the way a
group search by the user does, so it takes the same cases: a former member
and a stranger get nothing of the group (their own memory still answers), a
later joiner and a member who was away get nothing said outside their
windows, a forgotten record or message is absent, and another member's
records -- personal memory, a profile in the group -- never come along.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from httpx import ASGITransport, AsyncClient

from everos.core.errors import ForbiddenError, RosterConflictError
from everos.entrypoints.api.app import create_app
from everos.memory.access import AccessPolicy, resolve_identity
from everos.memory.models import (
    AgentCase,
    ExplicitInfoItem,
    Profile,
    ProfileSource,
    Scene,
)
from everos.memory.ports import NoSourceCheck, Scope, SessionKey, all_of, eq
from everos.memory.strategies import STRATEGY_SPECS
from everos.service import GetRequest, GroupForgetTarget, SearchRequest
from tests.fakes import make_application
from tests.helpers.groups import APP, PROJECT, GroupWorld, join, leave

NOT_YET: dict[int, str] = {
    6: "history route: not built yet (2.0 history); the group profile's versions"
    " (get include_superseded) are covered by"
    " test_group_profile_items_follow_the_windows",
    13: "MCP: a separate package (everos-mcp) forwarding the HTTP API",
}
"""Rows of section 9.3 this suite cannot exercise yet, and why."""

T0 = 1_772_323_200_000
"""Every time below is ``T0 + offset`` ms: message timestamps go through the
ingest's seconds-or-milliseconds detection, which needs real epoch values."""

# What each record says, so a leak is a string that shows up (offsets from T0).
ALPHA = "alpha marker plan Lisbon offsite"  # [1000, 2000]
BRAVO = "bravo marker plan Lisbon offsite"  # [3000, 4000]  Dan was away
STRADDLE = "straddle marker plan Lisbon offsite"  # [2000, 3000] spans Dan's leave
CHARLIE = "charlie marker plan Lisbon offsite"  # [8000, 9000]
REFERRER = "referrer marker plan Lisbon offsite floor plan"  # [8100, 8200]
PERSONAL = "personal marker plan Lisbon offsite of Chen"
PICTURE = "picture marker floor plan of the Lisbon venue"  # at 1500

GROUP_RECORDS = {
    ALPHA: (T0 + 1000, T0 + 2000),
    BRAVO: (T0 + 3000, T0 + 4000),
    STRADDLE: (T0 + 2000, T0 + 3000),
    CHARLIE: (T0 + 8000, T0 + 9000),
    REFERRER: (T0 + 8100, T0 + 8200),
}
WINDOWS = {
    "u_chen": [(T0, None)],
    "u_evan": [(T0 + 5000, None)],
    "u_dan": [(T0, T0 + 2500), (T0 + 7500, None)],
}


def _visible(asker: str, start: int, end: int) -> bool:
    return any(
        start >= lo and (hi is None or end < hi) for lo, hi in WINDOWS.get(asker, [])
    )


class _RecordingDecider:
    """Answers every decider call with "keep the first ten"; keeps the prompts."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, messages: Any, **_kwargs: Any) -> str:
        for message in messages:
            content = getattr(message, "content", "")
            self.prompts.append(
                content
                if isinstance(content, str)
                else json.dumps(
                    [p for p in content if isinstance(p, dict) and "text" in p]
                )
            )
        return json.dumps({"core": list(range(10)), "next_queries": []})


@dataclass
class _Fixture:
    world: GroupWorld
    decider: _RecordingDecider
    picture: str
    client: AsyncClient


@pytest.fixture
async def fixture(tmp_path: Path) -> AsyncIterator[_Fixture]:
    decider = _RecordingDecider()
    model = FakeLLMClient(handler=decider)
    async with GroupWorld.open(tmp_path, llm=model, decider=model) as world:
        await world.roster(
            "trip",
            [
                join("u_chen", T0, role="owner"),
                join("u_evan", T0 + 5000),
                join("u_dan", T0),
                leave("u_dan", T0 + 2500),
                join("u_dan", T0 + 7500),
                join("u_betty", T0),
                leave("u_betty", T0 + 6000),
            ],
        )
        for text, (start, end) in GROUP_RECORDS.items():
            if text != REFERRER:
                await world.episode("trip", text, start=start, end=end)
        picture = await world.picture(
            world.scope("trip"), PICTURE, at=T0 + 1500, shade=99
        )
        start, end = GROUP_RECORDS[REFERRER]
        await world.episode("trip", REFERRER, start=start, end=end, image_ids=[picture])
        await world.personal_episode("u_chen", PERSONAL, at=T0 + 1000)
        await world.profile("u_chen", "personal marker Chen lives in Porto")
        await world.stores.groups.bind(
            SessionKey(app_id=APP, project_id=PROJECT, session_id="s_group"),
            "trip",
            at=T0,
        )
        await world.memory.add(
            {
                "session_id": "s_group",
                "app_id": APP,
                "project_id": PROJECT,
                "messages": [
                    {"sender_id": "u_chen", "role": "user", "timestamp": T0 + 1000,
                     "content": "buffered marker early"},
                    {"sender_id": "u_chen", "role": "user", "timestamp": T0 + 8500,
                     "content": "buffered marker late"},
                ],
            },
            request_id="r",
            async_mode=False,
            defer_extraction=True,
        )  # fmt: skip
        await world.project()
        app = create_app(
            application=make_application(
                memory=world.memory, images=world.stores.images
            )
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t"
        ) as c:
            yield _Fixture(world=world, decider=decider, picture=picture, client=c)


def _search(**scope: Any) -> SearchRequest:
    return SearchRequest(
        app_id=APP,
        project_id=PROJECT,
        query="marker plan Lisbon offsite",
        top_k=20,
        filters={"session_id": "s_group"},
        include_profile=True,
        **scope,
    )


def _mixed(user: str, **fields: Any) -> SearchRequest:
    """A personal search of ``user`` that also covers the user's groups."""
    values: dict[str, Any] = {
        "app_id": APP,
        "project_id": PROJECT,
        "user_id": user,
        "query": "marker plan Lisbon offsite",
        "top_k": 20,
        "include_groups": True,
        "include_profile": True,
        **fields,
    }
    return SearchRequest(**values)


def _mixed_listing(user: str, **fields: Any) -> GetRequest:
    """A listing of ``user``'s episodes that also covers the user's groups."""
    values: dict[str, Any] = {
        "app_id": APP,
        "project_id": PROJECT,
        "user_id": user,
        "memory_type": "episode",
        "page_size": 100,
        "include_groups": True,
        **fields,
    }
    return GetRequest(**values)


def _everything_text(response: Any) -> str:
    """Every string a search response hands back, in one blob."""
    return json.dumps(response.data.model_dump(mode="json"))


@pytest.mark.parametrize("asker", ["u_chen", "u_evan", "u_dan"])
async def test_search_returns_nothing_outside_the_askers_windows(
    fixture: _Fixture, asker: str
) -> None:
    """Rows 1-4: every recall, the decider's input, attached pictures and
    profiles, and the session buffer."""
    fixture.decider.prompts.clear()
    response = await fixture.world.memory.search(
        _search(group_id="trip", asker_id=asker)
    )
    blob = _everything_text(response)
    seen_by_decider = "\n".join(fixture.decider.prompts)

    visible = [t for t, (s, e) in GROUP_RECORDS.items() if _visible(asker, s, e)]
    # The decider really ran, and saw the asker's own records.
    assert any(text in seen_by_decider for text in visible), asker
    assert any(text in blob for text in visible), asker
    for text, (start, end) in GROUP_RECORDS.items():
        if not _visible(asker, start, end):
            assert text not in blob, (asker, text)
            assert text not in seen_by_decider, (asker, text)
    # The picture (at 1500) and the early buffered message.
    if not _visible(asker, T0 + 1500, T0 + 1500):
        assert PICTURE not in blob and fixture.picture not in json.dumps(
            [i.image_id for i in response.data.images]
            + [i.image_id for e in response.data.episodes for i in e.images]
        )
        assert PICTURE not in seen_by_decider
    if not _visible(asker, T0 + 1000, T0 + 1000):
        assert "buffered marker early" not in blob
    # Personal memory never enters a group read.
    assert "personal marker" not in blob
    assert "personal marker" not in seen_by_decider
    assert response.data.profiles == []


@pytest.mark.parametrize("asker", ["u_betty", "u_zoe"])
async def test_former_and_non_members_read_nothing_of_the_group(
    fixture: _Fixture, asker: str
) -> None:
    """Rows 1, 5 and 10: a former member and a stranger are refused outright."""
    with pytest.raises(ForbiddenError):
        await fixture.world.memory.search(_search(group_id="trip", asker_id=asker))
    with pytest.raises(ForbiddenError):
        await fixture.world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id=asker,
                memory_type="episode",
            )
        )
    read = await fixture.client.get(
        "/api/v2/groups/trip", params={"project_id": PROJECT, "asker_id": asker}
    )
    assert read.status_code == 403


@pytest.mark.parametrize("asker", ["u_chen", "u_evan", "u_dan"])
async def test_get_lists_nothing_outside_the_askers_windows(
    fixture: _Fixture, asker: str
) -> None:
    """Row 5."""
    response = await fixture.world.memory.get(
        GetRequest(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id=asker,
            memory_type="episode",
            page_size=100,
        )
    )
    listed = {e.episode for e in response.data.episodes}
    expected = {t for t, (s, e) in GROUP_RECORDS.items() if _visible(asker, s, e)}
    assert listed == expected
    assert response.data.total_count == len(expected)


@pytest.mark.parametrize("reader", ["u_evan", "u_dan", "u_betty", "u_zoe"])
async def test_personal_memory_is_read_by_its_owner_only(
    fixture: _Fixture, reader: str
) -> None:
    """Row 4 and section 9.1: a personal record read by another user."""
    with pytest.raises(ForbiddenError):
        await fixture.world.memory.search(_search(user_id="u_chen", asker_id=reader))
    # Without an asker the read is the owner's own; another owner id sees
    # neither Chen's memory nor the group's buffer.
    other = await fixture.world.memory.search(_search(user_id=reader))
    blob = _everything_text(other)
    assert "personal marker" not in blob
    assert "buffered marker" not in blob
    assert not any(marker in blob for marker in GROUP_RECORDS)


async def test_a_group_buffer_never_reads_through_a_personal_scope(
    fixture: _Fixture,
) -> None:
    """Row 3: Chen sent those messages, but the session is the group's."""
    mine = await fixture.world.memory.search(_search(user_id="u_chen"))
    assert mine.data.unprocessed_messages == []


@pytest.mark.parametrize(
    ("asker", "status"),
    [("u_chen", 200), ("u_dan", 200), ("u_evan", 404), ("u_betty", 404),
     ("u_zoe", 404), (None, 404)],
)  # fmt: skip
async def test_picture_bytes_follow_the_windows(
    fixture: _Fixture, asker: str | None, status: int
) -> None:
    """Row 7: a hash probe for a group picture."""
    params = {"asker_id": asker} if asker else {}
    response = await fixture.client.get(
        f"/api/v2/memory/images/{fixture.picture}", params=params
    )
    assert response.status_code == status
    if status == 404:
        # The same answer as for a picture that does not exist: only the
        # request's own path and time differ.
        missing = await fixture.client.get(
            f"/api/v2/memory/images/{'ef' * 32}", params=params
        )
        error, absent = response.json()["error"], missing.json()["error"]
        assert (error["code"], error["message"]) == (absent["code"], absent["message"])


def _forget_body(asker: str | None, target: dict[str, Any], **extra: Any) -> Any:
    body = {"project_id": PROJECT, "group_id": "trip", "target": target, **extra}
    if asker is not None:
        body["asker_id"] = asker
    return body


async def test_group_forget_undo_revise_delete_are_authorised(
    fixture: _Fixture,
) -> None:
    """Row 8: authorise, then act on the group's own memory only."""
    client = fixture.client
    episode = await _record_id(fixture.world, ALPHA)
    record = {"kind": "record", "record_ids": [episode]}
    # A stranger and a former member are refused before anything is read; a
    # member may not forget what rests on another member's messages.
    for asker in ("u_zoe", "u_betty", "u_dan", None):
        refused = await client.post(
            "/api/v2/memory/forget", json=_forget_body(asker, record, dry_run=True)
        )
        assert refused.status_code == (422 if asker is None else 403), refused.text
    # A group delete is the owner's or an admin's; a plain delete of a group
    # session, a group revision and a group conversation forget stay refused.
    denied = await client.post(
        "/api/v2/memory/delete",
        json={
            "project_id": PROJECT,
            "group_id": "trip",
            "session_id": "s_group",
            "asker_id": "u_dan",
        },
    )
    assert denied.status_code == 403, denied.text
    for path, body in (
        ("/api/v2/memory/delete", {"project_id": PROJECT, "session_id": "s_group"}),
        (
            "/api/v2/memory/forget/session",
            {"user_id": "u_chen", "group_id": "trip", "session_id": "s_group"},
        ),
        (
            "/api/v2/memory/images/revise",
            {
                "user_id": "u_chen",
                "group_id": "trip",
                "image_id": fixture.picture,
                "description": "x",
            },
        ),
    ):
        refused = await client.post(path, json=body)
        assert refused.status_code == 422, (path, refused.text)
    # The owner forgets (a dry run unless told not to); the undo needs the
    # same authorisation.
    done = await client.post(
        "/api/v2/memory/forget", json=_forget_body("u_chen", record, dry_run=False)
    )
    assert done.status_code == 200, done.text
    forget_id = done.json()["data"]["forget_id"]
    undo = {"project_id": PROJECT, "group_id": "trip", "forget_id": forget_id}
    for asker, status in (("u_dan", 403), ("u_zoe", 403), ("u_chen", 200)):
        answer = await client.post(
            "/api/v2/memory/forget/undo", json={**undo, "asker_id": asker}
        )
        assert answer.status_code == status, (asker, answer.text)
    missing = await client.post(
        "/api/v2/memory/forget/undo",
        json={**undo, "asker_id": "u_chen", "forget_id": "fg_" + "0" * 16},
    )
    assert missing.status_code == 404


async def _record_id(world: GroupWorld, text: str) -> str:
    """The row id of the group episode that says ``text``."""
    response = await world.memory.get(
        GetRequest(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_chen",
            memory_type="episode",
            page_size=100,
        )
    )
    return next(e.id for e in response.data.episodes if e.episode == text)


async def _everything_read_by(fixture: _Fixture, asker: str) -> str:
    """Every read path's answer to ``asker``, in one blob."""
    world = fixture.world
    response = await world.memory.search(
        _search(group_id="trip", asker_id=asker).model_copy(update={"filters": None})
    )
    mixed = await world.memory.search(_mixed(asker))
    mixed_listing = await world.memory.get(_mixed_listing(asker))
    blobs = [
        _everything_text(response),
        _everything_text(mixed),
        json.dumps(mixed_listing.model_dump(mode="json")),
        "\n".join(fixture.decider.prompts),
    ]
    for memory_type, extra in (
        ("episode", {}),
        ("image_memory", {}),
        ("scene", {}),
        ("group_profile", {"include_superseded": True}),
        ("member_profile", {"member_id": "u_chen"}),
    ):
        listed = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id=asker,
                memory_type=memory_type,
                page_size=100,
                **extra,
            )
        )
        blobs.append(json.dumps(listed.model_dump(mode="json")))
    return "\n".join(blobs)


@pytest.mark.parametrize("kind", ["episode", "picture", "group_profile"])
async def test_a_forgotten_record_is_absent_for_every_asker(
    fixture: _Fixture, kind: str
) -> None:
    """Row 8: a forgotten record leaves search, get, group-profile reads and
    picture bytes, for every asker."""
    world = fixture.world
    await _seed_profile_items(world)
    if kind == "episode":
        text, record = CHARLIE, await _record_id(world, CHARLIE)
    elif kind == "picture":
        hits = await world.deps.indexes.image_memory.find(
            all_of(eq("image_id", fixture.picture)), limit=10
        )
        text, record = PICTURE, hits[0].stored.ref.id
    else:
        items = await world.stores.group_profile.items(world.scope("trip"))
        found = next(i for i in items if i.item.statement == "charlie decision marker")
        text, record = "charlie decision marker", found.ref.id
    seen_before = await _everything_read_by(fixture, "u_chen")
    assert text in seen_before

    done = await fixture.client.post(
        "/api/v2/memory/forget",
        json=_forget_body(
            "u_chen", {"kind": "record", "record_ids": [record]}, dry_run=False
        ),
    )
    assert done.status_code == 200, done.text
    await world.project()

    for asker in ("u_chen", "u_evan", "u_dan"):
        fixture.decider.prompts.clear()
        assert text not in await _everything_read_by(fixture, asker), (asker, kind)
    if kind != "picture":
        return
    for asker in ("u_chen", "u_dan", "u_evan", "u_zoe", None):
        params = {"asker_id": asker} if asker else {}
        picture = await fixture.client.get(
            f"/api/v2/memory/images/{fixture.picture}", params=params
        )
        assert picture.status_code == 404, asker


def _member_profile(member: str, *items: tuple[str, int, int]) -> Profile:
    return Profile(
        owner_id=member,
        summary=f"summary marker of {member}",
        timestamp=T0 + 9_000,
        explicit_info=[
            ExplicitInfoItem(
                description=text,
                category="plans",
                item_id=f"pi_{start}",
                source_memcell_ids=[f"mc_{start}"],
                source_start=T0 + start,
                source_end=T0 + end,
            )
            for text, start, end in items
        ],
    )


@pytest.mark.parametrize("asker", ["u_chen", "u_evan", "u_dan"])
async def test_profile_items_outside_the_window_never_attach(
    fixture: _Fixture, asker: str
) -> None:
    """Row 4 and section 13.5: a profile item whose source is invisible."""
    world = fixture.world
    early, late = "early profile marker Lisbon", "late profile marker Lisbon"
    await world.stores.profile.write(
        world.scope("trip"),
        _member_profile("u_chen", (early, 1000, 2000), (late, 8000, 9000)),
    )
    await world.project()
    # The query names Chen, so his profile is attached for every asker.
    response = await world.memory.search(
        _search(group_id="trip", asker_id=asker).model_copy(
            update={"query": "What does u_chen plan in Lisbon?"}
        )
    )
    listed = await world.memory.get(
        GetRequest(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id=asker,
            memory_type="member_profile",
            member_id="u_chen",
        )
    )
    for blob in (
        _everything_text(response),
        json.dumps(listed.model_dump(mode="json")),
    ):
        for text, (start, end) in ((early, (1000, 2000)), (late, (8000, 9000))):
            seen = text in blob
            assert seen == _visible(asker, T0 + start, T0 + end), (asker, text)
        # The summary is written over both items: only an asker who sees both.
        both = _visible(asker, T0 + 1000, T0 + 2000) and _visible(
            asker, T0 + 8000, T0 + 9000
        )
        assert ("summary marker" in blob) == both


@pytest.mark.parametrize("asker", ["u_chen", "u_evan", "u_dan"])
async def test_a_scene_reaches_only_askers_who_see_all_its_episodes(
    fixture: _Fixture, asker: str
) -> None:
    """Section 9.1: a record with several sources is visible only if all are."""
    world = fixture.world
    text = "scene marker plan Lisbon offsite story"
    await world.stores.scene.append(
        world.scope("trip"),
        Scene(
            owner_id="trip",
            scene_id="cl_leak",
            version=1,
            member_ids=["gep_a", "gep_b"],
            subject="Lisbon offsite",
            summary=text,
            narrative=text,
            timestamp=T0 + 9_000,
            source_start=T0 + 1_000,
        ),
        supersedes=None,
        only_if=NoSourceCheck(),
    )
    await world.project()
    fixture.decider.prompts.clear()
    # No session filter: a scene spans episodes and belongs to no session.
    response = await world.memory.search(
        _search(group_id="trip", asker_id=asker).model_copy(update={"filters": None})
    )
    visible = _visible(asker, T0 + 1_000, T0 + 9_000)
    assert (text in _everything_text(response)) == visible, asker
    if not visible:
        assert text not in "\n".join(fixture.decider.prompts)
    listed = await world.memory.get(
        GetRequest(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id=asker,
            memory_type="scene",
        )
    )
    assert bool(listed.data.scenes) == visible


async def test_an_agent_case_from_another_group_stays_there(fixture: _Fixture) -> None:
    """Row 12 and section 13.5: an agent case from another group."""
    world = fixture.world
    agent = Scope(
        app_id=APP, project_id=PROJECT, owner_id="bot_ava", owner_type="agent"
    )
    for group, intent in (
        ("trip", "trip case marker Lisbon"),
        ("other", "other case marker Lisbon"),
    ):
        await world.stores.agent_case.append(
            agent,
            AgentCase(
                owner_id="bot_ava",
                task_intent=intent,
                approach="Steps.",
                quality_score=0.9,
                timestamp=T0 + 1_000,
                session_id=f"s_{group}",
                parent_id=f"mc_{group}",
                source_group_id=group,
            ),
            only_if=NoSourceCheck(),
        )
    await world.project()

    def listing(**scope: Any) -> GetRequest:
        # Agent memory is read through get: multi-round search covers user
        # and group memory only, and refuses an agent scope.
        return GetRequest(
            app_id=APP,
            project_id=PROJECT,
            agent_id="bot_ava",
            memory_type="agent_case",
            **scope,
        )

    def text(response: Any) -> str:
        return json.dumps(response.data.model_dump(mode="json"))

    in_trip = text(
        await world.memory.get(listing(context_group_id="trip", asker_id="u_chen"))
    )
    assert "trip case marker" in in_trip and "other case marker" not in in_trip
    plain = text(await world.memory.get(listing()))
    assert "case marker" not in plain
    with pytest.raises(ForbiddenError):
        await world.memory.get(listing(context_group_id="trip", asker_id="u_zoe"))


async def test_group_strategies_read_only_the_groups_own_segments(
    tmp_path: Path,
) -> None:
    """Row 11: group strategies read only the group's segments, member
    profiles never read ``users/<m>/``, personal strategies never see a group
    segment, and the group profile's update reads only the group's own."""
    profile_prompts: list[str] = []

    def model(messages: Any, *, response_format: type, **_kw: Any) -> str:
        name = response_format.__name__
        if name == "GroupProfileReply":
            profile_prompts.append("\n".join(str(m.content) for m in messages))
            return json.dumps({"operations": []})
        if name == "_BatchReply":
            return json.dumps(
                {"reasoning": "s", "boundaries": [], "tail_complete": False}
            )
        if name == "EpisodeReply":
            return json.dumps({"title": "Plan", "summary": "Plan.", "content": "Plan."})
        if name == "_ProfileUpdateReply":
            prompt = "\n".join(str(m.content) for m in messages)
            assert "personal marker" not in prompt
            item = {"evidence": "x", "description": "Plans", "category": "plans"}
            return json.dumps(
                {
                    "operations": [
                        {"action": "add", "type": "explicit_info", "data": item}
                    ]
                }
            )
        raise AssertionError(name)

    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        await world.roster("trip", [join("u_chen", T0), join("u_betty", T0)])
        await world.personal_episode("u_chen", "personal marker Porto", at=T0 + 100)
        await world.profile("u_chen", "personal marker Chen lives in Porto")
        user_md = next((tmp_path / "default_app").rglob("user.md"))
        before = user_md.read_bytes()
        await world.memory.add(
            {"session_id": "s_group", "app_id": APP, "project_id": PROJECT,
             "group_id": "trip",
             "messages": [
                 {"sender_id": "u_chen", "role": "user", "timestamp": T0 + 1_000,
                  "content": "group marker: book the venue"},
                 {"sender_id": "u_betty", "role": "user", "timestamp": T0 + 2_000,
                  "content": "group marker: ok"},
             ]},
            request_id="r",
            async_mode=False,
            defer_extraction=True,
        )  # fmt: skip
        await world.memory.flush(
            session_id="s_group", request_id="f", app_id=APP, project_id=PROJECT
        )
        await world.run(*STRATEGY_SPECS)
        episodes = [
            e for e in world.bus.events if type(e).__name__ == "EpisodeExtracted"
        ]
        assert {e.owner_type for e in episodes} == {"group"}  # no user's episode
        group_cells = {e.memcell_id for e in episodes}
        for member in ("u_chen", "u_betty"):
            profile = await world.stores.profile.read(world.scope("trip"), member)
            assert profile is not None
            for item in profile.explicit_info:
                assert (
                    set((item.model_extra or {})["source_memcell_ids"]) <= group_cells
                )
        assert user_md.read_bytes() == before
        (prompt,) = profile_prompts  # the flush ran the group profile's batch
        assert "personal marker" not in prompt
        assert "group marker: book the venue" in prompt
        assert [m for m in group_cells if f"=== SEGMENT {m} ===" in prompt] == list(
            group_cells
        )


PROFILE_ITEMS = {
    "alpha decision marker": (1000, 2000),
    "bravo decision marker": (3000, 4000),
    "straddle decision marker": (2000, 3000),
    "charlie decision marker": (8000, 9000),
}
"""Group-profile items at the episodes' spans (offsets from T0)."""


async def _seed_profile_items(world: GroupWorld) -> None:
    for text, (start, end) in PROFILE_ITEMS.items():
        await world.profile_item("trip", text, start=T0 + start, end=T0 + end)
    # A decision whose newer version lies where a late joiner can see it: the
    # old one is superseded, so no current read returns it to anyone.
    old = await world.profile_item(
        "trip", "replaced decision marker", start=T0 + 1100, end=T0 + 1200
    )
    await world.profile_item(
        "trip",
        "replacing decision marker",
        start=T0 + 8100,
        end=T0 + 8200,
        supersedes=old.entry_id,
    )
    await world.project()


@pytest.mark.parametrize("asker", ["u_chen", "u_evan", "u_dan"])
@pytest.mark.parametrize("history", [False, True])
async def test_group_profile_items_follow_the_windows(
    fixture: _Fixture, asker: str, history: bool
) -> None:
    """Section 13.5: a group-profile item whose source is invisible is absent."""
    world = fixture.world
    await _seed_profile_items(world)

    listed = await world.memory.get(
        GetRequest(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id=asker,
            memory_type="group_profile",
            include_superseded=history,
        )
    )
    blob = json.dumps(listed.model_dump(mode="json"))
    for text, (start, end) in PROFILE_ITEMS.items():
        assert (text in blob) == _visible(asker, T0 + start, T0 + end), (asker, text)
    replaced_seen = history and _visible(asker, T0 + 1100, T0 + 1200)
    assert ("replaced decision marker" in blob) == replaced_seen
    assert ("replacing decision marker" in blob) == _visible(
        asker, T0 + 8100, T0 + 8200
    )


CONSTRAINED = {
    "kappa": (1_000, 2_000),
    "lambda": (3_000, 4_000),
    "sigma": (8_000, 9_000),
}
"""Segments with an episode and a decision each; the marker word is quoted."""


async def _seed_constrained(world: GroupWorld) -> None:
    for word, (start, end) in CONSTRAINED.items():
        memcell = f"mc_{word}"
        await world.segment(
            memcell,
            (
                f"m-{word}",
                "u_chen",
                f"{word}quote marker plan Lisbon offsite",
                T0 + start,
            ),
        )
        await world.episode(
            "trip",
            f"{word}episode marker plan Lisbon offsite",
            start=T0 + start,
            end=T0 + end,
            memcell_id=memcell,
        )
        await world.profile_item(
            "trip",
            f"{word}quote",
            start=T0 + start,
            end=T0 + start,
            memcell_id=memcell,
            statement=f"{word}decision marker",
            sources=[
                ProfileSource(
                    memcell_id=memcell,
                    message_id=f"m-{word}",
                    speaker_id="u_chen",
                    at=T0 + start,
                    quote=f"{word}quote",
                )
            ],
        )
    await world.project()


@pytest.mark.parametrize("asker", ["u_chen", "u_evan", "u_dan"])
async def test_constraints_and_attachments_follow_the_windows(
    fixture: _Fixture, asker: str
) -> None:
    """Section 13.5: a constraint or attached profile item from an invisible source."""
    world = fixture.world
    await _seed_constrained(world)

    response = await world.memory.search(
        SearchRequest(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id=asker,
            query="marker plan Lisbon offsite",
            top_k=20,
        )
    )

    blob = _everything_text(response)
    assert response.data.constraints, "the fixture must produce constraints"
    for word, (start, end) in CONSTRAINED.items():
        seen = _visible(asker, T0 + start, T0 + end)
        for marker in (f"{word}episode", f"{word}decision", f"{word}quote"):
            assert (marker in blob) == seen, (asker, marker)


async def test_a_project_scope_reads_only_the_askers_groups(fixture: _Fixture) -> None:
    """Row 10 and section 8.7: project scope, project profile, knowledge base."""
    world = fixture.world
    await world.roster("secret", [join("u_chen", T0), join("u_zoe", T0)])
    await world.episode("secret", "secret marker plan Lisbon offsite", start=T0, end=T0)
    await world.profile_item("secret", "secret decision", start=T0, end=T0)
    await _seed_constrained(world)

    searched = await world.memory.search(
        SearchRequest(
            app_id=APP,
            project_id=PROJECT,
            project_scope=True,
            asker_id="u_evan",
            query="marker plan Lisbon offsite",
            top_k=20,
        )
    )
    profile = await world.memory.project_profile(
        app_id=APP, project_id=PROJECT, asker_id="u_evan", asserted_asker=None
    )
    blob = _everything_text(searched) + json.dumps(
        [i.model_dump(mode="json") for i in profile.items]
    )
    assert "secret" not in blob
    for word, (start, end) in CONSTRAINED.items():
        seen = _visible("u_evan", T0 + start, T0 + end)
        assert (f"{word}decision" in blob) == seen, word
    access = AccessPolicy(world.stores.groups)
    for outsider in ("u_betty", "u_zoe_never", "u_amy"):
        with pytest.raises(ForbiddenError):
            await access.authorize_project_read(
                resolve_identity(header=None, body_asker=outsider), APP, PROJECT
            )
    with pytest.raises(ForbiddenError):
        await world.memory.project_profile(
            app_id=APP, project_id=PROJECT, asker_id="u_betty", asserted_asker=None
        )


# ── include_groups: a personal search over the user's groups ─────────────

OWN = {
    "u_chen": PERSONAL,
    "u_evan": "personal marker plan Lisbon offsite of Evan",
    "u_dan": "personal marker plan Lisbon offsite of Dan",
    "u_betty": "personal marker plan Lisbon offsite of Betty",
    "u_zoe": "personal marker plan Lisbon offsite of Zoe",
}
"""Each user's own personal record (Chen's is the fixture's)."""


async def _seed_own_memory(world: GroupWorld) -> None:
    for user, text in OWN.items():
        if user != "u_chen":
            await world.personal_episode(user, text, at=T0 + 1000)
    for member in ("u_chen", "u_evan", "u_dan"):
        await world.stores.profile.write(
            world.scope("trip"),
            _member_profile(
                member,
                (f"{member} early profile marker", 1000, 2000),
                (f"{member} late profile marker", 8000, 9000),
            ),
        )
    await _seed_profile_items(world)


@pytest.mark.parametrize("asker", ["u_chen", "u_evan", "u_dan"])
async def test_include_groups_reads_the_group_as_the_members_group_search(
    fixture: _Fixture, asker: str
) -> None:
    """A later joiner (Evan) and a member who was away (Dan) get nothing said
    outside their windows -- Bravo and Straddle were said while Dan was
    away, so their audience leaves him out -- in the results, the decider's
    input, attached pictures, profiles and decisions, and the buffer."""
    world = fixture.world
    await _seed_own_memory(world)
    fixture.decider.prompts.clear()
    response = await world.memory.search(
        _mixed(asker, query=f"What does u_chen plan? {OWN[asker]}")
    )
    buffered = await world.memory.search(
        _mixed(asker, filters={"session_id": "s_group"})
    )
    group = await world.memory.search(
        _search(group_id="trip", asker_id=asker).model_copy(update={"filters": None})
    )
    listed = await world.memory.get(_mixed_listing(asker))
    blob = (
        _everything_text(response)
        + _everything_text(buffered)
        + json.dumps(listed.model_dump(mode="json"))
    )
    seen_by_decider = "\n".join(fixture.decider.prompts)

    # The listing holds exactly their own record and what their windows hold.
    expected = {t for t, (s, e) in GROUP_RECORDS.items() if _visible(asker, s, e)}
    assert {e.episode for e in listed.data.episodes} == expected | {OWN[asker]}
    visible = [t for t, (s, e) in GROUP_RECORDS.items() if _visible(asker, s, e)]
    assert any(text in seen_by_decider for text in visible), asker
    assert any(text in blob for text in visible), asker
    for text, (start, end) in GROUP_RECORDS.items():
        if not _visible(asker, start, end):
            assert text not in blob, (asker, text)
            assert text not in seen_by_decider, (asker, text)
    if not _visible(asker, T0 + 1500, T0 + 1500):
        assert PICTURE not in blob and PICTURE not in seen_by_decider
        assert fixture.picture not in json.dumps(
            [i.image_id for i in response.data.images]
            + [i.image_id for e in response.data.episodes for i in e.images]
        )
    assert ("buffered marker early" in blob) == _visible(asker, T0 + 1000, T0 + 1000)
    assert "buffered marker late" in blob
    # Their own memory answers; nobody else's personal memory comes along.
    assert OWN[asker] in blob
    for user, text in OWN.items():
        if user != asker:
            assert text not in blob and text not in seen_by_decider, (asker, user)
    assert ("personal marker Chen lives in Porto" in blob) == (asker == "u_chen")
    # Their own profile in the group, cut to their windows; never another
    # member's, although the question names Chen.
    for member in ("u_chen", "u_evan", "u_dan"):
        for when, (start, end) in (("early", (1000, 2000)), ("late", (8000, 9000))):
            shown = f"{member} {when} profile marker" in blob
            mine = member == asker and _visible(asker, T0 + start, T0 + end)
            assert shown == mine, (asker, member, when)
    for text, (start, end) in PROFILE_ITEMS.items():
        assert (text in blob) == _visible(asker, T0 + start, T0 + end), (asker, text)
    assert "replaced decision marker" not in blob
    # Each group record carries the audience a group search gives it.
    audience = {e.id: e.audience for e in group.data.episodes}
    audience |= {i.id: i.audience for i in group.data.images}
    for record in (*response.data.episodes, *response.data.images):
        if record.owner_type == "group":
            assert record.audience == audience[record.id], (asker, record.id)
        else:
            assert record.owner_type == "user" and record.audience is None


@pytest.mark.parametrize("asker", ["u_betty", "u_zoe"])
@pytest.mark.parametrize("named", [False, True])
async def test_include_groups_gives_former_and_non_members_nothing_of_the_group(
    fixture: _Fixture, asker: str, named: bool
) -> None:
    """Betty left the group, Zoe never joined: their own memory answers, and
    nothing of the group does -- not even when they name it."""
    world = fixture.world
    await _seed_own_memory(world)
    fixture.decider.prompts.clear()
    extra: dict[str, Any] = {"include_group_ids": ["trip"]} if named else {}
    response = await world.memory.search(_mixed(asker, **extra))
    buffered = await world.memory.search(
        _mixed(asker, filters={"session_id": "s_group"}, **extra)
    )
    listed = await world.memory.get(_mixed_listing(asker, **extra))
    blob = (
        _everything_text(response)
        + _everything_text(buffered)
        + json.dumps(listed.model_dump(mode="json"))
    )
    seen_by_decider = "\n".join(fixture.decider.prompts)

    assert OWN[asker] in blob
    assert [e.episode for e in listed.data.episodes] == [OWN[asker]]
    for text in (*GROUP_RECORDS, PICTURE, *PROFILE_ITEMS, "buffered marker"):
        assert text not in blob, (asker, text)
        assert text not in seen_by_decider, (asker, text)
    assert "profile marker" not in blob
    assert {e.owner_type for e in response.data.episodes} == {"user"}
    for answer in (response, buffered):
        assert answer.data.scope is not None
        assert answer.data.scope.group_ids == [] and answer.data.scope.windows == []
        assert answer.data.group_profile is None
        assert answer.data.unprocessed_messages == []


SECRET_SAID = "ZEBRA4417"
"""Said in a message its sender then forgot."""
SECRET_BUFFERED = "QUOKKA88"
"""Said in a message forgotten while still in the buffer."""
KEPT_SAID = "LISBONVENUE"


class _ChatModel:
    """Episodes repeat the marker words they were shown; the decider keeps the
    first ten candidates and records what it was shown."""

    def __init__(self) -> None:
        self.decider_prompts: list[str] = []

    def __call__(self, messages: Any, *, response_format: type, **_: Any) -> str:
        name = response_format.__name__
        prompt = "\n".join(str(getattr(m, "content", m)) for m in messages)
        if name == "_BatchReply":
            return json.dumps(
                {"reasoning": "s", "boundaries": [], "tail_complete": False}
            )
        if name == "EpisodeReply":
            words = (SECRET_SAID, SECRET_BUFFERED, KEPT_SAID)
            said = ", ".join(w for w in words if w in prompt) or "nothing"
            return json.dumps(
                {
                    "title": "Trip plans marker",
                    "summary": f"Plans marker: {said}.",
                    "content": f"The group talked about the plan marker {said}.",
                }
            )
        if name == "_DecisionReply":
            self.decider_prompts.append(prompt)
            return json.dumps({"core": list(range(10)), "next_queries": []})
        raise AssertionError(f"unexpected model call {name}")


async def test_include_groups_never_reads_a_forgotten_message(tmp_path: Path) -> None:
    """A message its sender forgot: its segment is re-extracted without it,
    and a buffered one is left out of the buffer -- for every member's
    personal search over the group as for their group search."""
    model = _ChatModel()
    client = FakeLLMClient(handler=model)
    async with GroupWorld.open(tmp_path, llm=client, decider=client) as world:
        await world.roster(
            "trip",
            [
                join("u_chen", T0, role="owner", display_name="Chen"),
                join("u_dan", T0, display_name="Dan"),
                join("u_evan", T0, display_name="Evan"),
            ],
        )

        async def add(*lines: tuple[str, str, str, int]) -> None:
            await world.memory.add(
                {"session_id": "s_trip", "app_id": APP, "project_id": PROJECT,
                 "group_id": "trip",
                 "messages": [
                     {"message_id": mid, "sender_id": who, "role": "user",
                      "timestamp": T0 + at, "content": text}
                     for mid, who, text, at in lines
                 ]},
                request_id="r",
                async_mode=False,
                defer_extraction=True,
            )  # fmt: skip

        async def forget(message_id: str) -> Any:
            return await world.memory.forget_in_group(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_dan",
                target=GroupForgetTarget(
                    kind="message", session_id="s_trip", message_ids=(message_id,)
                ),
            )

        await add(
            ("m-1", "u_dan", f"My locker code is {SECRET_SAID}, keep it", 1000),
            ("m-2", "u_chen", f"We meet at the {KEPT_SAID} on Friday", 2000),
        )
        await world.memory.flush(
            session_id="s_trip", request_id="f", app_id=APP, project_id=PROJECT
        )
        await world.project()
        before = await world.memory.search(_mixed("u_evan"))
        assert SECRET_SAID in _everything_text(before)  # the group content flows

        await forget("m-1")
        await add(
            ("m-3", "u_dan", f"The alarm code is {SECRET_BUFFERED}", 3000),
            ("m-4", "u_chen", "We still need a BOATRIDE", 4000),
        )
        buffered = await forget("m-3")
        assert buffered.episodes == ()  # still in the buffer: nothing extracted
        await world.project()

        for asker in ("u_chen", "u_dan", "u_evan"):
            model.decider_prompts.clear()
            found = await world.memory.search(_mixed(asker))
            in_buffer = await world.memory.search(
                _mixed(asker, filters={"session_id": "s_trip"})
            )
            blob = _everything_text(found) + _everything_text(in_buffer)
            seen = "\n".join(model.decider_prompts)
            assert KEPT_SAID in blob, asker  # the rest is still remembered
            for secret in (SECRET_SAID, SECRET_BUFFERED):
                assert secret not in blob, (asker, secret)
                assert secret not in seen, (asker, secret)
            # The buffer is read by the group's rules: what was not forgotten.
            shown = [m.content for m in in_buffer.data.unprocessed_messages]
            assert shown == ["We still need a BOATRIDE"], asker


PAIR_SAID = "PAIRWALNUT31"
"""Said by Ann and Bob, two people talking without a group_id."""
LATER_PAIR_SAID = "PAIRHAZEL77"
"""Said by Ann and Bob again, in a later session of their own."""
TRIO_SAID = "TRIOPECAN52"
"""Said by Ann, Bob and Cat together."""
CAT_REFUSED = "CATALMOND09"
"""Said by Cat into Ann and Bob's session: refused."""
_SAID = (PAIR_SAID, LATER_PAIR_SAID, TRIO_SAID, CAT_REFUSED)


class _PeopleModel:
    """Episodes repeat the marker words they were shown; the decider keeps the
    first ten candidates and records what it was shown."""

    def __init__(self) -> None:
        self.decider_prompts: list[str] = []

    def __call__(self, messages: Any, *, response_format: type, **_: Any) -> str:
        name = response_format.__name__
        prompt = "\n".join(str(getattr(m, "content", m)) for m in messages)
        if name == "_BatchReply":
            return json.dumps(
                {"reasoning": "s", "boundaries": [], "tail_complete": False}
            )
        if name == "EpisodeReply":
            said = ", ".join(w for w in _SAID if w in prompt) or "nothing"
            return json.dumps(
                {
                    "title": "Cabin marker",
                    "summary": f"Cabin marker: {said}.",
                    "content": f"They planned the cabin marker {said}.",
                }
            )
        if name == "_DecisionReply":
            self.decider_prompts.append(prompt)
            return json.dumps({"core": list(range(10)), "next_queries": []})
        raise AssertionError(f"unexpected model call {name}")


async def _talk(world: GroupWorld, session: str, *lines: tuple[str, str]) -> None:
    """``lines`` of ``(sender, text)`` sent without a group_id, then flushed."""
    await world.memory.add(
        {"session_id": session, "app_id": APP, "project_id": PROJECT,
         "messages": [
             {"sender_id": who, "sender_name": who[2:].title(), "role": "user",
              "timestamp": T0 + 1000 * n, "content": text}
             for n, (who, text) in enumerate(lines, start=1)
         ]},
        request_id="r",
        async_mode=False,
        defer_extraction=True,
    )  # fmt: skip
    await world.memory.flush(
        session_id=session, request_id="f", app_id=APP, project_id=PROJECT
    )


def _people(**fields: Any) -> SearchRequest:
    values: dict[str, Any] = {
        "app_id": APP,
        "project_id": PROJECT,
        "query": "cabin marker",
        "top_k": 20,
        **fields,
    }
    return SearchRequest(**values)


async def test_an_implicit_group_is_read_by_its_people_only(tmp_path: Path) -> None:
    """Two people talking without a group_id are a group (section 7.9): both
    read it, a stranger reads nothing of it on any path, and neither person's
    own memory holds it."""
    model = _PeopleModel()
    client = FakeLLMClient(handler=model)
    async with GroupWorld.open(tmp_path, llm=client, decider=client) as world:
        await _talk(
            world,
            "s_pair",
            ("u_ann", f"Book the cabin, the code is {PAIR_SAID}"),
            ("u_bob", "Done, cabin marker booked"),
        )
        pair = await world.stores.groups.binding(world.session("s_pair"))
        assert pair is not None and pair.startswith("implicit_")
        await world.project()

        for person in ("u_ann", "u_bob"):
            model.decider_prompts.clear()
            group = await world.memory.search(_people(group_id=pair, asker_id=person))
            mixed = await world.memory.search(_mixed(person, query="cabin marker"))
            assert PAIR_SAID in _everything_text(group), person
            assert PAIR_SAID in _everything_text(mixed), person
            own = await world.memory.search(_people(user_id=person))
            assert PAIR_SAID not in _everything_text(own), person

        model.decider_prompts.clear()
        with pytest.raises(ForbiddenError):
            await world.memory.search(_people(group_id=pair, asker_id="u_zoe"))
        with pytest.raises(ForbiddenError):
            await world.memory.get(
                GetRequest(
                    app_id=APP,
                    project_id=PROJECT,
                    group_id=pair,
                    asker_id="u_zoe",
                    memory_type="episode",
                )
            )
        with pytest.raises(ForbiddenError):
            await world.memory.search(_people(user_id="u_ann", asker_id="u_zoe"))
        stranger = await world.memory.search(_mixed("u_zoe", query="cabin marker"))
        listed = await world.memory.get(_mixed_listing("u_zoe"))
        blob = _everything_text(stranger) + json.dumps(listed.model_dump(mode="json"))
        assert PAIR_SAID not in blob
        assert PAIR_SAID not in "\n".join(model.decider_prompts)


async def test_a_third_person_joining_later_reads_nothing_of_the_pair(
    tmp_path: Path,
) -> None:
    """Cat speaking into Ann and Bob's session is refused: an implicit group's
    people never change. The three of them in a session of their own are a
    group of their own, which Cat reads; the pair's group -- what they said
    before and what they say later in a session of their own -- Cat never
    reads, on any path."""
    model = _PeopleModel()
    client = FakeLLMClient(handler=model)
    async with GroupWorld.open(tmp_path, llm=client, decider=client) as world:
        await _talk(
            world,
            "s_pair",
            ("u_ann", f"Cabin marker: the door code is {PAIR_SAID}"),
            ("u_bob", "Noted, cabin marker"),
        )
        pair = await world.stores.groups.binding(world.session("s_pair"))
        assert pair is not None
        with pytest.raises(RosterConflictError):
            await _talk(
                world,
                "s_pair",
                ("u_ann", "Cat, meet Bob"),
                ("u_cat", f"Hi both, cabin marker {CAT_REFUSED}"),
            )
        await _talk(
            world,
            "s_trio",
            ("u_ann", f"Cabin marker for three: {TRIO_SAID}"),
            ("u_cat", "Great, cabin marker"),
            ("u_bob", "See you there"),
        )
        await _talk(
            world,
            "s_pair_later",
            ("u_bob", f"Just us: cabin marker {LATER_PAIR_SAID}"),
            ("u_ann", "Sure"),
        )
        assert await world.stores.groups.binding(world.session("s_pair_later")) == pair
        trio = await world.stores.groups.binding(world.session("s_trio"))
        assert trio is not None and trio != pair
        await world.project()

        model.decider_prompts.clear()
        with pytest.raises(ForbiddenError):
            await world.memory.search(_people(group_id=pair, asker_id="u_cat"))
        found = await world.memory.search(_mixed("u_cat", query="cabin marker"))
        listed = await world.memory.get(_mixed_listing("u_cat"))
        blob = _everything_text(found) + json.dumps(listed.model_dump(mode="json"))
        seen = "\n".join(model.decider_prompts)
        assert TRIO_SAID in blob
        for secret in (PAIR_SAID, LATER_PAIR_SAID, CAT_REFUSED):
            assert secret not in blob, secret
            assert secret not in seen, secret
        listed_ann = await world.memory.get(_mixed_listing("u_ann"))
        everything = json.dumps(listed_ann.model_dump(mode="json"))
        for said in (PAIR_SAID, LATER_PAIR_SAID, TRIO_SAID):
            assert said in everything, said
        assert CAT_REFUSED not in everything


def test_every_read_path_of_section_9_3_is_accounted_for() -> None:
    covered = {1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12}
    assert covered | set(NOT_YET) == set(range(1, 14))
    assert not covered & set(NOT_YET)
