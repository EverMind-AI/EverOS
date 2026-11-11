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

from everos.core.errors import ForbiddenError
from everos.entrypoints.api.app import create_app
from everos.memory.models import (
    AgentCase,
    ExplicitInfoItem,
    MemberProfile,
    Profile,
    Scene,
)
from everos.memory.ports import NoSourceCheck, Scope, SessionKey
from everos.memory.strategies import STRATEGY_SPECS
from everos.service import GetRequest, SearchRequest
from tests.fakes import make_application
from tests.helpers.groups import APP, PROJECT, GroupWorld, join, leave

NOT_YET: dict[int, str] = {
    6: "history route: not built yet (2.0 history); reads go through visible()",
    8: "group forget / undo / revise / delete: refused with 422 until Step 2",
    9: "knowledge reads by project members: Step 2 (task 17)",
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


async def test_group_forget_and_delete_are_refused(fixture: _Fixture) -> None:
    """Row 8, until Step 2: no partial forget can run."""
    for path, body in (
        ("/api/v2/memory/delete", {"project_id": PROJECT, "session_id": "s_group"}),
        ("/api/v2/memory/delete", {"group_id": "trip", "session_id": "s_group"}),
        (
            "/api/v2/memory/forget/session",
            {"user_id": "u_chen", "group_id": "trip", "session_id": "s_group"},
        ),
    ):
        refused = await fixture.client.post(path, json=body)
        assert refused.status_code == 422, (path, refused.text)


def _member_profile(member: str, *items: tuple[str, int, int]) -> MemberProfile:
    return MemberProfile(
        group_id="trip",
        member_id=member,
        profile=Profile(
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
        ),
    )


@pytest.mark.parametrize("asker", ["u_chen", "u_evan", "u_dan"])
async def test_profile_items_outside_the_window_never_attach(
    fixture: _Fixture, asker: str
) -> None:
    """Row 4 and section 13.5: a profile item whose source is invisible."""
    world = fixture.world
    early, late = "early profile marker Lisbon", "late profile marker Lisbon"
    await world.stores.member_profile.write(
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
    segment."""

    def model(messages: Any, *, response_format: type, **_kw: Any) -> str:
        name = response_format.__name__
        if name == "_BatchReply":
            return json.dumps(
                {"reasoning": "s", "boundaries": [], "tail_complete": False}
            )
        if name == "_EpisodeReply":
            return json.dumps({"title": "Plan", "summary": "Plan.", "content": "Plan."})
        if name == "_ProfileItems":
            prompt = "\n".join(str(m.content) for m in messages)
            assert "personal marker" not in prompt
            return json.dumps(
                {
                    "explicit_info": [
                        {"evidence": "x", "description": "Plans", "category": "plans"}
                    ],
                    "implicit_traits": [],
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
        kinds = {type(e).__name__ for e in world.bus.events}
        assert not kinds & {"UserPipelineStarted", "EpisodeExtracted"}
        group_cells = {
            e.memcell_id
            for e in world.bus.events
            if type(e).__name__ == "GroupEpisodeExtracted"
        }
        for member in ("u_chen", "u_betty"):
            profile = await world.stores.member_profile.read(
                world.scope("trip"), member
            )
            assert profile is not None
            for item in profile.profile.explicit_info:
                assert (
                    set((item.model_extra or {})["source_memcell_ids"]) <= group_cells
                )
        assert user_md.read_bytes() == before


def test_every_read_path_of_section_9_3_is_accounted_for() -> None:
    covered = {1, 2, 3, 4, 5, 7, 10, 11, 12}
    assert covered | set(NOT_YET) == set(range(1, 14))
    assert not covered & set(NOT_YET)
