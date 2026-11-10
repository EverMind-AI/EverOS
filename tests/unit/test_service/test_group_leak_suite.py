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
personal episode and profile of ``u_chen``.

Every case asserts the record is absent from search (keyword, vector,
hybrid and the multi-round decider's very input), get, attached pictures,
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
from everos.memory.ports import SessionKey
from everos.service import GetRequest, SearchRequest
from tests.fakes import make_application
from tests.helpers.groups import APP, PROJECT, GroupWorld, join, leave

NOT_YET: dict[int, str] = {
    6: "history route: not built yet (2.0 history); reads go through visible()",
    8: "group forget / undo / revise / delete: refused with 422 until Step 2",
    9: "knowledge reads by project members: Step 2 (task 17)",
    11: "group strategies: no group write path yet; group cells reach no "
    "one-to-one pipeline (tests/unit/test_service/test_memorize_groups.py)",
    12: "group-sourced agent cases: no agent pipeline runs on group cells yet",
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
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=decider)) as world:
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


def _search(method: str, **scope: Any) -> SearchRequest:
    return SearchRequest(
        app_id=APP,
        project_id=PROJECT,
        query="marker plan Lisbon offsite",
        method=method,
        top_k=20,
        filters={"session_id": "s_group"},
        include_profile=True,
        **scope,
    )


def _everything_text(response: Any) -> str:
    """Every string a search response hands back, in one blob."""
    return json.dumps(response.data.model_dump(mode="json"))


@pytest.mark.parametrize("asker", ["u_chen", "u_evan", "u_dan"])
@pytest.mark.parametrize("method", ["keyword", "vector", "hybrid", "llm_multiround"])
async def test_search_returns_nothing_outside_the_askers_windows(
    fixture: _Fixture, asker: str, method: str
) -> None:
    """Rows 1-4: every recall, the decider's input, attached pictures and
    profiles, and the session buffer."""
    fixture.decider.prompts.clear()
    response = await fixture.world.memory.search(
        _search(method, group_id="trip", asker_id=asker)
    )
    blob = _everything_text(response)
    seen_by_decider = "\n".join(fixture.decider.prompts)

    visible = [t for t, (s, e) in GROUP_RECORDS.items() if _visible(asker, s, e)]
    if method == "llm_multiround":
        # The decider really ran, and saw the asker's own records.
        assert any(text in seen_by_decider for text in visible), (asker, method)
    assert any(text in blob for text in visible), (asker, method)
    for text, (start, end) in GROUP_RECORDS.items():
        if not _visible(asker, start, end):
            assert text not in blob, (asker, method, text)
            assert text not in seen_by_decider, (asker, method, text)
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
        await fixture.world.memory.search(
            _search("keyword", group_id="trip", asker_id=asker)
        )
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
        await fixture.world.memory.search(
            _search("keyword", user_id="u_chen", asker_id=reader)
        )
    # Without an asker the read is the owner's own; another owner id sees
    # neither Chen's memory nor the group's buffer.
    other = await fixture.world.memory.search(_search("keyword", user_id=reader))
    blob = _everything_text(other)
    assert "personal marker" not in blob
    assert "buffered marker" not in blob
    assert not any(marker in blob for marker in GROUP_RECORDS)


async def test_a_group_buffer_never_reads_through_a_personal_scope(
    fixture: _Fixture,
) -> None:
    """Row 3: Chen sent those messages, but the session is the group's."""
    mine = await fixture.world.memory.search(_search("keyword", user_id="u_chen"))
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


def test_every_read_path_of_section_9_3_is_accounted_for() -> None:
    covered = {1, 2, 3, 4, 5, 7, 10}
    assert covered | set(NOT_YET) == set(range(1, 14))
    assert not covered & set(NOT_YET)
