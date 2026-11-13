"""The group's deliverer: one program per group, its key, its superadmin rights.

Real markdown, ``system.db`` and LanceDB behind the bootstrap's memory service
(``tests/helpers/groups.py``), and the real app over HTTP where the wire shape
matters. What these pin (docs/group-memory.md, sections 5.1, 5.2 and 9.1):

- owners and admins name the deliverer with ``PUT /groups/{id}``, rotate it
  and clear it (``null``); the deliverer itself may too; the group read
  returns it; ``group.md`` carries it once set (``null`` after a clear) and a
  rebuilt mirror reads it back; a group a batch creates has none;
- once set, ``/memory/add`` and ``/memory/flush`` of the group's sessions are
  taken from the deliverer only -- 403 for anyone else, before anything is
  queued or written -- through the body's ``asker_id`` or the gateway header;
  a group without one, and a one-to-one session, take anyone's, as before;
- the deliverer is the group's superadmin: any roster event (backdated too),
  every meta field (its own key included), forget, undo, edit and delete of
  anyone's records; the whole history to read through get, history, picture
  bytes and the group read; its search without an ``audience`` reads only
  what every current member can see, ``audience: asker`` the whole history;
- it reads nothing of another group, and the members' views are unchanged.
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
from everos.core.errors import ForbiddenError
from everos.entrypoints.api.app import create_app
from everos.service import ForgetTarget, GetRequest, MemoryChanges, SearchRequest
from tests.fakes import make_application
from tests.helpers.groups import APP, PROJECT, GroupWorld, join

T0 = 1_772_323_200_000
BOT = "gw_trip"
EARLY = "early marker Lisbon offsite plan"  # [1000, 2000], before Evan joined
LATE = "late marker Lisbon offsite plan"  # [8000, 9000], everyone was there


def _model(messages: Any, *, response_format: type, **_kw: Any) -> str:
    """Boundary: never cuts (a flush closes the segment). Episodes: scripted."""
    name = response_format.__name__
    if name == "_BatchReply":
        return json.dumps({"reasoning": "s", "boundaries": [], "tail_complete": False})
    if name == "EpisodeReply":
        return json.dumps(
            {"title": "Plan", "summary": "Lisbon plan", "content": "Lisbon plan talk"}
        )
    if name == "_DecisionReply":
        return json.dumps({"core": list(range(10)), "next_queries": []})
    return json.dumps({"operations": []})


def _llm() -> FakeLLMClient:
    return FakeLLMClient(handler=_model)


async def _group(world: GroupWorld, *, deliverer: str | None = BOT) -> None:
    """``trip``: Chen owns it, Betty from the start, Evan from ``T0 + 5000``."""
    await world.memory.upsert_group(
        app_id=APP,
        project_id=PROJECT,
        group_id="trip",
        asker_id="u_chen",
        asserted_asker=None,
        name="Trip",
        description=None,
        created_at=T0,
        history_visible_from=None,
        extra=None,
        roster=[
            world.roster_input(op="join", member_id="u_chen", role="owner", at=T0),
            world.roster_input(op="join", member_id="u_betty", at=T0),
            world.roster_input(op="join", member_id="u_evan", at=T0 + 5000),
        ],
        deliverer_id=deliverer,
    )


async def _upsert(world: GroupWorld, asker: str, **fields: Any) -> Any:
    values: dict[str, Any] = {
        "name": None,
        "description": None,
        "created_at": None,
        "history_visible_from": None,
        "extra": None,
        "roster": [],
        **fields,
    }
    return await world.memory.upsert_group(
        app_id=APP,
        project_id=PROJECT,
        group_id="trip",
        asker_id=asker,
        asserted_asker=None,
        **values,
    )


async def _read(world: GroupWorld, asker: str, group: str = "trip") -> Any:
    return await world.memory.read_group(
        app_id=APP,
        project_id=PROJECT,
        group_id=group,
        asker_id=asker,
        asserted_asker=None,
        include_history=False,
    )


async def _seed_records(world: GroupWorld) -> str:
    """Two episodes either side of Evan's join and a picture before it."""
    await world.episode("trip", EARLY, start=T0 + 1000, end=T0 + 2000)
    await world.episode(
        "trip", LATE, start=T0 + 8000, end=T0 + 9000, senders=("u_betty",)
    )
    picture = await world.picture(
        world.scope("trip"), "a floor plan", at=T0 + 1500, shade=77
    )
    await world.project()
    return picture


async def _search(world: GroupWorld, asker: str, **fields: Any) -> set[str]:
    response = await world.memory.search(
        SearchRequest(
            app_id=APP,
            project_id=PROJECT,
            group_id=fields.pop("group_id", "trip"),
            asker_id=asker,
            query="marker Lisbon offsite plan",
            top_k=10,
            **fields,
        )
    )
    return {e.episode for e in response.data.episodes}


def _batch(session: str, *lines: tuple[str, str, str], **fields: Any) -> dict[str, Any]:
    return {
        "session_id": session,
        "app_id": APP,
        "project_id": PROJECT,
        "group_id": "trip",
        "messages": [
            {
                "message_id": mid,
                "sender_id": who,
                "role": "user",
                "timestamp": T0 + 10_000 + i,
                "content": text,
            }
            for i, (mid, who, text) in enumerate(lines)
        ],
        **fields,
    }


async def _add(world: GroupWorld, payload: dict[str, Any], **flags: Any) -> Any:
    return await world.memory.add(
        payload,
        request_id="r",
        async_mode=flags.get("async_mode", False),
        defer_extraction=flags.get("defer", True),
    )


async def _flush(world: GroupWorld, session: str, asker: str | None) -> Any:
    return await world.memory.flush(
        session_id=session,
        request_id="f",
        app_id=APP,
        project_id=PROJECT,
        asker_id=asker,
    )


def _group_md(world: GroupWorld, group: str = "trip") -> str:
    path = world.root.groups_dir(APP, PROJECT) / group / "group.md"
    return path.read_text(encoding="utf-8")


# ── the deliverer on the group's meta ────────────────────────────────────


async def test_owners_name_and_clear_the_deliverer_and_the_read_shows_it(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await _group(world, deliverer=None)
        assert (await _read(world, "u_betty")).deliverer_id is None
        assert "deliverer_id" not in _group_md(world)

        with pytest.raises(ForbiddenError):
            await _upsert(world, "u_betty", deliverer_id=BOT)
        await _upsert(world, "u_chen", deliverer_id=BOT)
        assert (await _read(world, "u_betty")).deliverer_id == BOT
        assert f"deliverer_id: {BOT}" in _group_md(world)

        # A wiped mirror is rebuilt from group.md with the deliverer.
        await world.storage.sqlite.group_roster.remove(APP, PROJECT, "trip")
        record = await world.stores.groups.get(world.scope("trip"))
        assert record is not None and record.group.deliverer_id == BOT
        assert record.roster.deliverer_id == BOT

        # Left out: kept. Cleared: gone from the read and from group.md.
        await _upsert(world, "u_chen", name="Trip 2")
        assert (await _read(world, "u_betty")).deliverer_id == BOT
        await _upsert(world, "u_chen", clear_deliverer=True)
        assert (await _read(world, "u_betty")).deliverer_id is None
        assert "deliverer_id: null" in _group_md(world)
        await world.storage.sqlite.group_roster.remove(APP, PROJECT, "trip")
        record = await world.stores.groups.get(world.scope("trip"))
        assert record is not None and record.group.deliverer_id is None


async def test_a_group_a_batch_creates_has_no_deliverer(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=_llm()) as world:
        await _add(
            world, _batch("s1", ("m-1", "u_chen", "hello")) | {"group_id": "new"}
        )
        record = await world.stores.groups.get(world.scope("new"))
        assert record is not None and record.group.deliverer_id is None
        assert "deliverer_id" not in _group_md(world, "new")


async def test_the_deliverer_changes_the_meta_and_rotates_its_own_key(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await _group(world)
        await _upsert(world, BOT, history_visible_from="creation", name="Trip team")
        view = await _read(world, BOT)
        assert (view.name, view.history_visible_from) == ("Trip team", "creation")
        assert BOT not in {m["member_id"] for m in view.roster}

        await _upsert(world, BOT, deliverer_id="gw_trip_2")
        with pytest.raises(ForbiddenError):  # the old key is nobody now
            await _upsert(world, BOT, name="x")
        with pytest.raises(ForbiddenError):
            await _read(world, BOT)
        assert (await _read(world, "gw_trip_2")).deliverer_id == "gw_trip_2"


async def test_the_deliverer_sends_any_roster_event_backdated_too(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await _group(world)
        changed = await world.memory.change_roster(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id=BOT,
            asserted_asker=None,
            events=[
                world.roster_input(op="join", member_id="u_fay", at=T0 + 100),
                world.roster_input(op="leave", member_id="u_betty", at=T0 + 6000),
                world.roster_input(op="update", member_id="u_evan", role="admin"),
            ],
        )
        assert changed.events_applied == 3
        with pytest.raises(ForbiddenError):  # a member may not backdate
            await world.memory.change_roster(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_fay",
                asserted_asker=None,
                events=[world.roster_input(op="join", member_id="u_gus", at=T0)],
            )


# ── what the deliverer reads ─────────────────────────────────────────────


async def test_the_deliverer_reads_the_whole_history_and_answers_as_the_group(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await _group(world)
        picture = await _seed_records(world)

        # A search naming no audience: what every current member can see.
        assert await _search(world, BOT) == {LATE}
        scoped = await world.memory.search(
            SearchRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id=BOT,
                query="marker",
            )
        )
        assert scoped.data.scope is not None and scoped.data.scope.audience == "group"
        # The full superadmin view, asked for.
        assert await _search(world, BOT, audience="asker") == {EARLY, LATE}
        assert await _search(world, BOT, audience="group") == {LATE}

        listed = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id=BOT,
                memory_type="episode",
            )
        )
        assert {e["episode"] for e in listed.data.model_dump()["episodes"]} == {
            EARLY,
            LATE,
        }
        early_id = next(
            e["id"]
            for e in listed.data.model_dump()["episodes"]
            if e["episode"] == EARLY
        )
        history = await world.memory.memory_history(
            app_id=APP,
            project_id=PROJECT,
            record_id=early_id,
            group_id="trip",
            asker_id=BOT,
        )
        assert history.current == early_id
        assert await world.memory.picture_visible(
            image_id=picture, asker_id=BOT, asserted_asker=None
        )
        assert not await world.memory.picture_visible(
            image_id=picture, asker_id="u_evan", asserted_asker=None
        )
        assert (await _read(world, BOT)).group_id == "trip"


async def test_the_members_read_what_they_read_before(tmp_path: Path) -> None:
    seen: dict[str | None, dict[str, set[str]]] = {}
    for deliverer in (None, BOT):
        async with GroupWorld.open(tmp_path / str(deliverer)) as world:
            await _group(world, deliverer=deliverer)
            await _seed_records(world)
            seen[deliverer] = {
                asker: await _search(world, asker)
                for asker in ("u_chen", "u_betty", "u_evan")
            } | {
                f"{asker}+group": await _search(world, asker, audience="group")
                for asker in ("u_chen", "u_evan")
            }
    assert seen[None] == seen[BOT]
    assert seen[BOT]["u_chen"] == {EARLY, LATE}
    assert seen[BOT]["u_evan"] == {LATE}


async def test_the_deliverer_reads_nothing_of_another_group(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await _group(world)
        await world.roster("other", [join("u_zoe", T0, role="owner")])
        await world.episode("other", LATE, start=T0 + 8000, end=T0 + 9000)
        await world.project()
        for read in (
            lambda: _search(world, BOT, group_id="other"),
            lambda: _search(world, BOT, group_id="other", audience="asker"),
            lambda: _read(world, BOT, group="other"),
            lambda: world.memory.get(
                GetRequest(
                    app_id=APP,
                    project_id=PROJECT,
                    group_id="other",
                    asker_id=BOT,
                    memory_type="episode",
                )
            ),
            lambda: world.memory.forget_in_group(
                app_id=APP,
                project_id=PROJECT,
                group_id="other",
                asker_id=BOT,
                target=ForgetTarget(kind="member", member_id="u_zoe"),
                dry_run=True,
            ),
        ):
            with pytest.raises(ForbiddenError):
                await read()


# ── what the deliverer changes ───────────────────────────────────────────


async def test_the_deliverer_forgets_undoes_edits_and_deletes(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=_llm()) as world:
        await _group(world)
        await _add(
            world,
            _batch(
                "s1",
                ("m-1", "u_chen", "Lisbon plan: fly on Friday"),
                ("m-2", "u_betty", "Lisbon plan: I book the hotel"),
                asker_id=BOT,
            ),
        )
        await _flush(world, "s1", BOT)
        await world.project()

        forgot = await world.memory.forget_in_group(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id=BOT,
            target=ForgetTarget(kind="message", session_id="s1", message_ids=("m-2",)),
        )
        assert forgot.forget_id
        undone = await world.memory.undo_group_forget(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id=BOT,
            forget_id=forgot.forget_id,
        )
        assert undone.applied and undone.forget_id == forgot.forget_id
        await world.project()

        listed = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id=BOT,
                memory_type="episode",
            )
        )
        (episode,) = listed.data.model_dump()["episodes"]
        edited = await world.memory.update_memory(
            app_id=APP,
            project_id=PROJECT,
            record_id=episode["id"],
            changes=MemoryChanges(summary="The group's plan"),
            group_id="trip",
            asker_id=BOT,
        )
        assert edited.changed

        deleted, forgotten = await world.memory.delete_in_group(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id=BOT,
            session_id="s1",
            member_id=None,
        )
        assert forgotten.forget_id and deleted.filters == ["group_id", "session_id"]


# ── delivering ───────────────────────────────────────────────────────────


async def test_once_a_group_has_a_deliverer_only_it_delivers(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=_llm()) as world:
        await _group(world)
        session = world.session("s1")
        for asker in ("u_chen", None, "gw_other"):
            batch = _batch("s1", ("m-1", "u_chen", "hello"))
            if asker is not None:
                batch["asker_id"] = asker
            for flags in ({}, {"async_mode": True}, {"defer": False}):
                with pytest.raises(ForbiddenError, match="only the deliverer of group"):
                    await _add(world, batch, **flags)
        # Refused before anything: no binding, no buffer, nothing queued.
        assert await world.stores.groups.binding(session) is None
        assert await world.stores.session_buffer.list(session) == []
        assert not await world.memory.has_queued(
            session_id="s1", app_id=APP, project_id=PROJECT
        )

        result = await _add(
            world, _batch("s1", ("m-1", "u_chen", "hello"), asker_id=BOT)
        )
        assert result.message_count == 1
        with pytest.raises(ForbiddenError):
            await _flush(world, "s1", "u_chen")
        with pytest.raises(ForbiddenError):
            await _flush(world, "s1", None)
        flushed = await _flush(world, "s1", BOT)
        assert flushed.status == "extracted"

        # Clear the deliverer: anyone delivers again, as before.
        await _upsert(world, "u_chen", clear_deliverer=True)
        await _add(world, _batch("s2", ("m-9", "u_betty", "hi")))
        await _flush(world, "s2", None)


async def test_a_one_to_one_session_is_untouched(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=_llm()) as world:
        await _group(world)
        payload = {
            "session_id": "solo",
            "app_id": APP,
            "project_id": PROJECT,
            "messages": [
                {
                    "sender_id": "u_chen",
                    "role": "user",
                    "timestamp": T0,
                    "content": "hi",
                }
            ],
        }
        result = await _add(world, payload)
        assert result.message_count == 1 and result.duplicates == 0
        assert (await _flush(world, "solo", None)).status == "extracted"


# ── over HTTP ────────────────────────────────────────────────────────────


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[AsyncClient]:
    async with GroupWorld.open(tmp_path, llm=_llm()) as world:
        app = create_app(application=make_application(memory=world.memory))
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            yield c


async def test_the_wire_names_clears_and_enforces_the_deliverer(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = {"project_id": PROJECT, "asker_id": "u_chen"}
    created = await client.put(
        "/api/v2/groups/trip", json=base | {"deliverer_id": BOT, "created_at": T0}
    )
    assert created.status_code == 200, created.text
    assert created.json()["data"]["group"]["deliverer_id"] == BOT
    kept = await client.put("/api/v2/groups/trip", json=base | {"name": "Trip"})
    assert kept.json()["data"]["group"]["deliverer_id"] == BOT

    batch = _batch("s1", ("m-1", "u_chen", "hello")) | {"asker_id": "u_chen"}
    refused = await client.post("/api/v2/memory/add", json=batch)
    assert refused.status_code == 403
    assert "only the deliverer of group 'trip'" in refused.json()["error"]["message"]
    batch["asker_id"] = BOT
    delivered = await client.post("/api/v2/memory/add", json=batch)
    assert delivered.status_code == 200, delivered.text
    assert "duplicates" not in delivered.json()["data"]
    again = await client.post("/api/v2/memory/add", json=batch)
    assert again.json()["data"]["duplicates"] == 1

    # The session is the group's now: flushed by its deliverer only.
    flush = {"session_id": "s1", "project_id": PROJECT}
    assert (await client.post("/api/v2/memory/flush", json=flush)).status_code == 403
    member = flush | {"asker_id": "u_chen"}
    assert (await client.post("/api/v2/memory/flush", json=member)).status_code == 403

    # Behind a gateway: the header is the asker.
    monkeypatch.setenv("EVEROS_ACCESS__IDENTITY_HEADER", "X-Asker")
    load_settings.cache_clear()
    batch.pop("asker_id")
    by_member = await client.post(
        "/api/v2/memory/add", json=batch, headers={"X-Asker": "u_chen"}
    )
    assert by_member.status_code == 403
    flushed = await client.post(
        "/api/v2/memory/flush", json=flush, headers={"X-Asker": BOT}
    )
    assert flushed.status_code == 200, flushed.text

    cleared = await client.put(
        "/api/v2/groups/trip",
        json={"project_id": PROJECT, "deliverer_id": None},
        headers={"X-Asker": BOT},
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["data"]["group"]["deliverer_id"] is None
