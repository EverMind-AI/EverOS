"""The one forget engine: what it keeps, the order its undos take, old logs.

Real markdown, ``system.db`` and LanceDB (``tests/helpers/groups.py``). What
these pin, for a user's memory and a group's alike:

- a forget marks: every version it reaches stays in markdown, marked, and the
  version written in its place supersedes it;
- profile items come back in their places, by id;
- an undo is refused while a later change replaced what its forget wrote,
  and goes through once that change is undone;
- a user's message forget re-extracts the segment without the message, and a
  buffered message forgotten before its segment was cut is never read;
- in two people's conversation each forget acts on the asker's own copies
  only, and a message forget is refused (it would change the other person's
  memory, read from the same segments and buffer);
- forgets logged before this engine -- a user's in the log's columns or in an
  ``undo`` of records, a group's that took records out of markdown -- still
  undo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage

from everos.core.errors import ConflictError, InvalidInputError
from everos.memory.models import FORGOTTEN_MARK, Episode, ExplicitInfoItem, Profile
from everos.memory.ports import NoSourceCheck
from everos.service import ForgetTarget, SearchRequest
from tests.helpers.groups import APP, PROJECT, GroupWorld, join

T0 = 1_772_323_200_000
SECRET = "OTTERSECRET7"
KEPT = "HERONKEPT3"
BUFFERED = "BADGERLATER5"
_MARKERS = (SECRET, KEPT, BUFFERED)


class _Model:
    """Episodes repeat the markers they were shown; the decider keeps all."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, messages: Any, *, response_format: type, **_: Any) -> str:
        name = response_format.__name__
        prompt = "\n".join(str(getattr(m, "content", m)) for m in messages)
        self.calls.append(name)
        if name == "_BatchReply":
            return json.dumps(
                {"reasoning": "s", "boundaries": [], "tail_complete": False}
            )
        if name == "EpisodeReply":
            said = ", ".join(m for m in _MARKERS if m in prompt) or "nothing"
            return json.dumps(
                {"title": "Notes", "summary": f"Notes: {said}.", "content": said}
            )
        if name == "_DecisionReply":
            return json.dumps({"core": list(range(10)), "next_queries": []})
        if name == "_ProfileUpdateReply":
            return json.dumps({"operations": []})
        raise AssertionError(f"unexpected model call {name}")


async def _open(tmp_path: Path, model: _Model) -> Any:
    client = FakeLLMClient(handler=model)
    return GroupWorld.open(tmp_path, llm=client, decider=client)


async def _forget(world: GroupWorld, target: ForgetTarget, **kw: Any) -> Any:
    return await world.memory.forget(
        app_id=APP,
        project_id=PROJECT,
        user_id="amy",
        target=target,
        dry_run=False,
        **kw,
    )


async def _undo(world: GroupWorld, forget_id: str) -> Any:
    return await world.memory.undo_forget(
        app_id=APP, project_id=PROJECT, user_id="amy", forget_id=forget_id
    )


async def _marks(world: GroupWorld) -> dict[str, str]:
    marks: dict[str, str] = {}
    for stored, mark in await world.stores.episode.read_all(world.personal("amy")):
        if mark:
            marks[stored.ref.entry_id] = mark
    return marks


def _item(item_id: str, text: str) -> ExplicitInfoItem:
    return ExplicitInfoItem(description=text, category="facts", item_id=item_id)


async def test_profile_items_come_back_in_their_places(tmp_path: Path) -> None:
    async with await _open(tmp_path, _Model()) as world:
        scope = world.personal("amy")
        await world.stores.profile.write(
            scope,
            Profile(
                owner_id="amy",
                summary="",
                timestamp=T0,
                explicit_info=[
                    _item("pi_000000000001", "Lives in Lyon"),
                    _item("pi_000000000002", "Likes trains"),
                    _item("pi_000000000003", "Reads at night"),
                ],
            ),
        )
        done = await _forget(
            world, ForgetTarget(kind="record", record_ids=("pi_000000000002",))
        )
        assert done.profile_items == ("pi_000000000002",)

        await _undo(world, done.forget_id)

        held = await world.stores.profile.read(scope)
        assert held is not None
        # Back where it was, not at the end (an old row puts it by description).
        assert [i.description for i in held.explicit_info] == [
            "Lives in Lyon",
            "Likes trains",
            "Reads at night",
        ]


async def test_a_forget_marks_and_its_undo_writes_the_version_again(
    tmp_path: Path,
) -> None:
    async with await _open(tmp_path, _Model()) as world:
        scope = world.personal("amy")
        picture = await world.picture(scope, "a red kite", at=T0, shade=40)
        episode = await world.personal_episode(
            "amy", f"Flew the red kite (img-1: {picture}) at the beach.", at=T0
        )
        original = episode.removeprefix("amy_")

        forgot = await _forget(world, ForgetTarget(kind="image", image_id=picture))
        (rewrite,) = forgot.new_episodes
        marks = await _marks(world)
        # Nothing left markdown: the old version is there, superseded.
        assert marks[original] == rewrite.removeprefix("amy_")
        stored = await world.stores.episode.read_many(scope, [original])
        assert stored and picture in stored[0].item.episode

        undone = await _undo(world, forgot.forget_id)
        (copy,) = undone.episodes
        assert undone.retired == (rewrite,)
        marks = await _marks(world)
        assert marks[rewrite.removeprefix("amy_")] == copy.removeprefix("amy_")
        (back,) = await world.stores.episode.read_many(
            scope, [copy.removeprefix("amy_")]
        )
        assert back.item.episode == stored[0].item.episode


async def test_an_undo_waits_for_the_later_forget_that_holds_its_writes(
    tmp_path: Path,
) -> None:
    """The order rule: undo the later forget first."""
    async with await _open(tmp_path, _Model()) as world:
        scope = world.personal("amy")
        picture = await world.picture(scope, "a red kite", at=T0, shade=41)
        await world.personal_episode(
            "amy", f"Flew the red kite (img-1: {picture}).", at=T0
        )
        first = await _forget(world, ForgetTarget(kind="image", image_id=picture))
        (rewrite,) = first.new_episodes
        later = await _forget(world, ForgetTarget(kind="record", record_ids=(rewrite,)))

        with pytest.raises(ConflictError):
            await _undo(world, first.forget_id)

        assert (await _undo(world, later.forget_id)).applied
        assert (await _undo(world, first.forget_id)).applied
        live = [
            s.item.episode
            for s, mark in await world.stores.episode.read_all(scope)
            if mark is None
        ]
        assert live == [f"Flew the red kite (img-1: {picture})."]


async def _send(
    world: GroupWorld, *lines: tuple[str, str, int], flush: bool = True
) -> None:
    await world.memory.add(
        {"session_id": "s_amy", "app_id": APP, "project_id": PROJECT,
         "messages": [
             {"sender_id": who, "role": "user" if who != "bot" else "assistant",
              "timestamp": T0 + at, "content": text}
             for who, text, at in lines
         ]},
        request_id="r",
        async_mode=False,
        defer_extraction=True,
    )  # fmt: skip
    if flush:
        await world.memory.flush(
            session_id="s_amy", request_id="f", app_id=APP, project_id=PROJECT
        )


async def _message_ids(world: GroupWorld) -> dict[str, str]:
    """Each message's id in the session by its text, segments then buffer."""
    found: dict[str, str] = {}
    session = world.session("s_amy")
    for cell in (await world.stores.memcell.cells_in_session(session)).values():
        for item in cell.items:
            if isinstance(item, ChatMessage):
                found[str(item.content)] = item.id
    for message in await world.stores.session_buffer.list(session):
        found[message.text] = message.client_message_id or message.message_id
    return found


async def test_a_users_message_forget_reextracts_and_skips_the_buffer(
    tmp_path: Path,
) -> None:
    model = _Model()
    async with await _open(tmp_path, model) as world:
        await _send(
            world,
            ("amy", f"The safe code is {SECRET}", 1_000),
            ("amy", f"I keep the {KEPT} by the door", 2_000),
            ("bot", "Noted.", 3_000),
        )
        await world.project()
        ids = await _message_ids(world)
        model.calls.clear()

        target = ForgetTarget(
            kind="message",
            session_id="s_amy",
            message_ids=(ids[f"The safe code is {SECRET}"],),
        )
        done = await _forget(world, target)
        assert model.calls == ["EpisodeReply"]  # the segment, once, without it
        assert done.rewritten == 1 and len(done.new_episodes) == 1
        await world.project()

        # A message forgotten while it waits in the buffer is never read.
        await _send(world, ("amy", f"Later: {BUFFERED}", 4_000), flush=False)
        buffered = (await _message_ids(world))[f"Later: {BUFFERED}"]
        await _forget(
            world,
            ForgetTarget(kind="message", session_id="s_amy", message_ids=(buffered,)),
        )
        found = await world.memory.search(
            SearchRequest(
                app_id=APP,
                project_id=PROJECT,
                user_id="amy",
                query="safe code door",
                filters={"session_id": "s_amy"},
            )
        )
        shown = json.dumps(found.data.model_dump(mode="json"))
        assert SECRET not in shown and BUFFERED not in shown
        assert KEPT in shown
        await world.memory.flush(
            session_id="s_amy", request_id="f", app_id=APP, project_id=PROJECT
        )
        live = [
            s.item.episode
            for s, mark in await world.stores.episode.read_all(world.personal("amy"))
            if mark is None
        ]
        assert live == [KEPT]  # the buffered message made no memory

        undone = await _undo(world, done.forget_id)
        assert undone.applied and len(undone.episodes) == 1
        live = [
            s.item.episode
            for s, mark in await world.stores.episode.read_all(world.personal("amy"))
            if mark is None
        ]
        assert live == [f"{SECRET}, {KEPT}"]


async def test_two_peoples_forgets_act_on_the_askers_copies_only(
    tmp_path: Path,
) -> None:
    model = _Model()
    async with await _open(tmp_path, model) as world:
        await _send(
            world,
            ("amy", f"The safe code is {SECRET}", 1_000),
            ("ben", f"I keep the {KEPT} by the door", 2_000),
        )
        await world.project()
        ids = await _message_ids(world)
        model.calls.clear()

        message = ForgetTarget(
            kind="message",
            session_id="s_amy",
            message_ids=(ids[f"The safe code is {SECRET}"],),
        )
        with pytest.raises(InvalidInputError, match="two people"):
            await _forget(world, message)
        assert model.calls == []
        assert (
            await world.stores.message_forget.forgotten(world.session("s_amy"))
            == frozenset()
        )

        done = await _forget(world, ForgetTarget(kind="session", session_id="s_amy"))
        assert len(done.episodes) == 1 and model.calls == []
        for person, alive in (("amy", []), ("ben", [f"{SECRET}, {KEPT}"])):
            live = [
                s.item.episode
                for s, mark in await world.stores.episode.read_all(
                    world.personal(person)
                )
                if mark is None
            ]
            assert live == alive, person
        undone = await _undo(world, done.forget_id)
        assert undone.applied


# ── forgets logged before the one engine ─────────────────────────────────


async def test_a_users_old_conversation_forget_still_undoes(tmp_path: Path) -> None:
    """Logged in the columns; its profile item comes back by description."""
    async with await _open(tmp_path, _Model()) as world:
        scope = world.personal("amy")
        episode = (await world.personal_episode("amy", "old talk", at=T0)).removeprefix(
            "amy_"
        )
        (stored,) = await world.stores.episode.read_many(scope, [episode])
        await world.stores.episode.deprecate(
            scope,
            entries=[(stored.ref.md_path, episode)],
            replaced_by=FORGOTTEN_MARK,
            only_if=NoSourceCheck(),
        )
        await world.stores.profile.write(
            scope,
            Profile(
                owner_id="amy",
                summary="",
                timestamp=T0,
                explicit_info=[_item("pi_000000000009", "Kept fact")],
            ),
        )
        old = await world.stores.forget_log.record(
            target="session",
            target_id="s_amy",
            app_id=APP,
            project_id=PROJECT,
            owner_id="amy",
            owner_type="user",
            episode_ids=((episode, ""),),
            removed_profile_items=(
                ("explicit_info", {"description": "Gone fact", "category": "facts"}),
            ),
        )

        undone = await _undo(world, old.forget_id)

        assert undone.applied and undone.episodes == (f"amy_{episode}",)
        assert await _marks(world) == {}
        held = await world.stores.profile.read(scope)
        assert held is not None
        assert [i.description for i in held.explicit_info] == ["Kept fact", "Gone fact"]


async def test_a_users_old_record_forget_still_undoes(tmp_path: Path) -> None:
    async with await _open(tmp_path, _Model()) as world:
        scope = world.personal("amy")
        episode = (await world.personal_episode("amy", "old talk", at=T0)).removeprefix(
            "amy_"
        )
        (stored,) = await world.stores.episode.read_many(scope, [episode])
        await world.stores.episode.deprecate(
            scope,
            entries=[(stored.ref.md_path, episode)],
            replaced_by=FORGOTTEN_MARK,
            only_if=NoSourceCheck(),
        )
        old = await world.stores.forget_log.record(
            target="record",
            target_id=f"amy_{episode}",
            app_id=APP,
            project_id=PROJECT,
            owner_id="amy",
            owner_type="user",
            undo={
                "target": {"kind": "record", "record_ids": [f"amy_{episode}"]},
                "episodes": [[stored.ref.md_path, episode]],
                "image_memories": [],
                "scenes": [],
                "orphaned": [],
                "clusters": [],
                "files": [],
                "segments": [],
                "segment_items": [],
            },
        )

        undone = await _undo(world, old.forget_id)

        assert undone.applied and undone.episodes == (f"amy_{episode}",)
        assert await _marks(world) == {}


async def test_a_groups_old_forget_that_took_records_out_still_undoes(
    tmp_path: Path,
) -> None:
    """Such a forget took the episode out of markdown; its undo puts it back
    under its own id and takes out what the forget wrote."""
    async with await _open(tmp_path, _Model()) as world:
        await world.roster("trip", [join("u_chen", 0), join("u_dan", 0)])
        scope = world.scope("trip")
        old_id = (
            await world.episode("trip", "the plan", start=10, end=20)
        ).removeprefix("trip_")
        written = await world.stores.episode.append(
            scope,
            Episode(
                owner_id="trip",
                episode="the plan, without it",
                summary="the plan",
                timestamp=20,
                session_id="s_group",
                parent_id="mc_trip_10",
            ),
            only_if=NoSourceCheck(),
        )
        assert written is not None
        taken = await world.stores.episode.withdraw(scope, [old_id])
        old = await world.stores.forget_log.record(
            target="message",
            target_id="s_group:m-1",
            app_id=APP,
            project_id=PROJECT,
            owner_id="trip",
            owner_type="group",
            requested_by="u_chen",
            undo={
                "target": {"kind": "message", "session_id": "s_group"},
                "sources": ["u_chen"],
                "messages": [],
                "written": {"episode": [written.entry_id], "scene": []},
                "withdrawn": {
                    "episode": {
                        "entries": [
                            {
                                "entry_id": e.entry_id,
                                "md_path": e.md_path,
                                "text": e.text,
                                "mark": e.mark,
                            }
                            for e in taken.entries
                        ],
                        "released": [list(r) for r in taken.released],
                    }
                },
                "pictures": [],
                "members_removed": [],
                "member_items": [],
                "group_queued": [],
                "files": [],
                "clusters": [],
            },
        )

        undone = await world.memory.undo_group_forget(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_chen",
            forget_id=old.forget_id,
        )

        assert undone.applied
        assert undone.episodes == (f"trip_{old_id}",)
        assert undone.retired == (f"trip_{written.entry_id}",)
        present = {
            s.ref.entry_id for s, _ in await world.stores.episode.read_all(scope)
        }
        assert present == {old_id}
