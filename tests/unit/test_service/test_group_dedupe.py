"""A group takes each message once (docs/group-memory.md, section 5.2).

Real markdown, ``system.db`` and LanceDB behind the bootstrap's memory
service, a scripted model that records every call (``tests/helpers/groups.py``).
What these pin:

- a message the group already holds -- by ``message_id``, else by sender,
  time and content -- is skipped, in any session of the group, buffered or
  cut: the answer counts it (``duplicates``) and no model call is made for
  it; a batch of nothing but duplicates makes none at all;
- an explicit group and an implicit group of three people dedupe; a session
  of one person, or of two, never does;
- a retry of the same batch is safe; two batches racing with the same
  message store it once; a batch that failed recorded nothing, so its retry
  is taken in full;
- a forgotten message sent again stays forgotten, in its session or another;
- a ``/memory/delete`` of a session releases it: sending it again is taken as
  before dedupe, the session it lands in holds it from then on, and an undo
  of the delete makes the old session hold it again.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient

from everos.service import ForgetTarget, GetRequest
from tests.helpers.groups import APP, PROJECT, GroupWorld, join

T0 = 1_772_323_200_000


class _Model:
    """Boundary never cuts (a flush closes the segment); episodes scripted."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_boundary = 0
        self.fail_episode = 0
        self.close_tail = False

    def __call__(self, messages: Any, *, response_format: type, **_kw: Any) -> str:
        name = response_format.__name__
        self.calls.append(name)
        if name == "_BatchReply":
            if self.fail_boundary:
                self.fail_boundary -= 1
                raise RuntimeError("boundary model down")
            return json.dumps(
                {"reasoning": "s", "boundaries": [], "tail_complete": self.close_tail}
            )
        if name == "EpisodeReply":
            if self.fail_episode:
                self.fail_episode -= 1
                raise RuntimeError("episode model down")
            return json.dumps({"title": "T", "summary": "S", "content": "C"})
        return json.dumps({"operations": []})


def _message(sender: str, at: int, text: str, mid: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {
        "sender_id": sender,
        "role": "user",
        "timestamp": T0 + at,
        "content": text,
    }
    if mid is not None:
        message["message_id"] = mid
    return message


def _batch(session: str, *messages: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {
        "session_id": session,
        "app_id": APP,
        "project_id": PROJECT,
        "messages": list(messages),
        **fields,
    }


async def _add(
    world: GroupWorld, payload: dict[str, Any], *, defer: bool = True
) -> Any:
    return await world.memory.add(
        payload, request_id="r", async_mode=False, defer_extraction=defer
    )


async def _flush(world: GroupWorld, session: str) -> Any:
    return await world.memory.flush(
        session_id=session, request_id="f", app_id=APP, project_id=PROJECT
    )


async def _held_texts(world: GroupWorld, *sessions: str) -> list[str]:
    """Every message the sessions hold, buffered or cut, in order."""
    texts: list[str] = []
    for session in sessions:
        key = world.session(session)
        for cell in (await world.stores.memcell.cells_in_session(key)).values():
            texts.extend(str(item.content) for item in cell.items)
        texts.extend(m.text for m in await world.stores.session_buffer.list(key))
    return texts


async def _trip(world: GroupWorld) -> None:
    await world.roster(
        "trip",
        [join("u_chen", T0, role="owner"), join("u_betty", T0), join("u_dan", T0)],
    )


async def test_a_message_the_group_holds_is_skipped_in_any_session(
    tmp_path: Path,
) -> None:
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        await _trip(world)
        first = await _add(
            world,
            _batch(
                "s1",
                _message("u_chen", 1, "fly Friday", "m-1"),
                _message("u_betty", 2, "book the hotel", "m-2"),
                group_id="trip",
            ),
        )
        assert first.duplicates == 0
        await _flush(world, "s1")  # m-1 and m-2 cut into a segment

        # Another client forwards the same chat into another session.
        second = await _add(
            world,
            _batch(
                "s2",
                _message("u_betty", 2, "book the hotel", "m-2"),
                _message("u_dan", 3, "I drive", "m-3"),
                group_id="trip",
            ),
        )
        assert (second.message_count, second.duplicates) == (2, 1)
        # By content when no id was sent: same sender, time and text.
        third = await _add(
            world,
            _batch(
                "s2",
                _message("u_dan", 4, "see you there"),
                _message("u_dan", 4, "see you there"),  # a batch is taken as sent
                _message("u_chen", 5, "ok"),
            ),
        )
        assert third.duplicates == 0
        fourth = await _add(
            world,
            _batch(
                "s3",
                _message("u_dan", 4, "see you there"),
                _message("u_chen", 5, "ok!"),  # other content: a new message
                group_id="trip",
            ),
        )
        assert fourth.duplicates == 1
        assert await _held_texts(world, "s1", "s2", "s3") == [
            "fly Friday",
            "book the hotel",
            "I drive",
            "see you there",
            "see you there",
            "ok",
            "ok!",
        ]


async def test_a_retry_is_safe_and_makes_no_model_call(tmp_path: Path) -> None:
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        await _trip(world)
        batch = _batch(
            "s1",
            _message("u_chen", 1, "fly Friday", "m-1"),
            _message("u_betty", 2, "book the hotel"),
            group_id="trip",
        )
        await _add(world, batch, defer=False)
        calls = len(model.calls)
        retried = await _add(world, batch, defer=False)
        assert (retried.duplicates, retried.status) == (2, "accumulated")
        assert len(model.calls) == calls  # no boundary call for nothing new
        await _flush(world, "s1")
        assert await _held_texts(world, "s1") == ["fly Friday", "book the hotel"]


async def test_a_failed_batch_records_nothing(tmp_path: Path) -> None:
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        await _trip(world)
        batch = _batch(
            "s1", _message("u_chen", 1, "fly Friday", "m-1"), group_id="trip"
        )
        model.fail_boundary = 10
        with pytest.raises(RuntimeError, match="boundary model down"):
            await _add(world, batch, defer=False)
        model.fail_boundary = 0
        retried = await _add(world, batch, defer=False)
        assert retried.duplicates == 0
        assert await _held_texts(world, "s1") == ["fly Friday"]


async def test_a_batch_whose_extraction_failed_is_taken_again(tmp_path: Path) -> None:
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        await _trip(world)
        model.close_tail = True  # every batch is a segment, extracted at once
        batch = _batch(
            "s1", _message("u_chen", 1, "fly Friday", "m-1"), group_id="trip"
        )
        model.fail_episode = 10
        with pytest.raises(RuntimeError, match="episode model down"):
            await _add(world, batch, defer=False)
        model.fail_episode = 0
        retried = await _add(world, batch, defer=False)
        assert (retried.duplicates, retried.status) == (0, "extracted")
        assert (await _add(world, batch, defer=False)).duplicates == 1


async def test_two_batches_racing_with_one_message_store_it_once(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await _trip(world)
        results = await asyncio.gather(
            *(
                _add(
                    world,
                    _batch(
                        session,
                        _message("u_chen", 1, "fly Friday", "m-1"),
                        group_id="trip",
                    ),
                )
                for session in ("s1", "s2", "s3")
            )
        )
        assert sorted(r.duplicates for r in results) == [0, 1, 1]
        assert await _held_texts(world, "s1", "s2", "s3") == ["fly Friday"]


async def test_an_implicit_group_of_three_dedupes_two_people_do_not(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as world:
        three = [
            _message("u_ann", 1, "Lisbon?", "p-1"),
            _message("u_bob", 2, "Yes", "p-2"),
            _message("u_cat", 3, "Count me in", "p-3"),
        ]
        await _add(world, _batch("t1", *three))
        again = await _add(world, _batch("t2", *three))  # the same three people
        assert again.duplicates == 3
        assert await world.stores.groups.binding(world.session("t1")) == (
            await world.stores.groups.binding(world.session("t2"))
        )

        two = [_message("u_eve", 1, "hi", "q-1"), _message("u_fox", 2, "hey", "q-2")]
        await _add(world, _batch("u1", *two))
        repeated = await _add(world, _batch("u2", *two))
        assert repeated.duplicates == 0

        solo = [_message("u_gus", 1, "note to self", "r-1")]
        await _add(world, _batch("v1", *solo))
        assert (await _add(world, _batch("v1", *solo))).duplicates == 0
        assert (await _add(world, _batch("v2", *solo))).duplicates == 0


async def test_a_forgotten_message_sent_again_stays_forgotten(tmp_path: Path) -> None:
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        await _trip(world)
        await _add(
            world,
            _batch(
                "s1",
                _message("u_chen", 1, "fly Friday", "m-1"),
                _message("u_betty", 2, "my card number is secret", "m-2"),
                group_id="trip",
            ),
        )
        await _flush(world, "s1")
        await world.memory.forget_in_group(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_betty",
            target=ForgetTarget(kind="message", session_id="s1", message_ids=("m-2",)),
        )
        for session in ("s1", "s2"):
            again = await _add(
                world,
                _batch(
                    session,
                    _message("u_betty", 2, "my card number is secret", "m-2"),
                    group_id="trip",
                ),
            )
            assert again.duplicates == 1
        await _flush(world, "s2")
        await world.project()
        listed = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_chen",
                memory_type="episode",
            )
        )
        # Only s1's segment, written again without the forgotten message.
        assert listed.data.total_count == 1
        assert await _held_texts(world, "s2") == []


async def test_a_deleted_session_sent_again_is_taken(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as world:
        await _trip(world)
        batch = _batch(
            "s1",
            _message("u_chen", 1, "fly Friday", "m-1"),
            _message("u_betty", 2, "book the hotel"),
            group_id="trip",
        )
        await _add(world, batch)
        await _flush(world, "s1")
        _, deleted = await world.memory.delete_in_group(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_chen",
            session_id="s1",
            member_id=None,
        )

        # Sent again after its delete: taken, as it was before dedupe.
        assert (await _add(world, batch)).duplicates == 0
        # Into another session: taken, and held there from then on.
        moved = batch | {"session_id": "s2"}
        assert (await _add(world, moved)).duplicates == 0
        assert (await _add(world, moved)).duplicates == 2

        # An undone delete holds its session again.
        await world.memory.undo_group_forget(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_chen",
            forget_id=deleted.forget_id,
        )
        other = _batch("s1", _message("u_dan", 9, "I drive", "m-9"), group_id="trip")
        await _add(world, other)
        _, again = await world.memory.delete_in_group(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_chen",
            session_id="s1",
            member_id=None,
        )
        await world.memory.undo_group_forget(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_chen",
            forget_id=again.forget_id,
        )
        assert (await _add(world, other | {"session_id": "s3"})).duplicates == 1


async def test_a_one_to_one_session_never_dedupes(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as world:
        batch = _batch(
            "solo",
            _message("u_amy", 1, "Lisbon trip", "x-1"),
            {**_message("bot", 2, "Noted."), "role": "assistant"},
        )
        for _ in range(2):
            result = await _add(world, batch, defer=False)
            assert result.duplicates == 0
        assert await world.stores.groups.binding(world.session("solo")) is None
