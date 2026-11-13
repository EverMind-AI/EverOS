"""A user's forgotten message is read again by nothing.

docs/api.md, ``/memory/forget``, ``target.kind: "message"``: a buffered
message forgotten before its segment was cut is read by no model call and
written nowhere, and its undo makes it an ordinary message again; a fact
stated in a forgotten message leaves the user's profile -- the items resting
on its segment go and are written again by the next update from what the
segment still says, without the message -- and the undo puts them back.
Real markdown, ``system.db`` and LanceDB (``tests/helpers/forgets.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

from everos.service import ForgetTarget, SearchRequest
from tests.helpers.forgets import (
    ForgetModel,
    forget,
    items,
    live,
    message_ids,
    open_world,
    profile_update,
    send,
    undo,
)
from tests.helpers.groups import APP, PROJECT


async def test_a_forgotten_messages_fact_leaves_the_profile_and_is_written_again(
    tmp_path: Path,
) -> None:
    model = ForgetModel()
    async with open_world(tmp_path, model) as world:
        await send(
            world,
            "p1",
            ("amy", "My door code is ZQFPIN20, keep it.", 1_000),
            ("amy", "Unrelated: I like ZQFHIKE21.", 1_001),
            ("bot", "Ok.", 1_002),
        )
        await profile_update(world)
        assert sorted(await items(world)) == ["Holds ZQFHIKE21", "Holds ZQFPIN20"]
        ids = await message_ids(world, "p1")
        target = ForgetTarget(
            kind="message",
            session_id="p1",
            message_ids=(ids["My door code is ZQFPIN20, keep it."],),
        )
        done = await forget(world, target)
        assert len(done.profile_items) == 2
        assert await items(world) == []
        held = await world.stores.profile.read(world.personal("amy"))
        assert held is not None and len(held.pending_memcell_ids) == 1
        await world.project()
        found = await world.memory.search(
            SearchRequest(
                app_id=APP,
                project_id=PROJECT,
                user_id="amy",
                query="door code",
                include_profile=True,
            )
        )
        assert "ZQFPIN20" not in json.dumps(found.data.model_dump(mode="json"))
        # The next update reads the queued segment, without the message.
        model.take()
        await send(
            world, "p2", ("amy", "Later: ZQFNEW22.", 9_000), ("bot", "Ok.", 9_001)
        )
        await profile_update(world)
        (profile,) = [p for name, p in model.take() if name == "_ProfileUpdateReply"]
        assert "ZQFPIN20" not in profile and "ZQFHIKE21" in profile
        assert sorted(await items(world)) == ["Holds ZQFHIKE21", "Holds ZQFNEW22"]
        held = await world.stores.profile.read(world.personal("amy"))
        assert held is not None and held.pending_memcell_ids == []
        (user_md,) = world.root.root.rglob("user.md")
        assert "ZQFPIN20" not in user_md.read_text()
        # The undo puts the forgotten fact back; the one written anew is not
        # put back twice.
        await undo(world, done.forget_id)
        assert sorted(await items(world)) == [
            "Holds ZQFHIKE21",
            "Holds ZQFNEW22",
            "Holds ZQFPIN20",
        ]


async def test_undoing_a_message_forget_puts_its_profile_items_back(
    tmp_path: Path,
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        await send(
            world,
            "p1",
            ("amy", "My door code is ZQFPIN20.", 1_000),
            ("amy", "I like ZQFHIKE21.", 1_001),
            ("bot", "Ok.", 1_002),
        )
        await profile_update(world)
        before = await world.stores.profile.read(world.personal("amy"))
        ids = await message_ids(world, "p1")
        done = await forget(
            world,
            ForgetTarget(
                kind="message",
                session_id="p1",
                message_ids=(ids["My door code is ZQFPIN20."],),
            ),
        )
        undone = await undo(world, done.forget_id)
        assert undone.profile_item_count == 2
        after = await world.stores.profile.read(world.personal("amy"))
        assert before is not None and after is not None
        assert after.explicit_info == before.explicit_info
        assert after.pending_memcell_ids == []


# ── A-2: a forgotten buffered message is never read ────────────────────


async def test_a_forgotten_buffered_message_reaches_no_model_and_no_file(
    tmp_path: Path,
) -> None:
    model = ForgetModel()
    async with open_world(tmp_path, model) as world:
        await send(
            world,
            "s10",
            ("amy", "Hello. ZQFROME1", 10_000),
            ("amy", "Private: ZQFSECRETALLERGY", 11_000),
            flush=False,
        )
        ids = await message_ids(world, "s10")
        done = await forget(
            world,
            ForgetTarget(
                kind="message",
                session_id="s10",
                message_ids=(ids["Private: ZQFSECRETALLERGY"],),
            ),
        )
        model.take()
        await world.memory.flush(
            session_id="s10", request_id="f", app_id=APP, project_id=PROJECT
        )
        await profile_update(world)
        calls = model.take()
        assert {name for name, _ in calls} >= {
            "_BatchReply",
            "EpisodeReply",
            "_ProfileUpdateReply",
        }
        assert all("ZQFSECRETALLERGY" not in prompt for _, prompt in calls)
        files = "".join(p.read_text() for p in world.root.root.rglob("*.md"))
        assert "ZQFSECRETALLERGY" not in files
        # The undo makes it an ordinary message of the next cut.
        await undo(world, done.forget_id)
        await world.memory.flush(
            session_id="s10", request_id="f", app_id=APP, project_id=PROJECT
        )
        assert "ZQFSECRETALLERGY" in " ".join(await live(world))
