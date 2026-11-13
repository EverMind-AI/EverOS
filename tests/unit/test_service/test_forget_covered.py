"""A record stays forgotten while any live forget covers it.

docs/api.md, ``/memory/forget/undo``. A wider forget (a conversation,
everything) that reaches what a narrower one already took -- an episode, a
profile item, a picture memory and its file -- marks nothing again and logs
it as covered: undoing the narrower forget leaves it out while the wider one
stands, and undoing the wider one brings it back. Real markdown,
``system.db`` and LanceDB (``tests/helpers/forgets.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.service import ForgetTarget
from tests.helpers.forgets import (
    T0,
    ForgetModel,
    forget,
    items,
    live,
    open_world,
    profile_update,
    record,
    send,
    undo,
)


@pytest.mark.parametrize("wider", ["session", "owner"])
async def test_a_record_stays_forgotten_while_a_wider_forget_covers_it(
    tmp_path: Path, wider: str
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        await send(
            world, "s1", ("amy", "My dog is ZQFDOG1", 1_000), ("bot", "ok", 2_000)
        )
        await profile_update(world)
        await send(
            world, "s2", ("amy", "My cat is ZQFCAT2", 5_000), ("bot", "ok", 6_000)
        )
        await profile_update(world)
        assert sorted(await items(world)) == ["Holds ZQFCAT2", "Holds ZQFDOG1"]
        narrow = await forget(
            world,
            ForgetTarget(kind="record", record_ids=(await record(world, "ZQFDOG1"),)),
        )
        target = (
            ForgetTarget(kind="session", session_id="s1")
            if wider == "session"
            else ForgetTarget(kind="owner")
        )
        wide = await forget(world, target)
        if wider == "session":
            assert wide.episodes == ()  # its only episode was forgotten already
        undone = await undo(world, narrow.forget_id)
        assert undone.applied and undone.episodes == () and undone.profile_items == ()
        said = " ".join(await live(world))
        assert "ZQFDOG1" not in said
        assert "Holds ZQFDOG1" not in await items(world)
        if wider == "owner":
            assert said == "" and await items(world) == []
        back = await undo(world, wide.forget_id)
        assert back.applied
        assert sorted(await live(world)) == ["ZQFCAT2", "ZQFDOG1"]
        assert sorted(await items(world)) == ["Holds ZQFCAT2", "Holds ZQFDOG1"]


async def test_a_covered_pictures_file_stays_out_until_the_wider_undo(
    tmp_path: Path,
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        scope = world.personal("amy")
        picture = await world.picture(
            scope, "a red kite", at=T0, shade=47, poster="amy"
        )
        await world.project()
        (memory,) = await world.stores.image_memory.read_by_image(scope, picture)
        narrow = await forget(
            world,
            ForgetTarget(kind="record", record_ids=(f"amy_{memory.ref.entry_id}",)),
        )
        assert narrow.images == (picture,)
        wide = await forget(world, ForgetTarget(kind="owner"))

        async def visible() -> bool:
            return await world.memory.picture_visible(
                picture, asker_id=None, asserted_asker=None
            )

        await undo(world, narrow.forget_id)
        assert not await visible()
        (memory,) = await world.stores.image_memory.read_by_image(scope, picture)
        assert memory.item.superseded_by == "forgotten"
        back = await undo(world, wide.forget_id)
        assert back.images == (picture,)
        assert await visible()
