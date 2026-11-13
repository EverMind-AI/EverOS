"""Undo order: through later undone forgets, and over a person's edit.

docs/api.md, ``/memory/forget/undo``, "Order". Two forgets that rewrite one
record undo newest first, then the earlier: the earlier undo follows the
later forget's undone copy to the current version. A person's edit of the
version a forget wrote: the undo writes the old version over the edit when
the edit left the forget's change alone, and answers 409 naming the edit
when it changed the same text. Real markdown, ``system.db`` and LanceDB
(``tests/helpers/forgets.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from everos.core.errors import ConflictError
from everos.service import ForgetTarget, MemoryChanges
from tests.helpers.forgets import (
    T0,
    ForgetModel,
    forget,
    live,
    message_ids,
    open_world,
    record,
    send,
    undo,
)
from tests.helpers.groups import APP, PROJECT, GroupWorld


async def test_two_picture_forgets_of_one_episode_undo_newest_first(
    tmp_path: Path,
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        scope = world.personal("amy")
        kite = await world.picture(scope, "a red kite", at=T0, shade=40, poster="amy")
        boat = await world.picture(
            scope, "a blue boat", at=T0 + 1, shade=90, poster="amy"
        )
        text = f"Flew the kite (img-1: {kite}) and sailed the boat (img-2: {boat})."
        await world.personal_episode("amy", text, at=T0)
        first = await forget(world, ForgetTarget(kind="image", image_id=kite))
        later = await forget(world, ForgetTarget(kind="image", image_id=boat))
        with pytest.raises(ConflictError):
            await undo(world, first.forget_id)
        assert (await undo(world, later.forget_id)).applied
        assert (await undo(world, first.forget_id)).applied
        assert await live(world) == [text]
        rows = await world.stores.image_memory.read_all(scope)
        assert [r.item.superseded_by for r in rows] == [None, None]


async def test_two_message_forgets_of_one_segment_undo_newest_first(
    tmp_path: Path,
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        await send(
            world,
            "s3",
            ("amy", "Secret ZQSECRET3", 1_000),
            ("amy", "Keep ZQKEEP4", 2_000),
            ("amy", "City ZQCITY5", 3_000),
            ("bot", "ok", 4_000),
        )
        before = await live(world)
        ids = await message_ids(world, "s3")
        first = await forget(
            world,
            ForgetTarget(
                kind="message", session_id="s3", message_ids=(ids["Secret ZQSECRET3"],)
            ),
        )
        later = await forget(
            world,
            ForgetTarget(
                kind="message", session_id="s3", message_ids=(ids["Keep ZQKEEP4"],)
            ),
        )
        assert await live(world) == ["ZQCITY5"]
        assert (await undo(world, later.forget_id)).applied
        assert (await undo(world, first.forget_id)).applied
        assert await live(world) == before == ["ZQCITY5 ZQKEEP4 ZQSECRET3"]
        assert (
            await world.stores.message_forget.forgotten(world.session("s3"))
            == frozenset()
        )


# ── C-4: an edit of the version a forget wrote ──────────────────────────


async def _picture_forget_and_edit(
    world: GroupWorld, content: str | None, subject: str | None = None
) -> tuple[Any, str, str]:
    scope = world.personal("amy")
    kite = await world.picture(scope, "a red kite", at=T0, shade=44, poster="amy")
    text = f"Flew the kite (img-1: {kite}) on Sunday.\n\nThen lunch."
    await world.personal_episode("amy", text, at=T0)
    forgot = await forget(world, ForgetTarget(kind="image", image_id=kite))
    (wrote,) = forgot.new_episodes
    await world.memory.update_memory(
        app_id=APP,
        project_id=PROJECT,
        user_id="amy",
        record_id=wrote,
        changes=MemoryChanges(content=content, subject=subject),
    )
    return forgot, kite, text


async def test_an_undo_writes_the_old_text_over_an_edit_elsewhere(
    tmp_path: Path,
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        forgot, kite, _ = await _picture_forget_and_edit(
            world, "Flew the kite on Sunday.\n\nThen a long lunch.", subject="Kite day"
        )
        undone = await undo(world, forgot.forget_id)
        assert undone.applied and undone.images == (kite,)
        ((current, _),) = [
            pair
            for pair in await world.stores.episode.read_all(world.personal("amy"))
            if pair[1] is None
        ]
        assert current.item.episode == (
            f"Flew the kite (img-1: {kite}) on Sunday.\n\nThen a long lunch."
        )
        assert current.item.subject == "Kite day"
        assert current.item.edited_by == "amy" and current.item.held


async def test_an_undo_refuses_an_edit_of_the_text_it_changed(
    tmp_path: Path,
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        forgot, _, text = await _picture_forget_and_edit(world, "Kite day.")
        with pytest.raises(ConflictError, match="a person edited"):
            await undo(world, forgot.forget_id)
        # Edited back to what the forget wrote, the undo goes through.
        current = await record(world, "Kite day.")
        await world.memory.update_memory(
            app_id=APP,
            project_id=PROJECT,
            user_id="amy",
            record_id=current,
            changes=MemoryChanges(content="Flew the kite on Sunday.\n\nThen lunch."),
        )
        assert (await undo(world, forgot.forget_id)).applied
        assert await live(world) == [text]
