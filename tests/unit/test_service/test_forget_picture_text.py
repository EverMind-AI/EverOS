"""A picture's forget removes its reference and nothing else of the text.

docs/api.md, ``/memory/forget``, ``target.kind: "image"``: the episodes and
scenes that name the picture are written again without the reference, their
paragraphs, line breaks and every other character kept; in a group, so is
each group-profile item whose evidence quotes it (no model call), and the
undo writes the quote back.
"""

from __future__ import annotations

import json
from pathlib import Path

from everos.memory.get import GetRequest
from everos.service import ForgetTarget
from everos.service.memory._forget_apply import strip_reference
from tests.helpers.forgets import (
    T0,
    ForgetModel,
    forget,
    live,
    open_world,
)
from tests.helpers.groups import APP, PROJECT, join


def test_strip_reference_keeps_the_rest_of_the_text() -> None:
    image = "ab" * 32
    body = (
        f"A photo (img-1: {image}).\n\nSecond paragraph keeps () and  two spaces.\n"
        f"- a list item [IMAGE: {image}]\n- another"
    )
    assert strip_reference(body, image) == (
        "A photo.\n\nSecond paragraph keeps () and  two spaces.\n"
        "- a list item\n- another"
    )
    assert strip_reference(f"(img-1: {image}) Opens the line.", image) == (
        "Opens the line."
    )


async def test_a_picture_forget_keeps_the_episodes_paragraphs(tmp_path: Path) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        scope = world.personal("amy")
        kite = await world.picture(scope, "a red kite", at=T0, shade=45, poster="amy")
        await world.personal_episode(
            "amy", f"A kite (img-1: {kite}).\n\nSecond paragraph.", at=T0
        )
        await forget(world, ForgetTarget(kind="image", image_id=kite))
        assert await live(world) == ["A kite.\n\nSecond paragraph."]


async def test_a_group_picture_forget_cuts_the_reference_out_of_profile_quotes(
    tmp_path: Path,
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        await world.roster(
            "trip", [join("u_chen", T0, role="owner"), join("u_dan", T0)]
        )
        scope = world.scope("trip")
        picture = await world.picture(
            scope, "the floor plan", at=T0 + 100, shade=51, poster="u_chen"
        )
        quote = f"We take this floor plan [IMAGE: {picture}] for the cabin"
        old = await world.profile_item("trip", quote, start=T0 + 100, end=T0 + 100)
        await world.project()

        async def items() -> list[tuple[str, str | None, str]]:
            return [
                (s.item.sources[0].quote, s.item.status, s.item.statement)
                for s in await world.stores.group_profile.items(scope)
            ]

        done = await world.memory.forget_in_group(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_chen",
            target=ForgetTarget(kind="image", image_id=picture),
        )
        assert f"trip_{old.entry_id}" in done.group_profile_items
        clean = "We take this floor plan for the cabin"
        ((was, was_status, _), (now, status, statement)) = await items()
        assert (was, was_status) == (quote, "forgotten")
        assert (now, statement) == (clean, clean)
        assert status not in ("superseded", "forgotten")
        await world.project()
        listed = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_dan",
                memory_type="group_profile",
                include_superseded=True,
            )
        )
        assert picture not in json.dumps(listed.model_dump(mode="json"))
        undone = await world.memory.undo_group_forget(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_chen",
            forget_id=done.forget_id,
        )
        assert undone.applied and len(undone.group_profile_items) == 2
        statuses = [(text, status) for text, status, _ in await items()]
        assert statuses[0] == (quote, "superseded")  # the link back, unmarked
        assert statuses[1] == (clean, "superseded")
        assert statuses[2][0] == quote
        assert statuses[2][1] not in ("superseded", "forgotten")
