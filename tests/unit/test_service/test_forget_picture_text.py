"""A picture's forget removes its reference and nothing else of the text.

docs/api.md, ``/memory/forget``, ``target.kind: "image"``: the episodes and
scenes that name the picture are written again without the reference, their
paragraphs, line breaks and every other character kept.
"""

from __future__ import annotations

from pathlib import Path

from everos.service import ForgetTarget
from everos.service.memory._forget_apply import strip_reference
from tests.helpers.forgets import (
    T0,
    ForgetModel,
    forget,
    live,
    open_world,
)


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
