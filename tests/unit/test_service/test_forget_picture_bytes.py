"""A forgotten picture's bytes are served to nobody until the undo.

docs/api.md, ``GET /memory/images/{image_id}``: a picture file a forget took
out of circulation answers as a missing one while no live memory refers to
it, and is served again once the forget is undone.
"""

from __future__ import annotations

from pathlib import Path

from everos.service import ForgetTarget
from tests.helpers.forgets import (
    T0,
    ForgetModel,
    forget,
    open_world,
    undo,
)


async def test_a_forgotten_pictures_bytes_are_served_to_nobody_until_undo(
    tmp_path: Path,
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        scope = world.personal("amy")
        kite = await world.picture(scope, "a red kite", at=T0, shade=46, poster="amy")
        await world.project()

        async def visible() -> bool:
            return await world.memory.picture_visible(
                kite, asker_id=None, asserted_asker=None
            )

        assert await visible()
        done = await forget(world, ForgetTarget(kind="image", image_id=kite))
        assert done.images == (kite,)
        assert not await visible()
        await undo(world, done.forget_id)
        assert await visible()
