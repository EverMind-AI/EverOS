"""``/memory/get`` lists a user's picture memories, to the user only.

docs/api.md, ``POST /memory/get``: ``memory_type: "image_memory"`` with a
``user_id`` lists the user's current picture memories, in the shape a
group's listing has, under the read rules of every personal listing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.core.errors import ForbiddenError
from everos.memory.get import GetMemoryType, GetRequest
from tests.helpers.forgets import (
    T0,
    ForgetModel,
    open_world,
)
from tests.helpers.groups import APP, PROJECT


async def test_get_lists_a_users_picture_memories_to_the_user_only(
    tmp_path: Path,
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        scope = world.personal("amy")
        kite = await world.picture(scope, "a red kite", at=T0, shade=48, poster="amy")
        await world.project()
        listed = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                user_id="amy",
                memory_type=GetMemoryType.IMAGE_MEMORY,
            )
        )
        (image,) = listed.data.images
        assert (image.image_id, image.user_id, image.group_id) == (kite, "amy", None)
        assert image.description == "a red kite"
        with pytest.raises(ForbiddenError):
            await world.memory.get(
                GetRequest(
                    app_id=APP,
                    project_id=PROJECT,
                    user_id="amy",
                    asker_id="bob",
                    memory_type=GetMemoryType.IMAGE_MEMORY,
                )
            )
