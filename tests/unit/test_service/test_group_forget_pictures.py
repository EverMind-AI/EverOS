"""Forgetting a picture inside a group, and the message that carried one.

docs/group-memory.md, section 9.4 (a picture; a message's pictures) and 7.3.
A group batch with a picture goes through the bootstrap's memory service: the
episode call writes the group episode and one picture memory, and the poster's
member-profile update reads the picture and writes an item off it; then the
picture is forgotten. What these pin:

- only the poster (or an owner / admin) may forget a picture of the group;
- its group picture memories go, the episodes that referred to it are
  extracted again without it, the poster's items read off it go, and the
  file leaves circulation: its bytes answer 404 to everyone, members
  included, while no live memory refers to it;
- forgetting the message that carried a picture takes the picture's memory
  and items with it, and the segment is extracted again from what is left;
- undo brings the memory, the item and the bytes back.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from httpx import ASGITransport, AsyncClient

from everos.core.errors import ForbiddenError
from everos.entrypoints.api.app import create_app
from everos.memory.strategies import STRATEGY_SPECS
from everos.service import GroupForgetTarget
from tests.fakes import make_application
from tests.helpers.groups import APP, PROJECT, GroupWorld, join, png

T0 = 1_772_323_200_000
WORDS = ("PETSPLEASE", "BISCUITCAT", "TABBYPHOTO")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = [
        p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
        for p in content
    ]
    return "\n".join(str(part) for part in parts if part)


class _Model:
    """Episodes repeat the words they were shown; a picture is a tabby photo."""

    def __call__(self, messages: Any, *, response_format: type, **_kw: Any) -> str:
        name = response_format.__name__
        prompt = "\n".join(_text(m.content) for m in messages)
        said = ", ".join(w for w in WORDS[:2] if w in prompt) or "nothing"
        if name == "_BatchReply":
            return json.dumps(
                {"reasoning": "s", "boundaries": [], "tail_complete": False}
            )
        if name == "EpisodeReply":
            return json.dumps({"title": "Pets", "summary": said, "content": said})
        if name == "ImageEpisodeReply":
            labels = sorted(set(re.findall(r"label: (img-\d+)", prompt)))
            return json.dumps(
                {
                    "title": "Pets",
                    "summary": said,
                    "content": f"{said} ({', '.join(labels)})",
                    "images": [
                        {"ref": label, "description": f"A {WORDS[2]} of a cat"}
                        for label in labels
                    ],
                }
            )
        if name == "_ProfileUpdateReply":
            item = {"evidence": "x", "description": "Talks pets", "category": "pets"}
            ops: list[dict[str, Any]] = [
                {"action": "add", "type": "explicit_info", "data": item}
            ]
            # The poster's own update is attached the picture and reads it.
            for label in sorted(set(re.findall(r"label: (img-\d+)", prompt))):
                cat = {
                    "category": "pets",
                    "description": f"Has a cat ({WORDS[2]})",
                    "evidence": "the photo",
                }
                ops.append(
                    {
                        "action": "add",
                        "type": "explicit_info",
                        "data": cat,
                        "pictures": [label],
                    }
                )
            return json.dumps({"operations": ops})
        if name == "GroupProfileReply":
            return json.dumps({"operations": []})
        raise AssertionError(f"unexpected model call {name}")


@pytest.fixture
async def world(tmp_path: Path) -> AsyncIterator[tuple[GroupWorld, AsyncClient, str]]:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as w:
        await w.roster(
            "trip",
            [
                join("u_chen", T0, display_name="Chen", role="owner"),
                join("u_betty", T0, display_name="Betty"),
                join("u_dan", T0, display_name="Dan"),
            ],
        )
        picture = {
            "type": "image",
            "base64": base64.b64encode(png(120)).decode("ascii"),
            "ext": "png",
            "name": "cat.png",
        }
        await w.memory.add(
            {
                "session_id": "s_trip",
                "app_id": APP,
                "project_id": PROJECT,
                "group_id": "trip",
                "messages": [
                    {"message_id": "m-1", "sender_id": "u_chen", "role": "user",
                     "timestamp": T0 + 1_000, "content": f"{WORDS[0]} show us"},
                    {"message_id": "m-2", "sender_id": "u_betty", "role": "user",
                     "timestamp": T0 + 2_000,
                     "content": [{"type": "text", "text": f"{WORDS[1]} here"},
                                 picture]},
                ],
            },
            request_id="r",
            async_mode=False,
            defer_extraction=True,
        )  # fmt: skip
        await w.memory.flush(
            session_id="s_trip", request_id="f", app_id=APP, project_id=PROJECT
        )
        await w.run(*STRATEGY_SPECS)
        await w.project()
        rows = await w.storage.repos.image_memory.scan()
        app = create_app(
            application=make_application(memory=w.memory, images=w.stores.images)
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://t"
        ) as client:
            yield w, client, rows[0].image_id


async def _bytes_status(client: AsyncClient, image_id: str, asker: str) -> int:
    response = await client.get(
        f"/api/v2/memory/images/{image_id}", params={"asker_id": asker}
    )
    return response.status_code


async def _markdown(world: GroupWorld) -> str:
    return "\n".join(
        p.read_text("utf-8")
        for p in world.root.root.rglob("*.md")
        if not p.relative_to(world.root.root).parts[0].startswith(".")
    )


async def _forget(world: GroupWorld, asker: str, target: GroupForgetTarget) -> Any:
    return await world.memory.forget_in_group(
        app_id=APP, project_id=PROJECT, group_id="trip", asker_id=asker, target=target
    )


async def test_a_picture_is_forgotten_by_its_poster_and_comes_back_on_undo(
    world: tuple[GroupWorld, AsyncClient, str],
) -> None:
    w, client, image_id = world
    before = await _markdown(w)
    assert WORDS[2] in before  # the picture memory and Betty's item
    assert await _bytes_status(client, image_id, "u_dan") == 200
    target = GroupForgetTarget(kind="image", image_id=image_id)
    with pytest.raises(ForbiddenError):
        await _forget(w, "u_dan", target)

    result = await _forget(w, "u_betty", target)
    await w.project()

    assert len(result.image_memories) == 1 and result.images == (image_id,)
    assert result.rewritten == 1  # the episode, without the picture
    assert len(result.member_profile_items) == 1  # Betty's item read off it
    after = await _markdown(w)
    assert WORDS[2] not in after and image_id not in after
    assert WORDS[1] in after  # Betty's words stay: only the picture went
    for asker in ("u_chen", "u_betty", "u_dan"):
        assert await _bytes_status(client, image_id, asker) == 404

    await w.memory.undo_group_forget(
        app_id=APP,
        project_id=PROJECT,
        group_id="trip",
        asker_id="u_betty",
        forget_id=result.forget_id,
    )
    await w.project()
    assert WORDS[2] in await _markdown(w)
    assert await _bytes_status(client, image_id, "u_dan") == 200


async def test_the_message_that_carried_a_picture_takes_it_along(
    world: tuple[GroupWorld, AsyncClient, str],
) -> None:
    w, client, image_id = world
    result = await _forget(
        w,
        "u_betty",
        GroupForgetTarget(kind="message", session_id="s_trip", message_ids=("m-2",)),
    )
    await w.project()

    assert len(result.image_memories) == 1 and result.images == (image_id,)
    assert result.rewritten == 1
    after = await _markdown(w)
    assert WORDS[1] not in after and WORDS[2] not in after
    assert WORDS[0] in after  # Chen's message is what is left
    assert await _bytes_status(client, image_id, "u_chen") == 404
