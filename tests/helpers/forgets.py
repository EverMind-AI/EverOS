"""A user's forgets over the real stack, with a scripted model.

What the forget and undo tests of a user's memory share
(``tests/unit/test_service/test_forget_*.py``): the stack over a temporary
root (:class:`~tests.helpers.groups.GroupWorld`), a model whose episodes
repeat the markers they were shown and whose profile updates add one item
per fact marker (``ZQF...``) of each segment they read, and the calls the
tests make as the routes would.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage

from everos.memory.strategies import EXTRACT_USER_PROFILE_SPEC
from everos.service import ForgetTarget

from .groups import APP, PROJECT, GroupWorld

__all__ = [
    "T0",
    "ForgetModel",
    "forget",
    "items",
    "live",
    "message_ids",
    "open_world",
    "profile_update",
    "record",
    "send",
    "undo",
]

T0 = 1_772_323_200_000
_MARK = re.compile(r"\bZQ[A-Z0-9]+\b")
_FACT = re.compile(r"\bZQF[A-Z0-9]+\b")
_SEGMENT = re.compile(r'<segment id="(s\d+)">(.*?)</segment>', re.S)


class ForgetModel:
    """Episodes repeat the markers they were shown; a profile update adds one
    item per fact marker of each segment it reads; every prompt is kept."""

    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []

    def take(self) -> list[tuple[str, str]]:
        taken, self.prompts = self.prompts, []
        return taken

    def __call__(self, messages: Any, *, response_format: type, **_: Any) -> str:
        name = response_format.__name__
        prompt = "\n".join(str(getattr(m, "content", m)) for m in messages)
        self.prompts.append((name, prompt))
        if name == "_BatchReply":
            return json.dumps(
                {"reasoning": "s", "boundaries": [], "tail_complete": False}
            )
        if name == "EpisodeReply":
            said = " ".join(sorted(set(_MARK.findall(prompt)))) or "nothing"
            return json.dumps(
                {"title": "Notes", "summary": f"Notes on {said}", "content": said}
            )
        if name == "_ProfileUpdateReply":
            stored = prompt.split("<new_conversation>")[0]
            ops = [
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": {
                        "evidence": fact,
                        "description": f"Holds {fact}",
                        "category": "facts",
                    },
                    "sources": [segment],
                    "pictures": [],
                }
                for segment, body in _SEGMENT.findall(prompt)
                for fact in dict.fromkeys(_FACT.findall(body))
                if f"Holds {fact}" not in stored
            ]
            return json.dumps({"operations": ops or [{"action": "none"}]})
        if name == "_ConsolidateReply":
            return json.dumps({"changes": [], "dropped": []})
        if name == "_DecisionReply":
            return json.dumps({"core": list(range(10)), "next_queries": []})
        raise AssertionError(f"unexpected model call {name}")


def open_world(tmp_path: Path, model: ForgetModel) -> Any:
    client = FakeLLMClient(handler=model)
    return GroupWorld.open(tmp_path, llm=client, decider=client)


async def send(
    world: GroupWorld,
    session: str,
    *lines: tuple[str, str, int],
    flush: bool = True,
) -> None:
    await world.memory.add(
        {"session_id": session, "app_id": APP, "project_id": PROJECT,
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
            session_id=session, request_id="f", app_id=APP, project_id=PROJECT
        )


async def profile_update(world: GroupWorld) -> None:
    await world.run(EXTRACT_USER_PROFILE_SPEC)


async def forget(world: GroupWorld, target: ForgetTarget) -> Any:
    return await world.memory.forget(
        app_id=APP, project_id=PROJECT, user_id="amy", target=target, dry_run=False
    )


async def undo(world: GroupWorld, forget_id: str) -> Any:
    return await world.memory.undo_forget(
        app_id=APP, project_id=PROJECT, user_id="amy", forget_id=forget_id
    )


async def message_ids(world: GroupWorld, session: str) -> dict[str, str]:
    """Each message's id by its text, segments then buffer."""
    found: dict[str, str] = {}
    key = world.session(session)
    for cell in (await world.stores.memcell.cells_in_session(key)).values():
        for item in cell.items:
            if isinstance(item, ChatMessage):
                found[str(item.content)] = item.id
    for message in await world.stores.session_buffer.list(key):
        found[message.text] = message.client_message_id or message.message_id
    return found


async def live(world: GroupWorld) -> list[str]:
    """The text of amy's current episodes."""
    return [
        s.item.episode
        for s, mark in await world.stores.episode.read_all(world.personal("amy"))
        if mark is None
    ]


async def items(world: GroupWorld) -> list[str]:
    held = await world.stores.profile.read(world.personal("amy"))
    return [i.description for i in held.explicit_info] if held else []


async def record(world: GroupWorld, marker: str) -> str:
    """The id of amy's current episode that says ``marker``."""
    for stored, mark in await world.stores.episode.read_all(world.personal("amy")):
        if mark is None and marker in stored.item.episode:
            return f"amy_{stored.ref.entry_id}"
    raise AssertionError(f"no current episode says {marker}")
