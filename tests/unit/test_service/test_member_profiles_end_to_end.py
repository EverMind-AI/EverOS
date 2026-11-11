"""Member-in-group profiles through the real stack (docs/group-memory.md, 7.4).

The bootstrap's memory service writes a group segment and a flush; the
``update_member_profiles`` strategy then runs on the events it published.
What these pin (task 5):

- one ``groups/<g>/members/<m>.md`` per member who spoke -- none for the
  assistant -- and every item cites a segment with a message by that member;
- a flush brings a profile current even below ``member_profile_every``;
- the member's personal ``users/<m>/user.md`` is untouched by group segments;
- the cascade indexes the profile as the member's profile in the group.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from everalgo.testing import FakeLLMClient

from everos.memory.strategies import UPDATE_MEMBER_PROFILES_SPEC
from tests.helpers.groups import APP, PROJECT, GroupWorld, join

T0 = 1_772_323_200_000


def _model(messages: Any, *, response_format: type, **_: Any) -> str:
    name = response_format.__name__
    if name == "_BatchReply":
        return json.dumps({"reasoning": "s", "boundaries": [], "tail_complete": False})
    if name == "_EpisodeReply":
        return json.dumps(
            {"title": "Trip", "summary": "Trip plans.", "content": "Trip plans."}
        )
    if name == "_ProfileUpdateReply":
        text = "\n".join(str(m.content) for m in messages)
        found = re.search(r"The target user is (.+?), the speaker", text)
        who = found.group(1).strip() if found else "?"
        item = {
            "evidence": f"{who}: said it",
            "description": f"{who} plans the trip",
            "category": "plans",
        }
        return json.dumps(
            {"operations": [{"action": "add", "type": "explicit_info", "data": item}]}
        )
    raise AssertionError(f"unexpected model call {name}")


async def test_each_speaker_gets_a_member_profile_and_user_md_is_untouched(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_model)) as world:
        await world.roster(
            "trip",
            [
                join("u_chen", T0, display_name="Chen"),
                join("u_betty", T0, display_name="Betty"),
                join("bot_ava", T0, kind="assistant", display_name="Ava"),
            ],
        )
        await world.profile("u_chen", "Chen lives in Porto")
        user_md = tmp_path / "default_app" / PROJECT / "users" / "u_chen" / "user.md"
        before = user_md.read_bytes()

        await world.memory.add(
            {
                "session_id": "s_group",
                "app_id": APP,
                "project_id": PROJECT,
                "group_id": "trip",
                "messages": [
                    {"sender_id": "u_chen", "role": "user", "timestamp": T0 + 1_000,
                     "content": "I'll book the flights"},
                    {"sender_id": "u_betty", "role": "user", "timestamp": T0 + 2_000,
                     "content": "I'll book the hotel"},
                    {"sender_id": "bot_ava", "role": "assistant",
                     "timestamp": T0 + 3_000, "content": "Noted both."},
                ],
            },
            request_id="r",
            async_mode=False,
            defer_extraction=True,
        )  # fmt: skip
        await world.memory.flush(
            session_id="s_group", request_id="f", app_id=APP, project_id=PROJECT
        )
        # Below member_profile_every (5): the flush is what runs the update.
        await world.run(UPDATE_MEMBER_PROFILES_SPEC)

        members = tmp_path / "default_app" / PROJECT / "groups" / "trip" / "members"
        assert sorted(p.name for p in members.glob("*.md")) == [
            "u_betty.md",
            "u_chen.md",
        ]
        memcells = {
            e.memcell_id
            for e in world.bus.events
            if type(e).__name__ == "GroupEpisodeExtracted"
        }
        for member in ("u_chen", "u_betty"):
            profile = await world.stores.member_profile.read(
                world.scope("trip"), member
            )
            assert profile is not None and profile.pending_memcell_ids == []
            (item,) = profile.profile.explicit_info
            assert set((item.model_extra or {})["source_memcell_ids"]) <= memcells
        assert user_md.read_bytes() == before

        await world.project()
        index = world.deps.indexes.profile
        found = await index.member(world.scope("trip"), "u_betty")
        assert found is not None
        assert found.item.explicit_info[0].description == "Betty plans the trip"
