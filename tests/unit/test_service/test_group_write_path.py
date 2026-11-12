"""The group write path, end to end (docs/group-memory.md, section 7; task 4).

Real markdown, ``system.db`` and LanceDB, the bootstrap's memory service and a
scripted model (``tests/helpers/groups.py``). What these pin:

- a segment with three human senders writes one group episode, one index row
  and two embeddings (body and subject); the one-to-one path writes three
  entries and six embeddings for the same messages;
- ``sender_ids`` / ``subject_ids`` / ``origin`` match fixtures: a mixed
  segment, an assistant-only segment, and a former member named in the text;
- the extraction call sees the speakers by their roster names, the assistant
  as ``"<name> (assistant)"``, and the roster block where the operator's
  custom instructions go;
- a group segment publishes ``GroupEpisodeExtracted`` (and its flush
  ``GroupFlushed``) and no one-to-one event, and writes nothing under
  ``users/``;
- a picture writes one group picture memory with ``sender_ids = [poster]``,
  and only the poster's member-profile update reads it: the items read off it
  reach the poster's member-in-group profile and nobody else's;
- the extraction call is told the group's language, resolved from its human
  messages -- for a picture-only segment, from the session's earlier
  segments -- and that language, not the reply's, is what the episode and
  its event carry;
- in agent mode a group cell also reaches the agent pipeline, and the event
  names the group, so a case learned from it carries ``source_group_id``.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from everalgo.testing import FakeLLMClient

from everos.infra.persistence.index import eq
from everos.memory.events import GroupEpisodeExtracted
from everos.memory.models import Episode
from everos.memory.strategies import UPDATE_MEMBER_PROFILES_SPEC
from tests.helpers.groups import APP, PROJECT, GroupWorld, join, leave, png

T0 = 1_772_323_200_000


class _Model:
    """Boundary: cut nowhere (a flush closes the segment). Episodes: scripted."""

    def __init__(self, content: str = "", title: str = "Planning") -> None:
        self.content = content
        self.title = title
        self.episode_prompts: list[str] = []

    def __call__(self, messages: Any, *, response_format: type, **_kw: Any) -> str:
        name = response_format.__name__
        if name == "_BatchReply":
            return json.dumps(
                {"reasoning": "s", "boundaries": [], "tail_complete": False}
            )
        texts = [_text(m.content) for m in messages]
        self.episode_prompts.append("\n".join(texts))
        if name == "EpisodeReply":
            labels = sorted(set(re.findall(r"label: (img-\d+)", texts[-1])))
            if not labels:
                return json.dumps(
                    {
                        "title": self.title,
                        "summary": self.content,
                        "content": self.content,
                    }
                )
            return json.dumps(
                {
                    "title": self.title,
                    "summary": self.content,
                    "content": self.content,
                    "images": [
                        {"ref": label, "description": "A tabby cat on a sill"}
                        for label in labels
                    ],
                }
            )
        if name == "_ProfileUpdateReply":
            # Only the poster's update is attached the picture.
            labels = sorted(set(re.findall(r"label: (img-\d+)", texts[-1])))
            cat = {
                "category": "pets",
                "description": "Has a tabby cat",
                "evidence": "the photo shows the cat at home",
            }
            ops = [
                {"action": "add", "type": "explicit_info", "data": cat, "pictures": [x]}
                for x in labels
            ]
            return json.dumps({"operations": ops or [{"action": "none"}]})
        raise AssertionError(f"unexpected model call {name}")


def _text(content: Any) -> str:
    """A message's text, whether it is a string or a list of parts."""
    if isinstance(content, str):
        return content
    parts = [
        p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
        for p in content
    ]
    return "\n".join(str(part) for part in parts if part)


def _message(sender: str, at: int, text: Any, role: str = "user") -> dict[str, Any]:
    return {
        "sender_id": sender,
        "sender_name": sender,
        "role": role,
        "timestamp": T0 + at,
        "content": text,
    }


async def _roster(world: GroupWorld) -> None:
    await world.roster(
        "trip",
        [
            join("u_chen", T0, display_name="Chen", role="owner", title="Organiser"),
            join("u_betty", T0, display_name="Betty"),
            join("u_dan", T0, display_name="Dan"),
            join("bot_ava", T0, kind="assistant", role="assistant", display_name="Ava"),
        ],
    )


async def _send(
    world: GroupWorld, session: str, *messages: dict[str, Any], group: bool = True
) -> None:
    payload: dict[str, Any] = {
        "session_id": session,
        "app_id": APP,
        "project_id": PROJECT,
        "messages": list(messages),
    }
    if group:
        payload["group_id"] = "trip"
    await world.memory.add(
        payload, request_id="r", async_mode=False, defer_extraction=True
    )
    await world.memory.flush(
        session_id=session, request_id="f", app_id=APP, project_id=PROJECT
    )


async def _group_episodes(world: GroupWorld) -> list[Episode]:
    entries = [
        e.episode_entry_id
        for e in world.bus.events
        if isinstance(e, GroupEpisodeExtracted)
    ]
    stored = await world.stores.episode.read_many(world.scope("trip"), entries)
    return [s.item for s in sorted(stored, key=lambda s: s.item.timestamp)]


_THREE = (
    _message("u_chen", 1_000, "May 1 flights are sold out. Let's leave on April 30."),
    _message("u_betty", 2_000, "Fine by me. I'll book the hotel from April 30."),
    _message("u_dan", 3_000, "I can drive us to the airport."),
)


async def test_one_entry_one_row_two_embeddings_on_the_group_path(
    tmp_path: Path,
) -> None:
    content = "Chen moved departure to April 30; Betty books the hotel; Dan drives."
    async with GroupWorld.open(
        tmp_path / "group", llm=FakeLLMClient(handler=_Model(content))
    ) as world:
        await _roster(world)
        await _send(world, "s_group", *_THREE)
        assert world.embedding.calls == []
        await world.project()
        rows = await world.storage.repos.episode.find_where(
            eq("owner_type", "group"), limit=100
        )
        assert len(rows) == 1
        assert len(world.embedding.calls) == 2  # body and subject, once
        files = list((tmp_path / "group").rglob("episode-*.md"))
        assert [f.relative_to(tmp_path / "group").parts[2] for f in files] == ["groups"]
        assert not (tmp_path / "group" / "default_app" / PROJECT / "users").exists()

    async with GroupWorld.open(
        tmp_path / "solo", llm=FakeLLMClient(handler=_Model(content))
    ) as world:
        await _send(world, "s_solo", *_THREE, group=False)
        await world.project()
        rows = await world.storage.repos.episode.find_where(
            eq("owner_type", "user"), limit=100
        )
        assert len(rows) == 3
        assert len(world.embedding.calls) == 6


async def test_attribution_of_a_mixed_segment(tmp_path: Path) -> None:
    content = (
        "Chen said the May 1 flights were sold out and proposed April 30; "
        "Betty agreed. The assistant, Ava, listed two flights."
    )
    model = _Model(content)
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        await _roster(world)
        await _send(
            world,
            "s_group",
            _message("u_chen", 1_000, "May 1 is sold out, April 30?"),
            _message("u_betty", 2_000, "Fine by me."),
            _message("bot_ava", 3_000, "Two flights: 08:10, 13:40.", "assistant"),
        )
        (episode,) = await _group_episodes(world)
        assert episode.owner_id == "trip"
        assert episode.sender_ids == ["u_chen", "u_betty", "bot_ava"]
        assert episode.subject_ids == ["u_chen", "u_betty"]
        assert episode.origin == "mixed"
        assert episode.source_start == T0 + 1_000
        assert episode.timestamp == T0 + 3_000

        # The model saw roster names, the assistant labelled as such, and the
        # roster block where the operator's custom instructions go.
        (prompt,) = model.episode_prompts
        assert "] Chen: May 1 is sold out" in prompt
        assert "] Ava (assistant): Two flights" in prompt
        assert "u_chen:" not in prompt
        assert "This conversation is a group chat. People in this part: " in prompt
        assert "Chen (owner, Organiser), Betty (member)" in prompt
        assert '"Ava (assistant)" is the group\'s AI assistant.' in prompt

        # No one-to-one event: the segment and the flush are the group's.
        kinds = [type(e).__name__ for e in world.bus.events]
        assert kinds == ["GroupEpisodeExtracted", "GroupFlushed"]
        assert world.bus.events[0].speaker_ids == ["u_chen", "u_betty"]


async def test_an_assistant_only_segment(tmp_path: Path) -> None:
    async with GroupWorld.open(
        tmp_path, llm=FakeLLMClient(handler=_Model("Ava posted a check-in reminder."))
    ) as world:
        await _roster(world)
        await _send(
            world,
            "s_group",
            _message("bot_ava", 1_000, "Reminder: check in opens today.", "assistant"),
        )
        (episode,) = await _group_episodes(world)
        assert episode.sender_ids == ["bot_ava"]
        assert episode.origin == "assistant"
        assert episode.subject_ids == []


async def test_a_former_member_named_in_the_text_is_a_subject(tmp_path: Path) -> None:
    async with GroupWorld.open(
        tmp_path,
        llm=FakeLLMClient(handler=_Model("Chen said Dan left his charger behind.")),
    ) as world:
        await _roster(world)
        await world.roster("trip", [leave("u_dan", T0 + 500)])
        await _send(
            world, "s_group", _message("u_chen", 1_000, "Dan left his charger here.")
        )
        (episode,) = await _group_episodes(world)
        assert episode.subject_ids == ["u_chen", "u_dan"]
        assert episode.origin == "member"


async def test_a_picture_is_one_group_memory_and_reaches_only_its_poster(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(
        tmp_path,
        llm=FakeLLMClient(handler=_Model("Betty shared a photo of her cat (img-1).")),
    ) as world:
        await _roster(world)
        picture = {
            "type": "image",
            "base64": base64.b64encode(png(120)).decode("ascii"),
            "ext": "png",
            "name": "cat.png",
        }
        await _send(
            world,
            "s_group",
            _message("u_chen", 1_000, "Show us your pets!"),
            _message(
                "u_betty", 2_000, [{"type": "text", "text": "Meet Biscuit"}, picture]
            ),
        )
        images = list((tmp_path / "default_app" / PROJECT).rglob("image-*.md"))
        assert [i.relative_to(tmp_path).parts[2:4] for i in images] == [
            ("groups", "trip")
        ]
        await world.project()
        rows = await world.storage.repos.image_memory.find_where(
            eq("owner_type", "group"), limit=10
        )
        assert len(rows) == 1
        assert rows[0].sender_ids == ["u_betty"]
        assert rows[0].origin == "member"

        await world.run(UPDATE_MEMBER_PROFILES_SPEC)
        betty = await world.stores.member_profile.read(world.scope("trip"), "u_betty")
        assert betty is not None
        (item,) = betty.profile.explicit_info
        assert item.description == "Has a tabby cat"
        extra = item.model_extra or {}
        assert extra["source_memcell_ids"]
        assert extra["source_image_ids"] == [rows[0].image_id]
        # Nobody else's profile reads it, in the group or outside it.
        chen = await world.stores.member_profile.read(world.scope("trip"), "u_chen")
        assert chen is not None and chen.profile.explicit_info == []
        assert not (tmp_path / "default_app" / PROJECT / "users").exists()


async def test_a_picture_only_segment_is_told_the_group_s_language(
    tmp_path: Path,
) -> None:
    # The reply drifts into Chinese; the conversation is English.
    model = _Model("Betty 发来一张她的猫的照片（img-1）。")
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        await _roster(world)
        await _send(world, "s_group", *_THREE)
        picture = {
            "type": "image",
            "base64": base64.b64encode(png(90)).decode("ascii"),
            "ext": "png",
            "name": "cat.png",
        }
        await _send(world, "s_group", _message("u_betty", 9_000, [picture]))

        assert len(model.episode_prompts) == 2
        for prompt in model.episode_prompts:
            assert "Write every output field in English" in prompt
            assert "the language the participants use" not in prompt
        episodes = await _group_episodes(world)
        assert [e.language for e in episodes] == ["English", "English"]
        languages = [
            e.language for e in world.bus.events if isinstance(e, GroupEpisodeExtracted)
        ]
        assert languages == ["English", "English"]


async def test_in_agent_mode_a_group_cell_reaches_the_agent_track_with_its_group(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(
        tmp_path, llm=FakeLLMClient(handler=_Model("Ava answered Chen.")), mode="agent"
    ) as world:
        await _roster(world)
        await _send(
            world,
            "s_group",
            _message("u_chen", 1_000, "Which flight?"),
            _message("bot_ava", 2_000, "The 08:10 one.", "assistant"),
        )
        kinds = [type(e).__name__ for e in world.bus.events]
        assert sorted(kinds) == [
            "AgentPipelineStarted",
            "GroupEpisodeExtracted",
            "GroupFlushed",
        ]
        started = next(
            e for e in world.bus.events if type(e).__name__ == "AgentPipelineStarted"
        )
        assert started.group_id == "trip"
