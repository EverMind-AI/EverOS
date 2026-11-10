"""``GroupMemoryPipeline.run_cell`` names the conversation's language to the model.

The group twin of ``test_user_memory_language.py``. A group segment is
extracted by the one-to-one call, so it had the one-to-one defect: no
language named, a picture segment of an English group answered in Chinese
(episode, picture descriptions and the poster's picture profile items), and
that reply's language pinned on the event the member-in-group profiles and
the scenes read.

The contract now:

- The language is resolved in code from the segment's human messages, and
  from its session's earlier segments when it has too little text of its
  own, then named on both routes: a picture-only segment of an English group
  gets an English instruction.
- The resolved language, not the reply's, is what the group episode, its
  ``GroupEpisodeExtracted`` and the poster's member-in-group profile carry.
- Only a session without a letter anywhere leaves the choice to the model,
  and then, as before, the reply's language is what gets pinned.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.component.images import ImageLimits
from everos.memory import IngestResult
from everos.memory.events import GroupEpisodeExtracted
from everos.memory.extract.pipeline import GroupMemoryPipeline, PreparedCell
from everos.memory.models import Group, MemcellRecord, RosterEvent
from everos.memory.ports import MemoryRef, Scope, SessionKey
from tests.fakes import bundled_instructions
from tests.fakes.stores import (
    InMemoryGroupStore,
    InMemoryImageMemoryStore,
    InMemoryImageStore,
    InMemoryMemberProfileStore,
    InMemoryMemcellStore,
)

_T0 = 1_700_000_000_000
_IMAGE_ID = "9f" + "0" * 62
_SESSION = SessionKey("default", "default", "s_group")
_GROUP = Scope(
    app_id="default", project_id="default", owner_id="trip", owner_type="group"
)
_INGESTED = IngestResult(session_id="s_group", messages=[], group_id="trip")
_REF = MemoryRef(
    id="trip_gep_20260517_00000001",
    entry_id="gep_20260517_00000001",
    md_path="default_app/default_project/groups/trip/episodes/episode-2026-05-17.md",
)
_ENGLISH_RULE = "Write every output field in English"
_PARTICIPANT_RULE = "Write ALL output fields in the language the participants use"

_CHINESE_PICTURE_REPLY = {
    "content": "Betty 发来一张她的猫的照片（img-1）。",
    "summary": "Betty 分享了猫的照片。",
    "title": "猫的照片",
    "images": [{"ref": "img-1", "description": "窗台上的一只虎斑猫", "ocr": ""}],
    "profile": [
        {
            "action": "add",
            "type": "explicit_info",
            "source_image_id": "img-1",
            "data": {
                "category": "pets",
                "description": "养了一只虎斑猫",
                "evidence": "照片里的猫在她家",
            },
        }
    ],
}
_CHINESE_EPISODE_REPLY = {
    "content": "Chen 把出发日期改到了四月三十日。",
    "summary": "Chen 改了出发日期。",
    "title": "改出发日期",
}


def _limits() -> ImageLimits:
    return ImageLimits(
        max_bytes=5 * 1024 * 1024,
        max_per_message=4,
        max_request_bytes=10 * 1024 * 1024,
        max_per_request=8,
        formats=("png", "jpeg", "webp", "gif"),
        model_edge=1024,
        cache_quality=85,
    )


def _message(sender: str, content: str, *, at: int) -> ChatMessage:
    return ChatMessage(
        id=f"m_{at}", role="user", content=content, timestamp=at, sender_id=sender
    )


def _cell(*messages: ChatMessage) -> MemCell:
    return MemCell(items=list(messages), timestamp=messages[-1].timestamp)


_EARLIER = _cell(
    _message(
        "u_chen", "The May 1 flights are sold out, so we leave on April 30.", at=_T0
    ),
    _message("u_betty", "Fine by me, I will book the hotel from April 30.", at=_T0 + 1),
)
_PICTURE_ONLY = _cell(_message("u_betty", f"[IMAGE: {_IMAGE_ID}]", at=_T0 + 60_000))


def _replying(reply: dict) -> FakeLLMClient:
    return FakeLLMClient(handler=lambda *_a, **_k: json.dumps(reply))


async def _pipeline(
    llm: FakeLLMClient, *cells: tuple[str, MemCell]
) -> tuple[GroupMemoryPipeline, MagicMock, InMemoryMemberProfileStore]:
    episodes = MagicMock()
    episodes.append = AsyncMock(return_value=_REF)
    memcells = InMemoryMemcellStore()
    await memcells.insert_many(
        _SESSION,
        [
            MemcellRecord(
                memcell_id=mid,
                cell=cell,
                raw_type="chat",
                message_ids=[],
                sender_ids=[],
            )
            for mid, cell in cells
        ],
    )
    groups = InMemoryGroupStore()
    key = ("default", "default", "trip")
    groups.groups[key] = Group(group_id="trip", created_at=0)
    groups.events[key].extend(
        [
            RosterEvent(op="join", member_id="u_chen", at=0, display_name="Chen"),
            RosterEvent(op="join", member_id="u_betty", at=0, display_name="Betty"),
        ]
    )
    pictures = InMemoryImageStore()
    pictures.images[_IMAGE_ID] = (b"picture bytes", "png")
    pictures.refs[_IMAGE_ID] = 1
    members = InMemoryMemberProfileStore()
    pipeline = GroupMemoryPipeline(
        episode_store=episodes,  # type: ignore[arg-type]  # recording double
        memcell_store=memcells,
        groups=groups,
        instructions=bundled_instructions(),
        llm_client=llm,  # type: ignore[arg-type]  # FakeLLMClient
        image_store=pictures,
        image_memory_store=InMemoryImageMemoryStore(),
        image_limits=_limits(),
        member_profiles=members,
    )
    return pipeline, episodes, members


def _system_prompt(llm: FakeLLMClient) -> str:
    (call,) = llm.calls
    return str(call.messages[0].content)


async def test_an_english_group_s_picture_only_segment_is_told_english() -> None:
    llm = _replying(_CHINESE_PICTURE_REPLY)
    pipeline, _, _ = await _pipeline(llm, ("mc_1", _EARLIER), ("mc_2", _PICTURE_ONLY))

    await pipeline.run_cell(
        _INGESTED,
        PreparedCell(cell=_PICTURE_ONLY, memcell_id="mc_2", sender_ids=["u_betty"]),
    )

    system = _system_prompt(llm)
    assert _ENGLISH_RULE in system
    assert _PARTICIPANT_RULE not in system


async def test_the_conversation_s_language_is_pinned_not_the_reply_s() -> None:
    """A reply that drifted into Chinese does not drag the group's records after it."""
    llm = _replying(_CHINESE_PICTURE_REPLY)
    pipeline, episodes, members = await _pipeline(
        llm, ("mc_1", _EARLIER), ("mc_2", _PICTURE_ONLY)
    )

    outcome = await pipeline.run_cell(
        _INGESTED,
        PreparedCell(cell=_PICTURE_ONLY, memcell_id="mc_2", sender_ids=["u_betty"]),
    )

    (event,) = [e for e in outcome.events if isinstance(e, GroupEpisodeExtracted)]
    (written,) = [call.args[1] for call in episodes.append.await_args_list]
    assert event.language == "English"
    assert written.language == "English"
    # The poster's picture profile items freeze the profile's language.
    betty = await members.read(_GROUP, "u_betty")
    assert betty is not None
    assert betty.profile.primary_language == "English"


async def test_a_text_segment_is_told_its_language_too() -> None:
    llm = _replying(_CHINESE_EPISODE_REPLY)
    pipeline, _, _ = await _pipeline(llm, ("mc_1", _EARLIER))

    outcome = await pipeline.run_cell(
        _INGESTED,
        PreparedCell(
            cell=_EARLIER, memcell_id="mc_1", sender_ids=["u_chen", "u_betty"]
        ),
    )

    assert _ENGLISH_RULE in _system_prompt(llm)
    (event,) = [e for e in outcome.events if isinstance(e, GroupEpisodeExtracted)]
    assert event.language == "English"


async def test_a_chinese_group_is_told_chinese() -> None:
    chinese = _cell(
        _message("u_chen", "五月一号的机票卖完了，我们改成四月三十号出发吧。", at=_T0),
        _message("u_betty", "我没问题，我来订四月三十号开始的酒店。", at=_T0 + 1),
    )
    llm = _replying(_CHINESE_EPISODE_REPLY)
    pipeline, _, _ = await _pipeline(llm, ("mc_1", chinese))

    outcome = await pipeline.run_cell(
        _INGESTED,
        PreparedCell(cell=chinese, memcell_id="mc_1", sender_ids=["u_chen", "u_betty"]),
    )

    assert "Write every output field in Chinese" in _system_prompt(llm)
    (event,) = [e for e in outcome.events if isinstance(e, GroupEpisodeExtracted)]
    assert event.language == "Chinese"


async def test_a_group_session_without_text_leaves_the_choice_to_the_model() -> None:
    """Nothing to resolve from: the participant rule, and the reply's language."""
    llm = _replying(_CHINESE_PICTURE_REPLY)
    pipeline, _, _ = await _pipeline(llm, ("mc_2", _PICTURE_ONLY))

    outcome = await pipeline.run_cell(
        _INGESTED,
        PreparedCell(cell=_PICTURE_ONLY, memcell_id="mc_2", sender_ids=["u_betty"]),
    )

    assert _PARTICIPANT_RULE in _system_prompt(llm)
    (event,) = [e for e in outcome.events if isinstance(e, GroupEpisodeExtracted)]
    assert event.language == "Chinese"
