"""``ConversationPipeline.run_cell`` names the conversation's language to the model.

The defect this pins: with no language named, every extraction prompt asked
the model to work the language out (everalgo's participant rule), and the
2.0 backbone answered English conversations in Chinese -- a whole picture
cell at a time, descriptions included -- and the language read back off that
reply then pinned the cell's extractions and the profiles to Chinese. A
group segment goes through the same call, so it had the same defect.

The contract now, in a user's own space and in a group's alike:

- The language is resolved in code from the segment's human messages, and
  from its session's earlier segments when it has too little text of its
  own, before the call, and named in the prompt: a picture-only cell of an
  English chat gets an English instruction.
- The resolved language, not the reply's, is what the episode and its event
  carry, so the extractions downstream (the profile, the member-in-group
  profiles, the scenes) follow the conversation too.
- Only a session without a letter anywhere leaves the choice to the model,
  and then, as before, the reply's language is what gets pinned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.component.images import ImageLimits
from everos.memory import IngestResult
from everos.memory.events import EpisodeExtracted
from everos.memory.extract.pipeline import (
    CellOutcome,
    ConversationPipeline,
    PreparedCell,
)
from everos.memory.models import Group, MemcellRecord, RosterEvent
from everos.memory.ports import MemoryRef, SessionKey
from tests.fakes import bundled_instructions
from tests.fakes.stores import (
    InMemoryGroupStore,
    InMemoryImageMemoryStore,
    InMemoryImageStore,
    InMemoryMemcellStore,
)

_T0 = 1_700_000_000_000
_IMAGE_ID = "9f" + "0" * 62
_SESSION = SessionKey("default", "default", "s1")
_ENGLISH_RULE = "Write every output field in English"
_PARTICIPANT_RULE = "Write ALL output fields in the language the participants use"

_CHINESE_PICTURE_REPLY = {
    "content": "Betty 发来一张拿铁的照片（img-1）。",
    "summary": "Betty 分享了拿铁照片。",
    "title": "拿铁照片",
    "images": [{"ref": "img-1", "description": "一杯拉花拿铁", "ocr": ""}],
}
_CHINESE_EPISODE_REPLY = {
    "content": "他们周末重新粉刷了厨房橱柜。",
    "summary": "他们粉刷了橱柜。",
    "title": "粉刷橱柜",
}


@dataclass(frozen=True)
class _Space:
    """One space a segment can be remembered in, and its two speakers."""

    name: str
    group_id: str | None
    first: str
    second: str

    def ingested(self) -> IngestResult:
        return IngestResult(session_id="s1", messages=[], group_id=self.group_id)


_USER = _Space("user", None, "u1", "u1")
_GROUP = _Space("group", "trip", "u_chen", "u_betty")
_SPACES = pytest.mark.parametrize("space", [_USER, _GROUP], ids=["user", "group"])


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


def _earlier(space: _Space) -> MemCell:
    return _cell(
        _message(
            space.first, "I spent the weekend repainting the kitchen cabinets.", at=_T0
        ),
        _message(
            space.second,
            "They are a pale green now and I love how bright it is.",
            at=_T0 + 1,
        ),
    )


def _picture_only(space: _Space) -> MemCell:
    return _cell(_message(space.second, f"[IMAGE: {_IMAGE_ID}]", at=_T0 + 60_000))


def _replying(reply: dict) -> FakeLLMClient:
    return FakeLLMClient(handler=lambda *_a, **_k: json.dumps(reply))


async def _pipeline(
    llm: FakeLLMClient, *cells: tuple[str, MemCell]
) -> tuple[ConversationPipeline, MagicMock]:
    episodes = MagicMock()
    episodes.append = AsyncMock(
        return_value=MemoryRef(
            id="x_ep_20260517_00000001",
            entry_id="ep_20260517_00000001",
            md_path="default_app/default_project/x/episodes/episode-2026-05-17.md",
        )
    )
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
    pipeline = ConversationPipeline(
        episode_store=episodes,  # type: ignore[arg-type]  # recording double
        memcell_store=memcells,
        groups=groups,
        instructions=bundled_instructions(),
        llm_client=llm,  # type: ignore[arg-type]  # FakeLLMClient
        image_store=pictures,
        image_memory_store=InMemoryImageMemoryStore(),
        image_limits=_limits(),
    )
    return pipeline, episodes


def _system_prompt(llm: FakeLLMClient) -> str:
    (call,) = llm.calls
    return str(call.messages[0].content)


def _event(outcome: CellOutcome) -> EpisodeExtracted:
    (event,) = [e for e in outcome.events if isinstance(e, EpisodeExtracted)]
    return event


@_SPACES
async def test_an_english_chat_s_picture_only_cell_is_told_english(
    space: _Space,
) -> None:
    llm = _replying(_CHINESE_PICTURE_REPLY)
    picture = _picture_only(space)
    pipeline, _ = await _pipeline(llm, ("mc_1", _earlier(space)), ("mc_2", picture))

    await pipeline.run_cell(
        space.ingested(),
        PreparedCell(cell=picture, memcell_id="mc_2", sender_ids=[space.second]),
    )

    system = _system_prompt(llm)
    assert _ENGLISH_RULE in system
    assert _PARTICIPANT_RULE not in system


@_SPACES
async def test_the_conversation_s_language_is_pinned_not_the_reply_s(
    space: _Space,
) -> None:
    """A reply that drifted into Chinese does not drag the episode after it."""
    llm = _replying(_CHINESE_PICTURE_REPLY)
    picture = _picture_only(space)
    pipeline, episodes = await _pipeline(
        llm, ("mc_1", _earlier(space)), ("mc_2", picture)
    )

    outcome = await pipeline.run_cell(
        space.ingested(),
        PreparedCell(cell=picture, memcell_id="mc_2", sender_ids=[space.second]),
    )

    (written,) = [call.args[1] for call in episodes.append.await_args_list]
    assert _event(outcome).language == "English"
    assert written.language == "English"


@_SPACES
async def test_a_text_cell_is_told_its_language_too(space: _Space) -> None:
    llm = _replying(_CHINESE_EPISODE_REPLY)
    earlier = _earlier(space)
    pipeline, _ = await _pipeline(llm, ("mc_1", earlier))

    outcome = await pipeline.run_cell(
        space.ingested(),
        PreparedCell(
            cell=earlier, memcell_id="mc_1", sender_ids=[space.first, space.second]
        ),
    )

    assert _ENGLISH_RULE in _system_prompt(llm)
    assert _event(outcome).language == "English"


@_SPACES
async def test_a_chinese_chat_is_told_chinese(space: _Space) -> None:
    chinese = _cell(
        _message(
            space.first, "五月一号的机票卖完了，我们改成四月三十号出发吧。", at=_T0
        ),
        _message(space.second, "我没问题，我来订四月三十号开始的酒店。", at=_T0 + 1),
    )
    llm = _replying(_CHINESE_EPISODE_REPLY)
    pipeline, _ = await _pipeline(llm, ("mc_1", chinese))

    outcome = await pipeline.run_cell(
        space.ingested(),
        PreparedCell(
            cell=chinese, memcell_id="mc_1", sender_ids=[space.first, space.second]
        ),
    )

    assert "Write every output field in Chinese" in _system_prompt(llm)
    assert _event(outcome).language == "Chinese"


@_SPACES
async def test_a_session_without_text_leaves_the_choice_to_the_model(
    space: _Space,
) -> None:
    """Nothing to resolve from: the participant rule, and the reply's language."""
    llm = _replying(_CHINESE_PICTURE_REPLY)
    picture = _picture_only(space)
    pipeline, _ = await _pipeline(llm, ("mc_2", picture))

    outcome = await pipeline.run_cell(
        space.ingested(),
        PreparedCell(cell=picture, memcell_id="mc_2", sender_ids=[space.second]),
    )

    assert _PARTICIPANT_RULE in _system_prompt(llm)
    assert _event(outcome).language == "Chinese"
