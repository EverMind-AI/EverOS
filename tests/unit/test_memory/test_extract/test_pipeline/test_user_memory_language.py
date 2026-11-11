"""``UserMemoryPipeline.run_cell`` names the conversation's language to the model.

The defect this pins: with no language named, every extraction prompt asked
the model to work the language out (everalgo's participant rule), and the
2.0 backbone answered English conversations in Chinese -- a whole picture
cell at a time, descriptions included -- and the language read back off that
reply then pinned the cell's extractions and the owner's profile to Chinese.

The contract now:

- The language is resolved in code from the session's human messages, before
  the call, and named in the prompt: a picture-only cell of an English chat
  gets an English instruction.
- The resolved language, not the reply's, is what the episode and its event
  carry, so the extractions downstream follow the conversation too.
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
from everos.memory.events import EpisodeExtracted
from everos.memory.extract.pipeline import PreparedCell, UserMemoryPipeline
from everos.memory.models import MemcellRecord
from everos.memory.ports import MemoryRef, SessionKey
from tests.fakes import bundled_instructions
from tests.fakes.stores import (
    InMemoryImageMemoryStore,
    InMemoryImageStore,
    InMemoryMemcellStore,
)

_T0 = 1_700_000_000_000
_IMAGE_ID = "9f" + "0" * 62
_SESSION = SessionKey("default", "default", "s1")
_INGESTED = IngestResult(session_id="s1", messages=[])
_REF = MemoryRef(
    id="u1_ep_20260517_00000001",
    entry_id="ep_20260517_00000001",
    md_path="default_app/default_project/users/u1/episodes/episode-2026-05-17.md",
)
_ENGLISH_RULE = "Write every output field in English"
_PARTICIPANT_RULE = "Write ALL output fields in the language the participants use"

_CHINESE_PICTURE_REPLY = {
    "content": "用户发来一张拿铁的照片（img-1）。",
    "summary": "用户分享了拿铁照片。",
    "title": "拿铁照片",
    "images": [{"ref": "img-1", "description": "一杯拉花拿铁", "ocr": ""}],
}
_CHINESE_EPISODE_REPLY = {
    "content": "用户周末重新粉刷了厨房橱柜。",
    "summary": "用户粉刷了橱柜。",
    "title": "粉刷橱柜",
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


def _message(content: str, *, at: int) -> ChatMessage:
    return ChatMessage(
        id=f"m_{at}", role="user", content=content, timestamp=at, sender_id="u1"
    )


def _cell(*messages: ChatMessage) -> MemCell:
    return MemCell(items=list(messages), timestamp=messages[-1].timestamp)


_EARLIER = _cell(
    _message("I spent the weekend repainting the kitchen cabinets.", at=_T0),
    _message("They are a pale green now and I love how bright it is.", at=_T0 + 1),
)
_PICTURE_ONLY = _cell(_message(f"[IMAGE: {_IMAGE_ID}]", at=_T0 + 60_000))


def _replying(reply: dict) -> FakeLLMClient:
    return FakeLLMClient(handler=lambda *_a, **_k: json.dumps(reply))


async def _pipeline(
    llm: FakeLLMClient, *cells: tuple[str, MemCell]
) -> tuple[UserMemoryPipeline, MagicMock]:
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
    pictures = InMemoryImageStore()
    pictures.images[_IMAGE_ID] = (b"picture bytes", "png")
    pictures.refs[_IMAGE_ID] = 1
    pipeline = UserMemoryPipeline(
        episode_store=episodes,  # type: ignore[arg-type]  # recording double
        memcell_store=memcells,
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


async def test_an_english_chat_s_picture_only_cell_is_told_english() -> None:
    llm = _replying(_CHINESE_PICTURE_REPLY)
    pipeline, _ = await _pipeline(llm, ("mc_1", _EARLIER), ("mc_2", _PICTURE_ONLY))

    await pipeline.run_cell(
        _INGESTED,
        PreparedCell(cell=_PICTURE_ONLY, memcell_id="mc_2", sender_ids=["u1"]),
    )

    system = _system_prompt(llm)
    assert _ENGLISH_RULE in system
    assert _PARTICIPANT_RULE not in system


async def test_the_conversation_s_language_is_pinned_not_the_reply_s() -> None:
    """A reply that drifted into Chinese does not drag the episode after it."""
    llm = _replying(_CHINESE_PICTURE_REPLY)
    pipeline, episodes = await _pipeline(
        llm, ("mc_1", _EARLIER), ("mc_2", _PICTURE_ONLY)
    )

    outcome = await pipeline.run_cell(
        _INGESTED,
        PreparedCell(cell=_PICTURE_ONLY, memcell_id="mc_2", sender_ids=["u1"]),
    )

    (event,) = [e for e in outcome.events if isinstance(e, EpisodeExtracted)]
    (written,) = [call.args[1] for call in episodes.append.await_args_list]
    assert event.language == "English"
    assert written.language == "English"


async def test_a_text_cell_is_told_its_language_too() -> None:
    llm = _replying(_CHINESE_EPISODE_REPLY)
    pipeline, _ = await _pipeline(llm, ("mc_1", _EARLIER))

    outcome = await pipeline.run_cell(
        _INGESTED, PreparedCell(cell=_EARLIER, memcell_id="mc_1", sender_ids=["u1"])
    )

    assert _ENGLISH_RULE in _system_prompt(llm)
    (event,) = [e for e in outcome.events if isinstance(e, EpisodeExtracted)]
    assert event.language == "English"


async def test_a_chinese_chat_is_told_chinese() -> None:
    chinese = _cell(
        _message("这个周末我把厨房的橱柜重新刷了一遍，现在是浅绿色的。", at=_T0)
    )
    llm = _replying(_CHINESE_EPISODE_REPLY)
    pipeline, _ = await _pipeline(llm, ("mc_1", chinese))

    await pipeline.run_cell(
        _INGESTED, PreparedCell(cell=chinese, memcell_id="mc_1", sender_ids=["u1"])
    )

    assert "Write every output field in Chinese" in _system_prompt(llm)


async def test_a_session_without_text_leaves_the_choice_to_the_model() -> None:
    """Nothing to resolve from: the participant rule, and the reply's language."""
    llm = _replying(_CHINESE_PICTURE_REPLY)
    pipeline, _ = await _pipeline(llm, ("mc_2", _PICTURE_ONLY))

    outcome = await pipeline.run_cell(
        _INGESTED,
        PreparedCell(cell=_PICTURE_ONLY, memcell_id="mc_2", sender_ids=["u1"]),
    )

    assert _PARTICIPANT_RULE in _system_prompt(llm)
    (event,) = [e for e in outcome.events if isinstance(e, EpisodeExtracted)]
    assert event.language == "Chinese"
