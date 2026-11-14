"""Pictures in ``ConversationPipeline.run_cell``: one call, one prompt by rule.

What these pin — the contract that makes a picture a memory rather than an
attachment:

- One cell costs **one** extraction call whether or not it carries pictures.
  The call returns the episode and that cell's picture records together,
  because both are readings of the same conversation and splitting them would
  let the two drift.
- A picture memory is written in the cell's space, before the episode: once
  in a group's, once for each human sender outside one. Its ``image_id`` is
  the file hash from the conversation text — never the per-cell label the
  reply used (``img-1``). (A group's picture memory, dated by its message and
  attributed to its poster, is pinned by
  ``tests/unit/test_service/test_group_write_path.py``.)
- A text-only cell makes the same call, sent the original episode request
  (everalgo-user-memory 0.3.1's) and the language block; it writes no picture
  memory. A cell with pictures is sent the same request first, then the
  picture block, then the pictures.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory.prompts.en.episode_original import (
    EPISODE_GENERATION_PROMPT,
)

from everos.component.images import ImageLimits
from everos.memory import IngestResult
from everos.memory.events import EpisodeExtracted
from everos.memory.extract.pipeline import ConversationPipeline, PreparedCell
from everos.memory.ports import MemoryRef
from tests.fakes import bundled_instructions, user_track_llm, user_track_script
from tests.fakes.stores import (
    InMemoryGroupStore,
    InMemoryImageMemoryStore,
    InMemoryImageStore,
    InMemoryMemcellStore,
)

_IMAGE_ID = "9f" + "0" * 62
_EPISODE_REPLY = json.loads(user_track_script("episode"))
_REF = MemoryRef(
    id="u1_ep_20260517_0000000001",
    entry_id="ep_20260517_0000000001",
    md_path="default_app/default_project/users/u1/episodes/episode-2026-05-17.md",
)


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


def _text_message(sender: str = "u1") -> ChatMessage:
    return ChatMessage(
        id="m_1",
        role="user",
        content="look at this",
        timestamp=1_700_000_000_000,
        sender_id=sender,
    )


def _picture_message(sender: str = "u1") -> ChatMessage:
    """A message whose only content is the picture reference ingest wrote."""
    return ChatMessage(
        id="m_2",
        role="user",
        content=f"[IMAGE: {_IMAGE_ID}]",
        timestamp=1_700_000_000_000,
        sender_id=sender,
    )


class _RecordingLLM:
    """An everalgo-shaped client that records the messages it was shown."""

    def __init__(self, reply: dict) -> None:
        self.reply = reply
        self.calls: list[list[object]] = []

    async def achat(self, messages, *, response_format, **extra):
        self.calls.append(list(messages))
        return json.dumps(self.reply)


class _ExtractorLLM(_RecordingLLM):
    """Answers whatever the operator asks, in the operator's own shape."""

    def __init__(self) -> None:
        super().__init__(_EPISODE_REPLY)


def _image_reply() -> dict:
    return {
        "content": (
            "The user sent a photo of a latte (img-1) and asked what was in it."
        ),
        "summary": "The user shared a latte photo.",
        "title": "Latte Photo",
        "images": [
            {
                "ref": "img-1",
                "description": "a latte with heart-shaped foam art",
                "ocr": "",
            }
        ],
    }


def _pipeline(
    llm: object,
    *,
    with_images: bool = True,
) -> tuple[ConversationPipeline, InMemoryImageMemoryStore, InMemoryImageStore]:
    episodes = MagicMock()
    episodes.append = AsyncMock(return_value=_REF)
    episodes.append_many = AsyncMock(return_value=[_REF])
    memcells = InMemoryMemcellStore()
    # The cell exists, so the picture write passes its source check -- the same
    # precondition a real extraction runs under.
    memcells.seed("mc_a", MemCell(items=[_text_message()], timestamp=1_700_000_000_000))
    pictures = InMemoryImageStore()
    memories = InMemoryImageMemoryStore()
    pictures.images[_IMAGE_ID] = (b"picture bytes", "png")
    pictures.refs[_IMAGE_ID] = 1
    pipeline = ConversationPipeline(
        episode_store=episodes,  # type: ignore[arg-type]
        memcell_store=memcells,
        groups=InMemoryGroupStore(),
        instructions=bundled_instructions(),
        llm_client=llm,  # type: ignore[arg-type]
        image_store=pictures if with_images else None,
        image_memory_store=memories if with_images else None,
        image_limits=_limits() if with_images else None,
    )
    return pipeline, memories, pictures


_INGESTED = IngestResult(session_id="s1", messages=[])


async def test_a_text_only_cell_writes_no_picture_memory() -> None:
    llm = user_track_llm()
    pipeline, memories, _ = _pipeline(llm)
    cell = MemCell(items=[_text_message()], timestamp=1_700_000_000_000)

    outcome = await pipeline.run_cell(
        _INGESTED, PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1"])
    )

    assert any(isinstance(e, EpisodeExtracted) for e in outcome.events)
    assert memories.records == []


async def test_a_picture_cell_writes_one_memory_per_picture() -> None:
    llm = _RecordingLLM(_image_reply())
    pipeline, memories, _ = _pipeline(llm)
    cell = MemCell(
        items=[_text_message(), _picture_message()], timestamp=1_700_000_000_000
    )

    await pipeline.run_cell(
        _INGESTED, PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1"])
    )

    assert len(memories.records) == 1
    memory = memories.records[0].item
    assert memory.image_id == _IMAGE_ID
    assert memory.description == "a latte with heart-shaped foam art"
    assert memory.owner_id == "u1"
    assert memory.parent_id == "mc_a"


async def test_a_text_cell_and_a_picture_cell_share_the_original_request() -> None:
    """One prompt by rule: the picture block only where pictures are attached."""
    text_llm, picture_llm = _RecordingLLM(_EPISODE_REPLY), _RecordingLLM(_image_reply())
    for llm, items in (
        (text_llm, [_text_message()]),
        (picture_llm, [_text_message(), _picture_message()]),
    ):
        pipeline, _, _ = _pipeline(llm)
        cell = MemCell(items=items, timestamp=1_700_000_000_000)
        await pipeline.run_cell(
            _INGESTED, PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1"])
        )

    (text_call,), (picture_call,) = text_llm.calls, picture_llm.calls
    # A one-to-one text cell: the original request, one user message.
    (text_user,) = text_call
    assert text_user.role == "user"
    assert text_user.content.startswith(EPISODE_GENERATION_PROMPT[:60])
    assert "<pictures>" not in text_user.content
    assert text_user.content.endswith("</output_language>")
    (picture_user,) = picture_call
    picture_text = picture_user.content[0].text
    assert picture_text.startswith(EPISODE_GENERATION_PROMPT[:60])
    assert "\n<pictures>\nThis message carries pictures" in picture_text
    assert picture_text.endswith("</output_language>")


async def test_one_call_answers_for_the_episode_and_the_pictures() -> None:
    llm = _RecordingLLM(_image_reply())
    pipeline, _, _ = _pipeline(llm)
    cell = MemCell(
        items=[_text_message(), _picture_message()], timestamp=1_700_000_000_000
    )

    await pipeline.run_cell(
        _INGESTED, PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1"])
    )

    assert len(llm.calls) == 1
    parts = llm.calls[0][0].content
    assert any(getattr(p, "type", "") == "image_url" for p in parts)


async def test_the_episode_body_refers_to_the_picture_by_id() -> None:
    """The reference, not a description, is what the body carries."""
    llm = _RecordingLLM(_image_reply())
    pipeline, _, _ = _pipeline(llm)
    cell = MemCell(
        items=[_text_message(), _picture_message()], timestamp=1_700_000_000_000
    )

    outcome = await pipeline.run_cell(
        _INGESTED, PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1"])
    )

    written = outcome.events[-1]
    assert isinstance(written, EpisodeExtracted)
    # The label the model used stays (it reads naturally) and the picture's
    # durable id is attached: the label means nothing outside this cell.
    assert f"(img-1: {_IMAGE_ID})" in written.episode_text


async def test_each_user_sender_gets_their_own_copy() -> None:
    """Two people without a group: one call, a picture memory for each."""
    llm = _RecordingLLM(_image_reply())
    pipeline, memories, _ = _pipeline(llm)
    cell = MemCell(
        items=[_text_message("u1"), _text_message("u2"), _picture_message("u1")],
        timestamp=1_700_000_000_000,
    )

    await pipeline.run_cell(
        _INGESTED, PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1", "u2"])
    )

    owners = [memory.owner_id for memory in memories.items_for("u1")]
    assert owners == ["u1"]
    assert len(memories.items_for("u2")) == 1
    assert len(llm.calls) == 1


async def test_the_memory_keeps_the_name_the_client_sent() -> None:
    """The client's file name is provenance of this occurrence, not of the file.

    The cell text keeps only the content hash, so the name has to reach the
    memory from the content items through the boundary's per-cell map.
    """
    llm = _RecordingLLM(_image_reply())
    pipeline, memories, _ = _pipeline(llm)
    cell = MemCell(
        items=[_text_message(), _picture_message()], timestamp=1_700_000_000_000
    )

    await pipeline.run_cell(
        _INGESTED,
        PreparedCell(
            cell=cell,
            memcell_id="mc_a",
            sender_ids=["u1"],
            picture_names={_IMAGE_ID: "IMG_2231.jpg"},
        ),
    )

    assert memories.records[0].item.name == "IMG_2231.jpg"


async def test_a_picture_without_a_name_stores_an_empty_one() -> None:
    """An inline image part usually carries no file name; that is not an error."""
    llm = _RecordingLLM(_image_reply())
    pipeline, memories, _ = _pipeline(llm)
    cell = MemCell(
        items=[_text_message(), _picture_message()], timestamp=1_700_000_000_000
    )

    await pipeline.run_cell(
        _INGESTED, PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1"])
    )

    assert memories.records[0].item.name == ""


async def test_the_picture_call_writes_no_profile_item() -> None:
    """One profile writer: the profile update reads the segment's pictures itself.

    The picture call is asked for the episode and the pictures' memories only;
    a reply that still carries an empty ``profile`` key reads as without it.
    """
    llm = _RecordingLLM(_image_reply() | {"profile": []})
    pipeline, memories, _ = _pipeline(llm)
    cell = MemCell(
        items=[_text_message(), _picture_message()], timestamp=1_700_000_000_000
    )

    await pipeline.run_cell(
        _INGESTED, PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1"])
    )

    prompt = llm.calls[0][0].content[0].text
    assert "<profile>" not in prompt and '"profile"' not in prompt
    assert len(memories.items_for("u1")) == 1


async def test_a_picture_the_reply_never_named_is_still_referenced() -> None:
    """The body always names the pictures it carries, even when the model forgets.

    A forget finds the episodes to rewrite *through* that reference: without it
    the picture's content would stay in the episode body for ever.
    """
    reply = _image_reply() | {
        "content": "The user shared a photo and asked what was in it."
    }
    llm = _RecordingLLM(reply)
    pipeline, _, _ = _pipeline(llm)
    cell = MemCell(
        items=[_text_message(), _picture_message()], timestamp=1_700_000_000_000
    )

    outcome = await pipeline.run_cell(
        _INGESTED, PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1"])
    )

    body = outcome.events[-1].episode_text
    assert f"(img-1: {_IMAGE_ID})" in body
