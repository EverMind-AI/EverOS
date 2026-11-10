"""The picture route of ``UserMemoryPipeline.run_cell``.

What these pin — the contract that makes a picture a memory rather than an
attachment:

- One cell costs **one** extraction call whether or not it carries pictures.
  The call returns the episode and that cell's picture records together,
  because both are readings of the same conversation and splitting them would
  let the two drift.
- A picture memory is written for each user sender, before the episode, and
  its ``image_id`` is the file hash from the conversation text — never the
  per-cell label the reply used (``img-1``).
- A text-only cell runs everalgo's extractor unchanged. Nothing about the
  path that handles most cells moves.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from everalgo.types import ChatMessage, MemCell

from everos.component.images import ImageLimits
from everos.memory import IngestResult
from everos.memory.events import EpisodeExtracted
from everos.memory.extract.pipeline import PreparedCell, UserMemoryPipeline
from everos.memory.ports import MemoryRef
from tests.fakes import bundled_instructions, user_track_llm, user_track_script
from tests.fakes.stores import (
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
    profile: object | None = None,
) -> tuple[UserMemoryPipeline, InMemoryImageMemoryStore, InMemoryImageStore]:
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
    pipeline = UserMemoryPipeline(
        episode_store=episodes,  # type: ignore[arg-type]
        memcell_store=memcells,
        instructions=bundled_instructions(),
        llm_client=llm,  # type: ignore[arg-type]
        image_store=pictures if with_images else None,
        image_memory_store=memories if with_images else None,
        image_limits=_limits() if with_images else None,
        profile_store=profile,
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
    parts = llm.calls[0][1].content
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


async def test_the_pictures_profile_items_land_in_user_md() -> None:
    """The write-time half of the design: a photo moves the portrait now.

    The item records the picture it came from (as its durable hash, not the
    per-cell label), which is what lets a forget of that picture reach it.
    """
    from tests.fakes import InMemoryProfileStore

    reply = _image_reply() | {
        "profile": [
            {
                "action": "add",
                "type": "explicit_info",
                "source_image_id": "img-1",
                "data": {
                    "category": "pets",
                    "description": "Has a cat named Biscuit",
                    "evidence": "2026-05-14: the collar tag reads Biscuit",
                },
            }
        ]
    }
    profiles = InMemoryProfileStore()
    pipeline, _, _ = _pipeline(_RecordingLLM(reply), profile=profiles)
    cell = MemCell(
        items=[_text_message(), _picture_message()], timestamp=1_700_000_000_000
    )

    await pipeline.run_cell(
        _INGESTED, PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1"])
    )

    scope = next(iter(profiles.profiles))
    item = profiles.profiles[scope].explicit_info[0]
    assert item.description == "Has a cat named Biscuit"
    assert item.model_extra["source_image_ids"] == [_IMAGE_ID]


async def test_a_text_only_cell_never_touches_the_profile() -> None:
    from tests.fakes import InMemoryProfileStore

    profiles = InMemoryProfileStore()
    pipeline, _, _ = _pipeline(user_track_llm(), profile=profiles)
    cell = MemCell(items=[_text_message()], timestamp=1_700_000_000_000)

    await pipeline.run_cell(
        _INGESTED, PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1"])
    )

    assert profiles.writes == []


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


# ── D17: a picture's profile items go to its poster only ─────────────────


def _profile_reply(*ops: dict) -> dict:
    return _image_reply() | {"profile": list(ops)}


def _cat_fact(label: str = "img-1") -> dict:
    return {
        "action": "add",
        "type": "explicit_info",
        "source_image_id": label,
        "data": {
            "category": "pets",
            "description": "Has a cat named Biscuit",
            "evidence": "the collar tag reads Biscuit",
        },
    }


async def test_a_pictures_profile_items_reach_only_its_poster() -> None:
    """The leak this fixes (docs/group-memory.md, D17): before, every human
    sender of the cell had the poster's cat merged into their own profile."""
    from tests.fakes import InMemoryProfileStore

    profiles = InMemoryProfileStore()
    pipeline, memories, _ = _pipeline(
        _RecordingLLM(_profile_reply(_cat_fact())), profile=profiles
    )
    cell = MemCell(
        items=[_text_message("u1"), _text_message("u2"), _picture_message("u1")],
        timestamp=1_700_000_000_000,
    )

    await pipeline.run_cell(
        _INGESTED, PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1", "u2"])
    )

    owners = {scope.owner_id for scope in profiles.profiles}
    assert owners == {"u1"}
    # The picture memories themselves are unchanged: one copy per sender.
    assert len(memories.items_for("u2")) == 1


async def test_an_assistants_picture_moves_no_users_profile() -> None:
    """A photo says something about the person who sent it; the assistant is
    not a user, so its picture establishes nothing about the user."""
    from tests.fakes import InMemoryProfileStore

    profiles = InMemoryProfileStore()
    pipeline, _, _ = _pipeline(
        _RecordingLLM(_profile_reply(_cat_fact())), profile=profiles
    )
    bot_picture = ChatMessage(
        id="m_3",
        role="assistant",
        content=f"[IMAGE: {_IMAGE_ID}]",
        timestamp=1_700_000_000_000,
        sender_id="bot",
    )
    cell = MemCell(
        items=[_text_message("u1"), bot_picture], timestamp=1_700_000_000_000
    )

    await pipeline.run_cell(
        _INGESTED, PreparedCell(cell=cell, memcell_id="mc_a", sender_ids=["u1", "bot"])
    )

    assert profiles.profiles == {}
