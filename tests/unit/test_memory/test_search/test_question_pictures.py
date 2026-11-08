"""Pictures attached to the question rather than to the memory.

What these pin:

- A question picture is stored content-addressed and establishes nothing: no
  picture memory, no reference count, no episode. Its row starts at zero, so a
  later forget never under-releases a real arrival.
- The decider sees it on every round, ahead of the candidates, and the
  per-round candidate cap does not apply to it — a cap that dropped it would
  leave the decider answering a question it cannot see.
- A picture the store rejects is dropped, and the text question is still
  answered.
"""

from __future__ import annotations

import base64
import io

from PIL import Image

from everos.component.images import ImageLimits
from everos.memory.extract.ingest.multimodal import (
    PictureBudget,
    keep_question_pictures,
)
from everos.memory.search.dto import QuestionPicture
from everos.memory.search.llm_multiround import Picture, _content, _round_pictures


def _limits() -> ImageLimits:
    return ImageLimits(
        max_bytes=5 * 1024 * 1024,
        max_per_message=4,
        max_request_bytes=10 * 1024 * 1024,
        max_per_request=4,
        formats=("png", "jpeg", "webp", "gif"),
    )


def _png(colour: tuple[int, int, int] = (10, 120, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), colour).save(buffer, format="PNG")
    return buffer.getvalue()


async def _unused_uri(item: dict) -> object:
    """A raw-file builder that must not be called for base64 pictures."""
    raise AssertionError("base64 pictures never need the host file hydrator")


async def test_a_question_picture_is_kept_without_a_reference() -> None:
    from tests.fakes import InMemoryImageStore

    store = InMemoryImageStore()
    budget = PictureBudget(max_bytes=1 << 20, max_pictures=4)

    stored = await keep_question_pictures(
        [QuestionPicture(base64=base64.b64encode(_png()).decode())],
        store=store,
        limits=_limits(),
        raw_file_builder=_unused_uri,
        budget=budget,
    )

    assert len(stored) == 1
    assert stored[0].image_id in store.images
    # No arrival was counted: a question is not a fact about the user.
    assert store.refs == {}


async def test_unreadable_question_pictures_are_dropped_not_raised() -> None:
    from tests.fakes import InMemoryImageStore

    store = InMemoryImageStore()

    stored = await keep_question_pictures(
        [QuestionPicture(base64="bm90IGFuIGltYWdl")],  # b"not an image"
        store=store,
        limits=_limits(),
        raw_file_builder=_unused_uri,
        budget=PictureBudget(max_bytes=1 << 20, max_pictures=4),
    )

    assert stored == []


async def test_without_a_store_question_pictures_are_ignored() -> None:
    stored = await keep_question_pictures(
        [QuestionPicture(base64="aGk=")],
        store=None,
        limits=_limits(),
        raw_file_builder=_unused_uri,
        budget=PictureBudget(max_bytes=1 << 20, max_pictures=4),
    )

    assert stored == []


async def test_the_question_picture_is_shown_before_the_candidates() -> None:
    from tests.fakes import InMemoryImageStore

    store = InMemoryImageStore()
    store.images["q" * 64] = (b"question", "jpeg")

    pictures = await _round_pictures(
        [],
        store=store,
        k=0,
        edge=448,
        quality=85,
        question_image_ids=["q" * 64],
    )

    assert len(pictures) == 1
    assert pictures[0].question is True


async def test_the_question_picture_is_not_charged_to_the_candidate_cap() -> None:
    """``k`` bounds candidates only: the picture the question is about survives."""
    from everalgo.types import Candidate

    from tests.fakes import InMemoryImageStore

    store = InMemoryImageStore()
    question_id = "q" * 64
    store.images[question_id] = (b"question", "jpeg")
    candidates = []
    for index in range(3):
        image_id = str(index) * 64
        store.images[image_id] = (b"candidate", "jpeg")
        candidates.append(
            Candidate(
                id=f"u_im_{index}",
                score=1.0 - index / 10,
                metadata={"kind": "image_memory", "image_id": image_id},
            )
        )

    pictures = await _round_pictures(
        candidates,
        store=store,
        k=2,
        edge=448,
        quality=85,
        question_image_ids=[question_id],
    )

    assert [p.question for p in pictures] == [True, False, False]
    assert len(pictures) == 3  # one question picture + the two the cap allows


def test_the_decider_message_labels_the_question_picture_distinctly() -> None:
    from everos.config.settings import DeciderSettings

    tune = DeciderSettings()
    content = _content(
        "prompt",
        [Picture(label="", data=b"x", mime="image/jpeg", question=True)],
        tune,
    )

    assert isinstance(content, list)
    captions = [part["text"] for part in content if part["type"] == "text"]
    assert "Picture attached to the question:" in captions
