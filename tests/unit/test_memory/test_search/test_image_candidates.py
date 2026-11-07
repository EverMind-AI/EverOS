"""Pictures in the multi-round decider's view.

What these pin — the retrieval contract the picture path has to honour:

- A picture candidate is rendered in the evidence like any other, marked so the
  decider knows it is a picture; the label it carries is the evidence index, so
  an attached image can be tied back to its line.
- Pictures the decider may look at are **this round's** first-k in evidence
  order, and only when ``[decider].image_candidates_per_round`` is above 0.
  Everything past the cap stays a text candidate.
- A picture that cannot be read degrades to text; it never fails the round.
- The two kinds share one ranking and one ``top_k``, then are split into
  ``episodes`` and ``images`` for the response.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from everalgo.types import Candidate

from everos.config.search_runtime import SearchConfigSource
from everos.memory.models import ImageMemory
from everos.memory.ports import MemoryRef, Scope, Stored
from everos.memory.search.llm_multiround import (
    Picture,
    _content,
    _render_blocks,
    _round_pictures,
)
from everos.memory.search.recall.image_memory import IMAGE_KIND

_CONFIG = SearchConfigSource()


def _tune(**overrides: Any):
    tune = _CONFIG.tuning()
    return tune.model_copy(update=overrides) if overrides else tune


class _Store:
    """A picture store that can read some ids and not others."""

    def __init__(self, *, unreadable: set[str] = frozenset()) -> None:
        self.unreadable = set(unreadable)
        self.read: list[str] = []

    async def read_derived(self, image_id: str, *, edge: int, quality: int) -> bytes:
        self.read.append(image_id)
        if image_id in self.unreadable:
            raise FileNotFoundError(image_id)
        return b"thumb"


def _image_candidate(entry_id: str, image_id: str, score: float = 0.5) -> Candidate:
    item = ImageMemory(
        owner_id="u1",
        description="a latte with a heart on it",
        timestamp=1_700_000_000_000,
        image_id=image_id,
        original_relpath=f"figures/{image_id[:2]}/{image_id}.png",
        parent_id="mc_1",
        entry_id=entry_id,
    )
    return Candidate(
        id=f"u1_{entry_id}",
        score=score,
        source="vector",
        metadata={
            **item.model_dump(mode="python"),
            "kind": IMAGE_KIND,
            "subject": "a latte",
            "summary": "a latte with a heart on it",
        },
    )


def _episode_candidate(entry_id: str, score: float = 0.6) -> Candidate:
    return Candidate(
        id=f"u1_{entry_id}",
        score=score,
        source="vector",
        metadata={"subject": "Coffee", "summary": "they talked about coffee"},
    )


def test_a_picture_line_is_marked_in_the_evidence() -> None:
    rendered, global_cands, _meta = _render_blocks(
        [
            (
                "original question",
                [_episode_candidate("ep_1"), _image_candidate("im_1", "a" * 64)],
            )
        ],
        tuning_source=_CONFIG.tuning,
    )

    assert "Coffee - they talked about coffee" in rendered
    # The mark is what tells the decider the second item is a picture.
    assert rendered.count("(picture)") == 1
    assert len(global_cands) == 2


def test_the_description_is_what_the_decider_reads() -> None:
    rendered, _, _ = _render_blocks(
        [("q", [_image_candidate("im_1", "a" * 64)])],
        tuning_source=_CONFIG.tuning,
    )

    assert "a latte with a heart on it" in rendered


async def test_no_pictures_are_shown_unless_the_knob_is_on() -> None:
    """The default is off: an untrained decider is not shown pictures."""
    store = _Store()

    picked = await _round_pictures(
        [_image_candidate("im_1", "a" * 64)],
        store=store,
        k=0,
        edge=448,
        quality=85,
    )

    assert picked == []
    assert store.read == []


async def test_the_cap_keeps_the_first_pictures_in_evidence_order() -> None:
    store = _Store()
    cands = [
        _episode_candidate("ep_1"),
        _image_candidate("im_1", "a" * 64),
        _image_candidate("im_2", "b" * 64),
        _image_candidate("im_3", "c" * 64),
    ]

    picked = await _round_pictures(cands, store=store, k=2, edge=448, quality=85)

    assert [p.label for p in picked] == ["1", "2"]
    assert store.read == ["a" * 64, "b" * 64]


async def test_an_unreadable_picture_degrades_to_text() -> None:
    store = _Store(unreadable={"a" * 64})

    picked = await _round_pictures(
        [_image_candidate("im_1", "a" * 64), _image_candidate("im_2", "b" * 64)],
        store=store,
        k=2,
        edge=448,
        quality=85,
    )

    assert [p.label for p in picked] == ["1"]


def test_a_text_only_round_sends_the_plain_string_it_always_did() -> None:
    """Byte-identical requests for a deployment that shows no pictures."""
    assert _content("prompt", [], _tune()) == "prompt"


def test_pictures_ride_after_the_text_with_their_evidence_index() -> None:
    content = _content(
        "prompt", [Picture(label="3", data=b"x", mime="image/jpeg")], _tune()
    )

    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "prompt"}
    assert "evidence item 3" in content[1]["text"]
    assert content[2]["type"] == "image_url"
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def _stored_image(entry_id: str, image_id: str) -> Stored[ImageMemory]:
    return Stored(
        ref=MemoryRef(
            id=f"u1_{entry_id}", entry_id=entry_id, md_path="users/u1/images/i.md"
        ),
        scope=Scope(
            app_id="default", project_id="default", owner_id="u1", owner_type="user"
        ),
        item=ImageMemory(
            owner_id="u1",
            description="d",
            timestamp=int(_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC).timestamp() * 1000),
            image_id=image_id,
            original_relpath=f"figures/{image_id[:2]}/{image_id}.png",
            entry_id=entry_id,
        ),
    )


def test_shape_images_carries_both_paths_and_the_file_id() -> None:
    from everos.memory.search.shaper import shape_images

    image_id = "d" * 64
    stored = _stored_image("im_1", image_id)
    ranked = [Candidate(id=stored.ref.id, score=0.7, source="vector", metadata={})]

    shaped = shape_images(ranked, {stored.ref.id: stored}, thumbnail_edge=448)

    assert len(shaped) == 1
    item = shaped[0]
    assert item.image_id == image_id
    assert item.entry_id == "im_1"
    assert item.original_path == f"figures/dd/{image_id}.png"
    # One shared cache directory, named by hash + size: disposable, and a
    # single directory to drop when the copies are rebuilt.
    assert item.thumbnail_path == f"figures/.cache/{image_id}-448.jpeg"
    assert item.status == "current"
