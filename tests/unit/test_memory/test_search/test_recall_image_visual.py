"""The picture recaller's third route: ANN over the picture's own embedding.

What these pin:

- ``visual_recall`` scores ``visual_vector`` — a different column from the
  description's ``vector`` — and produces the same ``image_memory``-tagged
  candidates the other routes do, so the mixed ranking treats it identically.
- An empty vector (no vision embedder configured) recalls nothing rather than
  querying a column that is all NULL.
"""

from __future__ import annotations

import datetime as _dt

from everos.component.tokenizer import Tokenizer
from everos.memory.models import ImageMemory
from everos.memory.ports import MemoryRef, Scope, SearchHit, Stored
from everos.memory.search.recall.base import RecallerDeps
from everos.memory.search.recall.image_memory import IMAGE_KIND, ImageMemoryRecaller
from tests.unit.test_memory.test_search._records import ScriptedIndex

_SCOPE = Scope(app_id="default", project_id="default", owner_id="u1", owner_type="user")


class _Tokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


def _picture(image_id: str = "a" * 64) -> SearchHit[ImageMemory]:
    stored = Stored(
        ref=MemoryRef(
            id="u1_im_1", entry_id="im_1", md_path="users/u1/images/image.md"
        ),
        scope=_SCOPE,
        item=ImageMemory(
            owner_id="u1",
            description="a red bicycle leaning on a wall",
            timestamp=int(_dt.datetime(2026, 5, 14, tzinfo=_dt.UTC).timestamp() * 1000),
            image_id=image_id,
            original_path=f"figures/aa/{image_id}.png",
            media_type="png",
            width=640,
            height=480,
        ),
    )
    return SearchHit(stored=stored, score=0.83)


async def test_the_visual_route_scores_the_picture_vector() -> None:
    index = ScriptedIndex(visual=[_picture()])
    recaller = ImageMemoryRecaller(RecallerDeps(tokenizer=_Tokenizer()), index)  # type: ignore[arg-type]

    result = await recaller.visual_recall([0.4] * 8, _SCOPE.to_filter(), limit=5)

    route, vector, _where, limit = index.calls[0]
    assert route == "visual"
    assert vector == [0.4] * 8
    assert limit == 5
    assert [c.metadata["kind"] for c in result.candidates] == [IMAGE_KIND]


async def test_an_empty_visual_query_recalls_nothing() -> None:
    index = ScriptedIndex()
    recaller = ImageMemoryRecaller(RecallerDeps(tokenizer=_Tokenizer()), index)  # type: ignore[arg-type]

    result = await recaller.visual_recall([], _SCOPE.to_filter(), limit=5)

    assert result.candidates == []
    assert index.calls == []
