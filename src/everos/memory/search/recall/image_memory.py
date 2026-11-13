"""Image-memory recaller — BM25 over description+text, ANN over two vectors.

Three routes, because a picture carries two embeddings from two models:

- BM25 over the description and the transcribed text. Not a fallback here, it
  is the load-bearing one for screenshots and signs: the words *in* the
  picture ("EST. 2019", "Transfer", "Total 28.00") exist in no other memory.
- ANN over ``vector`` — the description embedded by the text model. Answers
  "a question worded like the description".
- ANN over ``visual_vector`` — the picture's bytes embedded by the vision
  model, queried by that same model's text encoder. Answers "a question about
  what the picture looks like", including pictures whose description never
  names the thing asked about.

The two vector routes are never compared by score — different models,
different spaces — only fused by rank upstream. ``visual_recall`` returns
empty on a deployment with no ``[vision_embedding]``, which degrades this
class to the previous behaviour exactly.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import ClassVar

from everos.memory.models import ImageMemory
from everos.memory.ports import (
    Filter,
    ImageMemoryIndex,
    Stored,
    all_of,
    one_of,
)

from .base import (
    RecallerDeps,
    RecallResult,
    find_entries,
    from_hits,
    stable_sparse_top,
)

IMAGE_KIND = "image_memory"
"""``Candidate.metadata["kind"]`` stamped on every picture candidate.

The search loop fuses picture and episode candidates into one ranking, so
something has to say which is which when the two are shaped back into separate
DTO arrays and when the decider is deciding which ones to show as images.
Episodes carry no ``kind`` key, which makes its presence the discriminator.
"""


class ImageMemoryRecaller:
    """BM25 + vector recall over the picture-memory index.

    Args:
        deps: Shared recaller dependencies (tokenizer).
        index: The picture-memory index port.
    """

    kind: ClassVar[str] = "image_memory"

    def __init__(self, deps: RecallerDeps, index: ImageMemoryIndex) -> None:
        self._deps = deps
        self._index = index

    async def sparse_recall(
        self, query: str, where: Filter, *, limit: int
    ) -> RecallResult[ImageMemory]:
        """BM25 over description + in-picture text (OR-mode BooleanQuery).

        Cut like the episodes' (:func:`~.base.stable_sparse_top`): score ties
        are settled by id, so a repeated search recalls the same pictures in
        the same order -- the rule the exp tree's 2.0 picture port applies.
        """
        terms = [term for term in self._deps.tokenizer.tokenize(query) if term]
        if not terms:
            return RecallResult.empty()
        hits = await stable_sparse_top(
            lambda n: self._index.sparse_search(terms, where, limit=n), limit
        )
        return _tagged(from_hits(hits, source="keyword"))

    async def dense_recall(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> RecallResult[ImageMemory]:
        """ANN over the picture memory's vector; empty when none is stored."""
        if not vector:
            return RecallResult.empty()
        hits = await self._index.dense_search(vector, where, limit=limit)
        return _tagged(from_hits(hits, source="vector"))

    async def visual_recall(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> RecallResult[ImageMemory]:
        """ANN over the picture's own embedding; empty when none is stored.

        ``vector`` must come from the *vision* embedder's text encoder, not
        the text embedder: the two spaces are different models, and a cosine
        between them means nothing. An empty result is the normal case on a
        deployment with no ``[vision_embedding]``.
        """
        if not vector:
            return RecallResult.empty()
        hits = await self._index.visual_search(vector, where, limit=limit)
        return _tagged(from_hits(hits, source="vector"))

    async def find_by_image_ids(
        self, where: Filter, image_ids: Sequence[str]
    ) -> list[Stored[ImageMemory]]:
        """The current picture memories about these files, inside a read scope.

        What an episode's ``image_ids`` resolve through. ``where`` is the read
        scope's predicate of the episode's owner, which already excludes
        superseded and forgotten rows -- a picture must not be revived into an
        answer because an episode still refers to it -- and, for a group,
        everything outside the asker's windows: an episode the asker may see
        does not bring along a picture they may not. Several occurrences of
        one picture may come back (one per arrival).
        """
        if not image_ids:
            return []
        hits = await self._index.find(
            all_of(where, one_of("image_id", list(image_ids))),
            limit=max(len(image_ids) * 4, 1),
        )
        return [hit.stored for hit in hits]

    async def find_entries(
        self, where: Filter, entries: Mapping[str, Collection[str]]
    ) -> list[Stored[ImageMemory]]:
        """The picture memories ``entries`` names (owner id -> entry ids) in ``where``.

        What a search for the past reads the version that replaced a
        returned one from; ``where`` is its read scope.
        """
        return await find_entries(self._index, where, entries)


def _tagged(result: RecallResult[ImageMemory]) -> RecallResult[ImageMemory]:
    """Mark every candidate as a picture, for the mixed ranking.

    ``metadata`` is also what the decider's rendering reads, so the description
    is put where the shared renderer looks (``summary``) with no picture
    fallback: an episode's ``episode`` column does not exist on a picture, and
    silently borrowing it would make full-text mode show nothing.
    """
    return RecallResult(
        candidates=[
            c.model_copy(
                update={
                    "metadata": {
                        **c.metadata,
                        "kind": IMAGE_KIND,
                        "subject": c.metadata.get("description", "")[:60],
                        "summary": c.metadata.get("description", ""),
                    }
                }
            )
            for c in result.candidates
        ],
        stored_by_id=result.stored_by_id,
    )
