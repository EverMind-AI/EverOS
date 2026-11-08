"""Image-memory recaller — BM25 over description+text, ANN over the picture.

The two routes are the same two episodes get, over the same vector space, which
is the design's point: a text question can recall a picture directly, and a
picture memory competes for a slot on the same terms as a narrative.

The keyword route is not a fallback here, it is the load-bearing one for
screenshots and signs: the words *in* the picture ("EST. 2019", "Transfer",
"Total 28.00") exist in no other memory, and BM25 over the description and the
transcribed text is the only way a lexical question reaches them.

The vector route embeds the *description*, not the picture bytes. That is a
deliberate inversion of the design's target state, forced by the configured
provider: ``[embedding]`` is a text endpoint (``Qwen3-Embedding-4B`` by
default), so handing it pixels is not possible today. What the inversion costs
is precision on questions whose words appear nowhere in the description; what
it does not cost is the keyword route above. When the embedder becomes
multimodal the producer switches to picture bytes and this class does not
change at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from everos.memory.models import ImageMemory
from everos.memory.ports import (
    Filter,
    ImageMemoryIndex,
    Scope,
    Stored,
    all_of,
    one_of,
)

from .base import RecallerDeps, RecallResult, from_hits

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
    everalgo_memory_type: ClassVar[str] = "image"

    def __init__(self, deps: RecallerDeps, index: ImageMemoryIndex) -> None:
        self._deps = deps
        self._index = index

    async def sparse_recall(
        self, query: str, where: Filter, *, limit: int
    ) -> RecallResult[ImageMemory]:
        """BM25 over description + in-picture text (OR-mode BooleanQuery)."""
        terms = [term for term in self._deps.tokenizer.tokenize(query) if term]
        if not terms:
            return RecallResult.empty()
        hits = await self._index.sparse_search(terms, where, limit=limit)
        return _tagged(from_hits(hits, source="keyword"))

    async def dense_recall(
        self, vector: Sequence[float], where: Filter, *, limit: int
    ) -> RecallResult[ImageMemory]:
        """ANN over the picture memory's vector; empty when none is stored."""
        if not vector:
            return RecallResult.empty()
        hits = await self._index.dense_search(vector, where, limit=limit)
        return _tagged(from_hits(hits, source="vector"))

    async def fetch_by_entry_ids(
        self, entry_ids: Sequence[str], where: Filter
    ) -> RecallResult[ImageMemory]:
        """Pictures by ``entry_id``, keyed by global id, unscored."""
        if not entry_ids:
            return RecallResult.empty()
        full_where = all_of(where, one_of("entry_id", list(entry_ids)))
        hits = await self._index.find(full_where, limit=len(entry_ids))
        return _tagged(from_hits(hits, source="vector"))

    async def find_by_image_ids(
        self, scope: Scope, image_ids: Sequence[str]
    ) -> list[Stored[ImageMemory]]:
        """The current picture memories about these files.

        What an episode's ``image_ids`` resolve through.

        Only current occurrences: a forgotten or superseded picture must not be
        revived into an answer because an episode still refers to it.
        """
        if not image_ids:
            return []
        return await self._index.find_by_image_ids(scope, list(image_ids))

    async def fetch_for_parent(
        self, *, parent_type: str, parent_id: str
    ) -> list[Stored[ImageMemory]]:
        """Pictures that arrived with one source segment — the forget edge."""
        return await self._index.find_by_parent(
            parent_type=parent_type, parent_id=parent_id
        )


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
