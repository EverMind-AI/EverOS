"""Vision-embedding provider protocol.

The text provider's contract (:class:`~.protocol.EmbeddingProvider`) takes
strings only, which is what every table but one needs. A picture is the
exception: its *bytes* are content, and the design puts them in one vector
space with text so a text question recalls a picture directly.

This is deliberately a second protocol rather than another method on the first
one. A text-only deployment must be able to satisfy the text contract without
carrying an image code path it will never use, and the two providers are
different models with different endpoints and different widths.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class VisionEmbeddingProvider(Protocol):
    """Async provider that embeds pictures **and** text into one space.

    ``dim`` is the width of that space — the width of the ``visual_vector``
    column, set apart from ``[embedding] dim`` because the two columns come
    from different models.

    The text method exists because the *query* is text: to search a picture's
    bytes with a sentence, the sentence has to be embedded by the same model
    that embedded the pixels.
    """

    dim: int

    async def embed_image(self, image: bytes, media_type: str) -> list[float]:
        """Embed one picture. Returns a ``[dim]`` vector.

        Raises:
            EmbeddingServiceError: The endpoint failed (retryable).
            EmbeddingInputError: The endpoint refused the picture (permanent).
        """
        ...

    async def embed_images(
        self, images: Sequence[tuple[bytes, str]]
    ) -> list[list[float]]:
        """Embed many pictures, preserving input order."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed one interactive text query in the picture space.

        Draws on the query budget, so a search never queues behind the
        cascade's picture embeds (issue #473, same reasoning as the text
        provider's ``embed_query``).
        """
        ...
