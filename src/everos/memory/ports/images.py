"""Picture store — originals on disk plus their reference-counted metadata.

Declared here so the memory / service layers can store pictures without knowing
they are files: the port names what an arrival means (store once, count a
reference), not where the bytes live.

``store`` is idempotent by content: storing the same picture twice returns the
same :class:`StoredImage` and only adds a reference, which is what makes a
forwarded picture cost nothing.

``read``/``thumbnail`` return bytes rather than paths because their only callers
hand those bytes to a model as a data URI; a caller that wants a path for the
API response uses ``path_of``, which also enforces that the path stays inside
the memory root.
"""

from __future__ import annotations

__all__ = ["ImageStore"]

from pathlib import Path
from typing import Protocol, runtime_checkable

from everos.component.images import ImageLimits, StoredImage, StoredImageInfo


@runtime_checkable
class ImageStore(Protocol):
    """The picture files of one memory root, by content hash."""

    async def store(self, data: bytes, *, limits: ImageLimits) -> StoredImage:
        """Store one original if new and count one arrival.

        Raises:
            ImageRejectedError: Bytes no accepted format can be read from.
        """
        ...

    async def keep_unreferenced(
        self, data: bytes, *, limits: ImageLimits
    ) -> StoredImage:
        """Store one picture with **no** memory and no reference count.

        The question-picture path: a picture attached to a search query is
        content-addressed like any original — so a second query with the same
        picture costs nothing and the decider reads it from the same place as
        a candidate — but it establishes no memory, because "what did I ask
        with" is not a fact about the user. Its row starts at ``ref_count`` 0;
        if the picture later arrives in a real message, the ordinary store
        raises the count on the same file.

        Raises:
            ImageRejectedError: Bytes no accepted format can be read from.
        """
        ...

    async def read(self, image_id: str) -> bytes:
        """The original bytes of ``image_id``.

        Raises:
            FileNotFoundError: The store holds no such picture.
        """
        ...

    async def read_derived(self, image_id: str, *, edge: int, quality: int) -> bytes:
        """Bytes of the ``edge``-sized copy, rendered on first request."""
        ...

    async def path_of(self, image_id: str) -> Path:
        """Absolute path of the original, for delete / download routes."""
        ...

    async def relpath_of(self, image_id: str) -> str:
        """Memory-root-relative path of the original, as recorded in memories."""
        ...

    async def metadata(self, image_id: str) -> StoredImageInfo | None:
        """Format, size, dimensions and reference count, or ``None``."""
        ...

    async def release(self, image_id: str) -> int:
        """Drop one reference; returns the count left (0 = nothing points here)."""
        ...

    async def forget(self, image_id: str) -> bool:
        """Take the picture out of circulation; the bytes stay on disk.

        Whether the picture was known. Undo is :meth:`restore`: because the file
        is never unlinked, restoring is a status change, not a copy back.
        """
        ...

    async def restore(self, image_id: str) -> bool:
        """Undo :meth:`forget`; the bytes were never removed."""
        ...

    async def exists(self, image_id: str) -> bool:
        """Whether an original is on disk. Used to prove a forget took effect."""
        ...
