"""The picture files on disk: originals under ``figures/``, copies under ``.cache/``.

Binds :class:`~everos.memory.ports.ImageStore`: the filesystem side of a
picture and the SQLite row that counts its references, in one object, because
every caller that stores bytes also has to count the arrival. Splitting them
would let one succeed and the other not, and a reference count that disagrees
with the files is worse than no count.

Two invariants carry the design:

- **The original is written once per content hash and never rewritten.** Bytes
  go to a temporary file, are read back and hashed, and only a match is renamed
  into place. A crash leaves a temporary file, never a file whose name promises
  a hash it does not hold.
- **Derived copies are named by ``<hash>-<edge>``.** They are a pure function
  of the original, so changing a configured edge produces a new name instead of
  serving a stale copy at the wrong size, and dropping the whole directory is
  always safe.

Filesystem work is offloaded with :func:`anyio.to_thread.run_sync`: one decode
is tens of milliseconds of CPU on a path that is async end to end.
"""

from __future__ import annotations

__all__ = ["FileImageStore"]

import os
import uuid
from pathlib import Path

import anyio

from everos.component.images import (
    ImageLimits,
    StoredImage,
    StoredImageInfo,
    decode_image,
    derived_relpath,
    hash_bytes,
    scaled_copy,
)
from everos.core.errors import PathTraversalError
from everos.core.observability.logging import get_logger
from everos.infra.persistence.memory_root import MemoryRoot
from everos.infra.persistence.sqlite import ImageOriginalRepo

logger = get_logger(__name__)

_FIGURES_DIR = "figures"
_CACHE_DIR = ".cache"
_DERIVED_EXTENSIONS = ("jpeg", "png")
"""What :func:`~everos.component.images.scaled_copy` writes: JPEG, or PNG for a
greyscale picture."""


class FileImageStore:
    """Content-addressed picture files plus their reference counts.

    Args:
        root: The memory root; ``figures/`` hangs off it directly, beside the
            ``<app>/<project>/`` trees, because one picture is shared by every
            memory that references it whatever its scope.
        originals: The SQLite rows that record each file and count references.
    """

    def __init__(self, root: MemoryRoot, originals: ImageOriginalRepo) -> None:
        self._root = root
        self._originals = originals

    # ── Paths ──────────────────────────────────────────────────────────────

    @property
    def figures_dir(self) -> Path:
        """``<root>/figures/`` — originals, by content hash."""
        return self._root.root / _FIGURES_DIR

    @property
    def cache_dir(self) -> Path:
        """``<root>/figures/.cache/`` — derived copies, rebuildable."""
        return self.figures_dir / _CACHE_DIR

    async def relpath_of(self, image_id: str) -> str:
        """Memory-root-relative path of the original, as memories record it.

        The extension is part of the path but not of the identity, so this
        reads the metadata row rather than guessing a format — and falls back
        to listing the hash bucket, which is correct even for a store whose
        SQLite rows were rebuilt.

        Raises:
            FileNotFoundError: No original is on disk under this id.
        """
        row = await self._originals.get_by_id(image_id)
        if row is not None:
            return row.original_path
        found = await anyio.to_thread.run_sync(
            lambda: _existing_original(self.figures_dir, image_id)
        )
        if found is None:
            raise FileNotFoundError(f"picture {image_id!r} is not in the store")
        return found

    async def path_of(self, image_id: str) -> Path:
        """Absolute path of the original, guaranteed inside the memory root.

        These paths are served back over HTTP, so a malformed or hostile id
        must not be able to name a file elsewhere on the host.
        """
        return self._resolve(await self.relpath_of(image_id))

    def _resolve(self, relative: str) -> Path:
        candidate = (self._root.root / relative).resolve()
        if not candidate.is_relative_to(self._root.root):
            raise PathTraversalError(
                f"picture path escapes the memory root: {relative!r}"
            )
        return candidate

    # ── Writing ─────────────────────────────────────────────────────────────

    async def store(self, data: bytes, *, limits: ImageLimits) -> StoredImage:
        """Write one original if new, count one arrival, return its descriptor.

        Storing identical bytes twice costs one file and one more reference —
        that is the whole point of content addressing.
        """
        stored = await anyio.to_thread.run_sync(
            lambda: decode_image(data, limits=limits)
        )
        relative = (
            f"{_FIGURES_DIR}/{stored.image_id[:2]}/{stored.image_id}.{stored.extension}"
        )
        await self._write_atomic(
            self._resolve(relative), stored.data, expected_hash=stored.image_id
        )
        row = await self._originals.record_arrival(
            image_id=stored.image_id,
            media_type=stored.media_type,
            byte_size=len(stored.data),
            width=stored.width,
            height=stored.height,
            original_path=relative,
        )
        logger.info(
            "memory.images.stored",
            image_id=stored.image_id,
            media_type=stored.media_type,
            byte_size=len(stored.data),
            ref_count=row.ref_count,
        )
        return stored

    async def keep_unreferenced(
        self, data: bytes, *, limits: ImageLimits
    ) -> StoredImage:
        """Write the original and record its row at zero references.

        Deliberately not :meth:`store` with a flag: an arrival *is* a reference
        and forgetting one decrements it, so a question picture counted as an
        arrival would leave a reference no memory owns.
        """
        stored = await anyio.to_thread.run_sync(
            lambda: decode_image(data, limits=limits)
        )
        relative = (
            f"{_FIGURES_DIR}/{stored.image_id[:2]}/{stored.image_id}.{stored.extension}"
        )
        await self._write_atomic(
            self._resolve(relative), stored.data, expected_hash=stored.image_id
        )
        await self._originals.record_unreferenced(
            image_id=stored.image_id,
            media_type=stored.media_type,
            byte_size=len(stored.data),
            width=stored.width,
            height=stored.height,
            original_path=relative,
        )
        logger.info(
            "memory.images.kept_unreferenced",
            image_id=stored.image_id,
            media_type=stored.media_type,
        )
        return stored

    async def _write_atomic(
        self, target: Path, data: bytes, *, expected_hash: str | None
    ) -> None:
        """Land ``data`` at ``target`` through a temporary file and a rename.

        Skips the write when the target already holds this content. When
        ``expected_hash`` is given the bytes are read back and hashed first:
        that read-back is what makes the file name trustworthy.
        """

        def write() -> None:
            if target.exists():
                if expected_hash is None:
                    return
                if hash_bytes(target.read_bytes()) == expected_hash:
                    return
                # A file under this hash that does not hash to it: the store
                # would serve corrupt bytes for as long as it lives. Replace.
                logger.warning(
                    "memory.images.original_corrupt",
                    path=target.name,
                    image_id=expected_hash,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_dir / f"_write-{uuid.uuid4().hex}"
            try:
                tmp.write_bytes(data)
                if expected_hash is not None:
                    actual = hash_bytes(tmp.read_bytes())
                    if actual != expected_hash:
                        raise OSError(
                            f"picture hashed to {actual}, expected {expected_hash}"
                        )
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)

        await anyio.to_thread.run_sync(write)

    # ── Reading ─────────────────────────────────────────────────────────────

    async def read(self, image_id: str) -> bytes:
        """The original bytes, exactly as stored."""
        return await anyio.Path(await self.path_of(image_id)).read_bytes()

    async def read_derived(self, image_id: str, *, edge: int, quality: int) -> bytes:
        """Bytes of the ``edge``-sized copy, rendering it on first request."""
        rendered, _ = await self._render_derived(image_id, edge=edge, quality=quality)
        return rendered

    async def ensure_derived(self, image_id: str, *, edge: int, quality: int) -> str:
        """Root-relative path of the ``edge``-sized copy, rendered only if missing.

        A copy already on disk is named without decoding anything: it is a pure
        function of the original, so the extension it was written under is the
        one this picture always renders to.

        Raises:
            FileNotFoundError: No copy on disk and no original to render it from.
        """
        for extension in _DERIVED_EXTENSIONS:
            relative = derived_relpath(image_id, edge, extension)
            if await anyio.Path(self._resolve(relative)).is_file():
                return relative
        _, relative = await self._render_derived(image_id, edge=edge, quality=quality)
        return relative

    async def _render_derived(
        self, image_id: str, *, edge: int, quality: int
    ) -> tuple[bytes, str]:
        """Render the ``edge``-sized copy, write it if new; ``(bytes, relpath)``."""
        path = await self.path_of(image_id)
        data = await anyio.Path(path).read_bytes()
        rendered, extension = await anyio.to_thread.run_sync(
            lambda: scaled_copy(
                data,
                format_name=_format_of_path(path),
                edge=edge,
                quality=quality,
            )
        )
        # The renderer answers PNG for a greyscale picture; the name has to say
        # what the bytes are or a browser will refuse to display it.
        relative = derived_relpath(image_id, edge, extension)
        await self._write_atomic(self._resolve(relative), rendered, expected_hash=None)
        return rendered, relative

    async def exists(self, image_id: str) -> bool:
        """Whether the original is on disk."""
        try:
            return await anyio.Path(await self.path_of(image_id)).is_file()
        except FileNotFoundError:
            return False

    # ── Reference accounting ────────────────────────────────────────────────

    async def release(self, image_id: str) -> int:
        """Drop one reference to a picture; returns the count left.

        The file stays on disk at zero: forgetting is a status change on the
        memories, and undo has to be able to restore them without restoring
        the bytes.
        """
        remaining = await self._originals.release_reference(image_id)
        logger.info("memory.images.released", image_id=image_id, ref_count=remaining)
        return remaining

    async def forget(self, image_id: str) -> bool:
        """Take the picture out of circulation; the file stays on disk.

        The bytes are kept because undo must be able to bring the memories back
        without a restore, and because proving a deletion by the file's absence
        would make an undo indistinguishable from a re-upload.
        """
        forgotten = await self._originals.mark_forgotten(image_id)
        logger.info("memory.images.marked_forgotten", image_id=image_id)
        return forgotten

    async def restore(self, image_id: str) -> bool:
        """Undo :meth:`forget`."""
        restored = await self._originals.restore(image_id)
        logger.info("memory.images.restored", image_id=image_id)
        return restored

    async def metadata(self, image_id: str) -> StoredImageInfo | None:
        """Format, size, dimensions and reference count, or ``None``."""
        row = await self._originals.get_by_id(image_id)
        if row is None:
            return None
        return StoredImageInfo(
            image_id=row.image_id,
            media_type=row.media_type,
            byte_size=row.byte_size,
            width=row.width,
            height=row.height,
            original_path=row.original_path,
            ref_count=row.ref_count,
        )


def _existing_original(figures_dir: Path, image_id: str) -> str | None:
    """Root-relative path of ``image_id``'s file, discovered by its bucket.

    The extension is not part of the identity, so the two-level bucket is
    listed rather than guessed: a picture stored as PNG must not be reported
    under a ``.jpeg`` name that does not exist.
    """
    bucket = figures_dir / image_id[:2]
    if not bucket.is_dir():
        return None
    for candidate in bucket.glob(f"{image_id}.*"):
        return f"{_FIGURES_DIR}/{image_id[:2]}/{candidate.name}"
    return None


def _format_of_path(path: Path) -> str:
    """Pillow format name for an original, from its extension."""
    return {
        ".png": "PNG",
        ".jpeg": "JPEG",
        ".jpg": "JPEG",
        ".webp": "WEBP",
        ".gif": "GIF",
    }.get(path.suffix.lower(), "JPEG")
