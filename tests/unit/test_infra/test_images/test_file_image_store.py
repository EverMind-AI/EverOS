"""The picture files: content addressing, derived copies, reference counting.

What these pin — the invariants the whole picture path rests on:

- The original is stored **once per content hash**, byte-for-byte, and the
  same bytes arriving again add a reference and no file. That is what keeps a
  store whose originals are never re-encoded from growing with conversation
  volume.
- The file's name is its hash and nothing else: a file whose bytes do not hash
  to its name is replaced, because the store would otherwise serve corrupt
  bytes for as long as it lives.
- Derived copies are named by ``<hash>-<edge>`` and rebuilt on demand, so they
  can be deleted at any time and never stand in for the original.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from everos.component.images import ImageLimits, ImageRejectedError, hash_bytes
from everos.config import SqliteSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.images import FileImageStore
from everos.infra.persistence.sqlite import SqliteDatabase, SqliteRepos


def _limits(**overrides: object) -> ImageLimits:
    values: dict[str, object] = {
        "max_bytes": 5 * 1024 * 1024,
        "max_per_message": 4,
        "max_request_bytes": 10 * 1024 * 1024,
        "max_per_request": 8,
        "formats": ("png", "jpeg", "webp", "gif"),
    }
    values.update(overrides)
    return ImageLimits(**values)  # type: ignore[arg-type]


def _png(colour: tuple[int, int, int] = (10, 120, 200), size: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(buffer, format="PNG")
    return buffer.getvalue()


async def _store(tmp_path: Path) -> FileImageStore:
    root = MemoryRoot(tmp_path)
    root.ensure()
    database = SqliteDatabase(root, SqliteSettings())
    await database.create_schema()
    return FileImageStore(root, SqliteRepos.over(database).image_original)


async def test_original_is_stored_once_under_its_hash(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    data = _png()

    first = await store.store(data, limits=_limits())
    second = await store.store(data, limits=_limits())

    assert first.image_id == second.image_id == hash_bytes(data)
    path = await store.path_of(first.image_id)
    assert path.relative_to(tmp_path).as_posix() == (
        f"figures/{first.image_id[:2]}/{first.image_id}.png"
    )
    assert path.read_bytes() == data
    info = await store.metadata(first.image_id)
    assert info is not None
    assert info.ref_count == 2
    assert info.width == 64 and info.height == 64
    assert info.byte_size == len(data)


async def test_distinct_pictures_get_distinct_files(tmp_path: Path) -> None:
    store = await _store(tmp_path)

    a = await store.store(_png((1, 2, 3)), limits=_limits())
    b = await store.store(_png((4, 5, 6)), limits=_limits())

    assert a.image_id != b.image_id
    assert len(list((tmp_path / "figures").glob("*/*"))) == 2


async def test_a_stray_file_under_a_hash_is_replaced(tmp_path: Path) -> None:
    """A corrupt original must not survive: the name is the integrity claim."""
    store = await _store(tmp_path)
    data = _png()
    image = await store.store(data, limits=_limits())
    path = await store.path_of(image.image_id)
    path.write_bytes(b"corrupted")

    await store.store(data, limits=_limits())

    assert path.read_bytes() == data


async def test_derived_copy_is_smaller_and_named_by_edge(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    data = _png(size=1024)
    image = await store.store(data, limits=_limits())

    thumbnail = await store.read_derived(image.image_id, edge=64, quality=85)

    with Image.open(io.BytesIO(thumbnail)) as rendered:
        assert max(rendered.size) <= 64
    assert (tmp_path / "figures/.cache" / f"{image.image_id}-64.jpeg").is_file()
    # The original is untouched by rendering it.
    assert (await store.read(image.image_id)) == data


async def test_release_counts_down_and_keeps_the_file(tmp_path: Path) -> None:
    """Forget is a status change; undo must not need the bytes restored."""
    store = await _store(tmp_path)
    image = await store.store(_png(), limits=_limits())

    remaining = await store.release(image.image_id)

    assert remaining == 0
    assert await store.exists(image.image_id)


async def test_forget_marks_and_restore_undoes(tmp_path: Path) -> None:
    store = await _store(tmp_path)
    image = await store.store(_png(), limits=_limits())

    assert await store.forget(image.image_id)
    assert await store.restore(image.image_id)
    assert await store.exists(image.image_id)


async def test_an_unknown_id_is_not_found(tmp_path: Path) -> None:
    store = await _store(tmp_path)

    with pytest.raises(FileNotFoundError):
        await store.read("b" * 64)
    assert not await store.exists("b" * 64)


async def test_a_non_image_is_refused(tmp_path: Path) -> None:
    store = await _store(tmp_path)

    with pytest.raises(ImageRejectedError):
        await store.store(b"not an image", limits=_limits())
    assert not list((tmp_path / "figures").glob("*/*"))


async def test_oversized_picture_is_refused(tmp_path: Path) -> None:
    store = await _store(tmp_path)

    with pytest.raises(ImageRejectedError):
        await store.store(_png(), limits=_limits(max_bytes=10))
