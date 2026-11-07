"""Image memories in markdown: the file, the entry, and the supersession mark.

What these pin:

- A picture memory lands in ``users/<owner>/images/image-<date>.md`` as an
  append-only entry whose inline block records the picture's content hash, so
  the cascade can rebuild the index row without decoding the file.
- The same picture stored twice is **one file and two references**, and two
  memories when it arrives twice — the file is the picture, the memory is the
  appearance.
- A description edit or a forget writes a new entry and marks the old one
  superseded; nothing is ever rewritten in place.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    MarkdownImageMemoryStore,
    MarkdownReader,
    parse_structured_entry,
)
from everos.memory.models import ImageMemory
from everos.memory.ports import NO_SOURCE_CHECK, Scope

_TS = 1_747_216_800_000  # 2025-05-14T10:00:00Z
_IMAGE_ID = "a" * 64


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    memory_root = MemoryRoot(tmp_path)
    memory_root.ensure()
    return memory_root


def _user() -> Scope:
    return Scope(app_id="app", project_id="proj", owner_id="u1", owner_type="user")


def _image(**overrides: object) -> ImageMemory:
    fields: dict[str, object] = {
        "owner_id": "u1",
        "description": "a latte with heart-shaped latte art",
        "timestamp": _TS,
        "ocr": "",
        "image_id": _IMAGE_ID,
        "original_relpath": f"figures/aa/{_IMAGE_ID}.png",
        "media_type": "png",
        "width": 4,
        "height": 4,
        "session_id": "s1",
        "parent_id": "mc_1",
    }
    return ImageMemory(**(fields | overrides))  # type: ignore[arg-type]


async def test_entry_lands_under_the_owner_images_dir(root: MemoryRoot) -> None:
    """The file bucket is the day the memory was *written*, like every daily log.

    The picture's own time is a different thing and lives in the entry's
    ``timestamp`` inline field: one file per write-day keeps appends cheap, and
    a memory can be about a picture sent weeks ago.
    """
    store = MarkdownImageMemoryStore(root)

    ref = await store.append(_user(), _image(), only_if=NO_SOURCE_CHECK)

    assert ref is not None
    assert ref.md_path.startswith("app/proj/users/u1/images/image-")
    assert ref.md_path.endswith(".md")
    assert ref.entry_id.startswith("im_")


async def test_inline_block_carries_what_the_cascade_reads(root: MemoryRoot) -> None:
    """The picture's identity and path are inline; description is a section."""
    store = MarkdownImageMemoryStore(root)
    ref = await store.append(
        _user(), _image(ocr="DOU SHE EST 2019"), only_if=NO_SOURCE_CHECK
    )
    assert ref is not None

    parsed = await MarkdownReader.read(root.root / ref.md_path)
    entry = parsed.entries[0]
    structured = parse_structured_entry(entry.body)

    assert structured.inline["image_id"] == _IMAGE_ID
    assert structured.inline["original_path"] == f"figures/aa/{_IMAGE_ID}.png"
    assert structured.inline["media_type"] == "png"
    assert structured.inline["width"] == "4"
    assert structured.inline["parent_id"] == "mc_1"
    assert structured.inline["session_id"] == "s1"
    assert structured.sections["Description"].startswith("a latte")
    assert structured.sections["OCR"] == "DOU SHE EST 2019"


async def test_no_text_in_the_picture_means_no_ocr_section(root: MemoryRoot) -> None:
    store = MarkdownImageMemoryStore(root)
    ref = await store.append(_user(), _image(), only_if=NO_SOURCE_CHECK)
    assert ref is not None

    parsed = await MarkdownReader.read(root.root / ref.md_path)
    structured = parse_structured_entry(parsed.entries[0].body)

    assert "OCR" not in structured.sections


async def test_two_occurrences_are_two_entries_pointing_at_one_file(
    root: MemoryRoot,
) -> None:
    """The memory is the appearance; the file is the picture."""
    store = MarkdownImageMemoryStore(root)

    refs = await store.append_many(
        _user(),
        [_image(), _image(session_id="s2", parent_id="mc_2")],
        only_if=NO_SOURCE_CHECK,
    )

    assert len(refs) == 2
    assert refs[0].entry_id != refs[1].entry_id
    parsed = await MarkdownReader.read(root.root / refs[0].md_path)
    inline = [parse_structured_entry(e.body).inline for e in parsed.entries]
    assert {block["image_id"] for block in inline} == {_IMAGE_ID}
    assert {block["session_id"] for block in inline} == {"s1", "s2"}


async def test_deprecate_marks_the_old_entry_and_keeps_it(root: MemoryRoot) -> None:
    """Supersession is a frontmatter mark; the entry itself is never rewritten."""
    store = MarkdownImageMemoryStore(root)
    old = await store.append(_user(), _image(), only_if=NO_SOURCE_CHECK)
    new = await store.append(
        _user(),
        _image(description="the same latte, described again"),
        only_if=NO_SOURCE_CHECK,
    )
    assert old is not None and new is not None

    marked = await store.deprecate(
        _user(),
        entries=[(old.md_path, old.entry_id)],
        replaced_by=new.entry_id,
        only_if=NO_SOURCE_CHECK,
    )

    assert marked
    parsed = await MarkdownReader.read(root.root / old.md_path)
    assert parsed.frontmatter["deprecated_entries"] == {old.entry_id: new.entry_id}
    assert len(parsed.entries) == 2


async def test_remove_session_drops_only_that_session(root: MemoryRoot) -> None:
    store = MarkdownImageMemoryStore(root)
    await store.append_many(
        _user(),
        [_image(), _image(session_id="s2")],
        only_if=NO_SOURCE_CHECK,
    )

    removed = await store.remove_session(_user(), "s1")

    assert [ref.entry_id for ref in removed]
    parsed = await MarkdownReader.read(root.root / removed[0].md_path)
    remaining = [
        parse_structured_entry(e.body).inline["session_id"] for e in parsed.entries
    ]
    assert remaining == ["s2"]


async def test_a_deleted_source_skips_the_write(root: MemoryRoot) -> None:
    store = MarkdownImageMemoryStore(root)

    async def gone() -> bool:
        return False

    ref = await store.append(_user(), _image(), only_if=gone)

    assert ref is None
    assert not (root.root / "app/proj/users/u1/images").exists()
