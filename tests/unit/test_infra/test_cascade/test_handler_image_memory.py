"""Tests for :class:`ImageMemoryHandler` — image-memory md → LanceDB row.

What these pin:

- A picture entry becomes one index row whose identity is the entry id, and
  whose ``image_id`` is the *file's* content hash — two identities that must
  not be confused: forgetting takes the file id, a result carries the entry id.
- Both keyword channels are produced: the description and the visible text.
  Pictures have no keyword surface of their own, so those two fields are the
  only lexical route to a picture.
- The vector is the description's. When no embedder is configured the row is
  still written, with no vector, and stays keyword-searchable.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.infra.cascade._projections.image_memory import ImageMemoryHandler
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.index import Comparison, Predicate
from everos.infra.persistence.markdown import ImageMemoryWriter
from everos.infra.persistence.records import ImageMemory
from tests.fakes import handler_deps

_IMAGE_ID = "c" * 64


class _StubTokenizer(Tokenizer):
    """Whitespace split — deterministic for assertions."""

    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _StubEmbedder(EmbeddingProvider):
    dim = 8

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append([text])
        return [0.1] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        self.calls.append(list(texts))
        return [[0.1] * self.dim for _ in texts]


class _FakeImageRepo:
    """Recording repo for the ``image_memory`` table."""

    def __init__(self) -> None:
        self.upserts: list[list[ImageMemory]] = []
        self.rows: list[ImageMemory] = []

    async def find_where(
        self, where: Predicate, *, limit: int = 100, skip_vectors: bool = False
    ) -> list[ImageMemory]:
        if isinstance(where, Comparison) and where.field == "md_path":
            return [r for r in self.rows if r.md_path == str(where.value)]
        return []

    async def upsert(self, rows: list[ImageMemory]) -> None:
        self.upserts.append(list(rows))
        by_id = {r.id: r for r in self.rows}
        for r in rows:
            by_id[r.id] = r
        self.rows = list(by_id.values())

    async def delete(self, predicate: Predicate) -> None:
        return None

    async def delete_by_md_path(self, md_path: str) -> int:
        before = len(self.rows)
        self.rows = [r for r in self.rows if r.md_path != md_path]
        return before - len(self.rows)

    async def update(self, updates: dict, *, where: Predicate) -> None:
        field = getattr(where, "field", None)
        if field != "owner_id":
            return
        for row in self.rows:
            for name, value in updates.items():
                setattr(row, name, value)


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


@pytest.fixture
def fake_repo(monkeypatch: pytest.MonkeyPatch) -> _FakeImageRepo:
    repo = _FakeImageRepo()
    monkeypatch.setattr(ImageMemoryHandler, "index_repo", repo)
    return repo


async def _write_entry(
    writer: ImageMemoryWriter,
    *,
    description: str,
    ocr: str | None = None,
) -> str:
    sections = {"Description": description}
    if ocr is not None:
        sections["OCR"] = ocr
    await writer.append_entry(
        "u1",
        inline={
            "owner_id": "u1",
            "session_id": "s1",
            "timestamp": "2026-05-14T10:00:00+00:00",
            "parent_type": "memcell",
            "parent_id": "mc_1",
            "image_id": _IMAGE_ID,
            "original_path": f"figures/cc/{_IMAGE_ID}.png",
            "media_type": "png",
            "width": 640,
            "height": 480,
        },
        sections=sections,
        date=_dt.date(2026, 5, 14),
    )
    return "default_app/default_project/users/u1/images/image-2026-05-14.md"


def _handler(
    memory_root: MemoryRoot, embedding: EmbeddingProvider | None
) -> ImageMemoryHandler:
    return ImageMemoryHandler(
        handler_deps(
            memory_root=memory_root,
            tokenizer=_StubTokenizer(),
            embedding=embedding,
        )
    )


async def test_entry_becomes_one_row_with_both_identities(
    memory_root: MemoryRoot, fake_repo: _FakeImageRepo
) -> None:
    rel = await _write_entry(
        ImageMemoryWriter(memory_root),
        description="a latte with heart-shaped art",
        ocr="DOU SHE EST 2019",
    )

    outcome = await _handler(memory_root, _StubEmbedder()).handle_added_or_modified(rel)

    assert outcome.upserted == 1
    row = fake_repo.upserts[0][0]
    assert row.entry_id.startswith("im_")
    assert row.id == f"u1_{row.entry_id}"
    assert row.image_id == _IMAGE_ID
    assert row.original_path == f"figures/cc/{_IMAGE_ID}.png"
    assert row.media_type == "png"
    assert row.width == 640 and row.height == 480
    assert row.description == "a latte with heart-shaped art"
    assert row.ocr == "DOU SHE EST 2019"
    assert row.parent_id == "mc_1"
    assert row.owner_type == "user"


async def test_keyword_fields_cover_description_and_visible_text(
    memory_root: MemoryRoot, fake_repo: _FakeImageRepo
) -> None:
    rel = await _write_entry(
        ImageMemoryWriter(memory_root),
        description="shop sign outside",
        ocr="EST 2019",
    )

    await _handler(memory_root, _StubEmbedder()).handle_added_or_modified(rel)

    row = fake_repo.upserts[0][0]
    assert row.description_tokens == "shop sign outside"
    assert row.ocr_tokens == "EST 2019"


async def test_a_picture_without_text_has_no_ocr_tokens(
    memory_root: MemoryRoot, fake_repo: _FakeImageRepo
) -> None:
    rel = await _write_entry(ImageMemoryWriter(memory_root), description="a sunset")

    await _handler(memory_root, _StubEmbedder()).handle_added_or_modified(rel)

    row = fake_repo.upserts[0][0]
    assert row.ocr == ""
    assert row.ocr_tokens == ""


async def test_the_description_is_what_gets_embedded(
    memory_root: MemoryRoot, fake_repo: _FakeImageRepo
) -> None:
    embedder = _StubEmbedder()
    rel = await _write_entry(
        ImageMemoryWriter(memory_root), description="a latte", ocr="EST 2019"
    )

    await _handler(memory_root, embedder).handle_added_or_modified(rel)

    # One text, not two: the vector stands for the picture's description. The
    # visible text rides the keyword channel only.
    assert embedder.calls == [["a latte"]]


async def test_without_an_embedder_the_row_still_lands(
    memory_root: MemoryRoot, fake_repo: _FakeImageRepo
) -> None:
    rel = await _write_entry(ImageMemoryWriter(memory_root), description="a latte")

    outcome = await _handler(memory_root, None).handle_added_or_modified(rel)

    assert outcome.upserted == 1
    assert fake_repo.upserts[0][0].vector is None


async def test_an_unchanged_file_is_skipped_not_re_embedded(
    memory_root: MemoryRoot, fake_repo: _FakeImageRepo
) -> None:
    rel = await _write_entry(ImageMemoryWriter(memory_root), description="a latte")
    handler = _handler(memory_root, _StubEmbedder())
    await handler.handle_added_or_modified(rel)

    second = await handler.handle_added_or_modified(rel)

    assert second.upserted == 0
    assert second.skipped == 1
