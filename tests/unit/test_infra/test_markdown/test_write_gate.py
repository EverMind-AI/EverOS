"""Every markdown write into a scope being purged is refused before it lands."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from everos.core.errors import PurgeInProgressError
from everos.core.purge_marks import MARKS, PurgeScope
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import MarkdownKnowledgeStore
from everos.infra.persistence.markdown import knowledge_store as ks
from everos.infra.persistence.markdown._base.writer import MarkdownWriter
from everos.memory.models import KnowledgeMemory

_DOC = "d_abc123000000"
S = PurgeScope("project", "a1", "p1", "a1", "p1")


def _memories(title: str = "Guide") -> list[KnowledgeMemory]:
    return [
        KnowledgeMemory(
            doc_id=_DOC,
            topic_index=i,
            topic=title if i == 0 else f"Topic {i}",
            topic_path=title,
            summary="s",
            content=f"content {i}",
            depth=0 if i == 0 else 1,
            parent_index=None if i == 0 else 0,
            category_id="Tech",
        )
        for i in range(2)
    ]


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    MARKS.reset()
    yield
    MARKS.reset()


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    return MemoryRoot(tmp_path)


def _mark(root: MemoryRoot) -> None:
    MARKS.for_root(root.root).raise_mark("j1", S)


def _tree(path: Path) -> list[str]:
    return sorted(p.relative_to(path).as_posix() for p in path.rglob("*"))


async def test_writer_rejects_target_scope_while_marked(root: MemoryRoot) -> None:
    _mark(root)
    writer = MarkdownWriter(root)
    target = root.root / "a1" / "p1" / "users" / "u1" / "user.md"
    with pytest.raises(PurgeInProgressError, match="j1"):
        await writer.write(target, "x")
    assert not target.parent.exists()
    other = root.root / "a1" / "p2" / "users" / "u1" / "user.md"
    await writer.write(other, "x")
    assert other.read_text() == "x"


async def _written_doc(root: MemoryRoot) -> tuple[MarkdownKnowledgeStore, Path]:
    store = MarkdownKnowledgeStore(root)
    written = await store.write_document(
        "a1", "p1", _memories(), source_name="g.pdf", source_type="file", original=b"1"
    )
    return store, Path(written.location)


async def test_knowledge_writer_keeps_the_old_document(root: MemoryRoot) -> None:
    store, doc_dir = await _written_doc(root)
    before = _tree(doc_dir)
    _mark(root)
    with pytest.raises(PurgeInProgressError):
        await store.write_document(
            "a1",
            "p1",
            _memories("New"),
            source_name=None,
            source_type=None,
            original=None,
        )
    assert _tree(doc_dir) == before


async def test_replace_is_refused_before_the_backup(root: MemoryRoot) -> None:
    store, doc_dir = await _written_doc(root)
    document = await store.read_document("a1", "p1", _DOC)
    assert document is not None
    before = _tree(doc_dir.parent)
    _mark(root)
    with pytest.raises(PurgeInProgressError):
        await store.replace_document(
            document,
            _memories("New"),
            source_name=None,
            source_type=None,
            original=None,
        )
    assert _tree(doc_dir.parent) == before  # no hidden backup either


async def test_metadata_patch_is_refused(root: MemoryRoot) -> None:
    store, doc_dir = await _written_doc(root)
    document = await store.read_document("a1", "p1", _DOC)
    assert document is not None
    index = (doc_dir / "index.md").read_bytes()
    _mark(root)
    with pytest.raises(PurgeInProgressError):  # _rewrite_frontmatter
        await store.update_document(
            document.model_copy(update={"title": "Renamed"}),
            previous_category_id="Tech",
        )
    assert (doc_dir / "index.md").read_bytes() == index


async def test_category_move_is_refused(root: MemoryRoot) -> None:
    _, doc_dir = await _written_doc(root)
    _mark(root)
    with pytest.raises(PurgeInProgressError):
        await ks._move_to_category(doc_dir.parent.parent, doc_dir, "Science")
    assert doc_dir.is_dir()


async def test_original_file_write_is_refused(root: MemoryRoot) -> None:
    _, doc_dir = await _written_doc(root)
    _mark(root)
    with pytest.raises(PurgeInProgressError):
        await ks._write_original(doc_dir, "other.pdf", b"2")
    assert not (doc_dir / "_original" / "other.pdf").exists()


async def test_backup_restore_is_refused(root: MemoryRoot) -> None:
    _, doc_dir = await _written_doc(root)
    backup = await ks._back_up([doc_dir])
    _mark(root)
    with pytest.raises(PurgeInProgressError):
        await ks._restore(backup, _DOC)
    assert not doc_dir.exists()  # the purge removes the hidden backup with S


async def test_ensure_taxonomy_exempt(root: MemoryRoot) -> None:
    _mark(root)
    categories = await MarkdownKnowledgeStore(root).categories("a1", "p1")
    assert "Others" in {c.id for c in categories}
