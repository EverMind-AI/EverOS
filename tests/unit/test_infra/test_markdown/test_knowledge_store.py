"""Markdown adapter of the knowledge truth port, against a real memory root."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.core.errors import DocumentNotFoundError
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import MarkdownKnowledgeStore
from everos.memory.models import KnowledgeMemory
from everos.memory.ports import KnowledgeStore

_DOC = "d_abc123000000"


def _memories(title: str = "Guide", category: str = "Tech") -> list[KnowledgeMemory]:
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
            category_id=category,
        )
        for i in range(3)
    ]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    return MemoryRoot(tmp_path)


def test_adapter_satisfies_the_port(root: MemoryRoot) -> None:
    assert isinstance(MarkdownKnowledgeStore(root), KnowledgeStore)


async def test_categories_seed_the_default_taxonomy(root: MemoryRoot) -> None:
    categories = await MarkdownKnowledgeStore(root).categories("app", "proj")
    assert "Others" in {c.id for c in categories}


async def test_write_read_update_delete_round_trip(root: MemoryRoot) -> None:
    store = MarkdownKnowledgeStore(root)

    written = await store.write_document(
        "app",
        "proj",
        _memories(),
        source_name="guide.pdf",
        source_type="file",
        original=b"pdf",
    )
    doc_dir = Path(written.location)
    assert doc_dir.name.endswith(f"_{_DOC}")
    assert written.original_file_path == str(doc_dir / "_original" / "guide.pdf")

    document = await store.read_document("app", "proj", _DOC)
    assert document is not None
    assert (document.title, document.category_id) == ("Guide", "Tech")
    assert document.md_path == str((doc_dir / "index.md").relative_to(root.root))
    assert await store.original_file(document) == written.original_file_path

    moved = await store.update_document(
        document.model_copy(update={"title": "Renamed", "category_id": "Science"}),
        previous_category_id="Tech",
    )
    new_index = root.root / (moved.md_path or "")
    assert new_index.parent.parent.name == "Science"
    assert "title: Renamed" in _read(new_index)
    assert all(
        "category_id: Science" in _read(f) for f in new_index.parent.glob("[0-9]*.md")
    )

    removed = await store.delete_document("app", "proj", _DOC, md_path=None)
    assert removed == 2
    assert await store.read_document("app", "proj", _DOC) is None


async def test_update_of_a_missing_document_raises(root: MemoryRoot) -> None:
    store = MarkdownKnowledgeStore(root)
    await store.write_document(
        "app", "proj", _memories(), source_name=None, source_type=None, original=None
    )
    document = await store.read_document("app", "proj", _DOC)
    assert document is not None
    await store.delete_document("app", "proj", _DOC, md_path=document.md_path)

    with pytest.raises(DocumentNotFoundError):
        await store.update_document(document, previous_category_id="Tech")


async def test_replace_restores_the_previous_files_on_failure(
    root: MemoryRoot,
) -> None:
    store = MarkdownKnowledgeStore(root)
    await store.write_document(
        "app", "proj", _memories(), source_name=None, source_type=None, original=None
    )
    previous = await store.read_document("app", "proj", _DOC)
    assert previous is not None

    with pytest.raises(ValueError, match="root node"):
        await store.replace_document(
            previous,
            _memories()[1:],  # no root node: the writer refuses
            source_name=None,
            source_type=None,
            original=None,
        )

    restored = await store.read_document("app", "proj", _DOC)
    assert restored is not None and restored.md_path == previous.md_path
    assert not list(root.knowledge_dir("app", "proj").rglob("*.backup"))
