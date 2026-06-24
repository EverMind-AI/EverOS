"""Tests for :class:`KnowledgeWriter` — knowledge document directory layout."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from everos.infra.persistence.markdown.writers import KnowledgeWriter
from everos.infra.persistence.markdown.writers.knowledge_writer import (
    KnowledgeMemory,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


def _root_node(
    doc_id: str = "d_abc123",
    topic: str = "Olympics Plan",
    summary: str = "Overview of the Olympic plan.",
    category_id: str = "Sports",
    **overrides: object,
) -> KnowledgeMemory:
    defaults: dict[str, object] = {
        "doc_id": doc_id,
        "topic_index": 0,
        "topic": topic,
        "summary": summary,
        "content": "",
        "depth": 0,
        "parent_index": None,
        "children_index": [1, 2],
        "topic_path": topic,
        "content_labels": [],
        "category_id": category_id,
    }
    defaults.update(overrides)
    return KnowledgeMemory(**defaults)  # type: ignore[arg-type]


def _topic_node(
    doc_id: str = "d_abc123",
    topic_index: int = 1,
    topic: str = "Opening Ceremony",
    summary: str = "Details on the opening ceremony.",
    content: str = "The opening ceremony will feature...",
    depth: int = 1,
    parent_index: int | None = 0,
    children_index: list[int] | None = None,
    topic_path: str = "Olympics Plan > Opening Ceremony",
    content_labels: list[str] | None = None,
    category_id: str = "Sports",
) -> KnowledgeMemory:
    return KnowledgeMemory(
        doc_id=doc_id,
        topic_index=topic_index,
        topic=topic,
        summary=summary,
        content=content,
        depth=depth,
        parent_index=parent_index,
        children_index=children_index or [],
        topic_path=topic_path,
        content_labels=content_labels or [],
        category_id=category_id,
    )


def _parse_md(path: Path) -> tuple[dict[str, object], str]:
    """Parse a markdown file into (frontmatter_dict, body)."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    assert len(parts) >= 3, f"Expected YAML frontmatter in {path}"
    fm = yaml.safe_load(parts[1]) or {}
    body = parts[2]
    if body.startswith("\n"):
        body = body[1:]
    return fm, body


# ── Tests ─────────────────────────────────────────────────────────────────


async def test_basic_write_creates_correct_structure(tmp_path: Path) -> None:
    """Root + 2 topics -> index.md + 2 topic files."""
    memories = [
        _root_node(),
        _topic_node(topic_index=1, topic="Opening Ceremony"),
        _topic_node(topic_index=2, topic="Closing Ceremony"),
    ]
    doc_dir = await KnowledgeWriter.write(memories, tmp_path)

    assert doc_dir == tmp_path / "Sports" / "Olympics_Plan_d_abc123"
    assert (doc_dir / "index.md").is_file()
    assert (doc_dir / "1_Opening_Ceremony.md").is_file()
    assert (doc_dir / "2_Closing_Ceremony.md").is_file()


async def test_index_frontmatter_has_knowledge_document_type(
    tmp_path: Path,
) -> None:
    memories = [_root_node()]
    doc_dir = await KnowledgeWriter.write(memories, tmp_path)

    fm, body = _parse_md(doc_dir / "index.md")
    assert fm["type"] == "knowledge_document"
    assert fm["id"] == "d_abc123"
    assert fm["doc_id"] == "d_abc123"
    assert fm["category_id"] == "Sports"
    assert fm["title"] == "Olympics Plan"
    assert fm["schema_version"] == 1
    assert body.rstrip("\n") == "Overview of the Olympic plan."


async def test_topic_frontmatter_has_knowledge_topic_type(
    tmp_path: Path,
) -> None:
    memories = [
        _root_node(),
        _topic_node(topic_index=1, topic="Opening Ceremony"),
    ]
    doc_dir = await KnowledgeWriter.write(memories, tmp_path)

    fm, body = _parse_md(doc_dir / "1_Opening_Ceremony.md")
    assert fm["type"] == "knowledge_topic"
    assert fm["topic_name"] == "Opening Ceremony"
    assert fm["topic_index"] == 1
    assert body.rstrip("\n") == "The opening ceremony will feature..."


async def test_node_id_format(tmp_path: Path) -> None:
    """node_id and id follow ``{doc_id}_{topic_index}`` pattern."""
    memories = [
        _root_node(doc_id="d_xyz789"),
        _topic_node(doc_id="d_xyz789", topic_index=3, topic="Venues"),
    ]
    doc_dir = await KnowledgeWriter.write(memories, tmp_path)

    fm, _ = _parse_md(doc_dir / "3_Venues.md")
    assert fm["id"] == "d_xyz789_3"
    assert fm["node_id"] == "d_xyz789_3"
    assert fm["doc_id"] == "d_xyz789"


async def test_depth_1_parent_node_id_is_null(tmp_path: Path) -> None:
    """Direct children of root (depth=1) have parent_node_id=null."""
    memories = [
        _root_node(),
        _topic_node(topic_index=1, depth=1, parent_index=0),
    ]
    doc_dir = await KnowledgeWriter.write(memories, tmp_path)

    fm, _ = _parse_md(doc_dir / "1_Opening_Ceremony.md")
    assert fm["parent_node_id"] is None


async def test_depth_gt1_parent_node_id_set(tmp_path: Path) -> None:
    """Nested topics (depth>1) get ``{doc_id}_{parent_index}``."""
    memories = [
        _root_node(children_index=[1]),
        _topic_node(
            topic_index=1,
            topic="Section A",
            depth=1,
            parent_index=0,
            children_index=[2],
        ),
        _topic_node(
            topic_index=2,
            topic="Sub Section",
            depth=2,
            parent_index=1,
            topic_path="Olympics Plan > Section A > Sub Section",
        ),
    ]
    doc_dir = await KnowledgeWriter.write(memories, tmp_path)

    fm, _ = _parse_md(doc_dir / "2_Sub_Section.md")
    assert fm["parent_node_id"] == "d_abc123_1"


async def test_children_node_ids_mapping(tmp_path: Path) -> None:
    memories = [
        _root_node(children_index=[1, 2]),
        _topic_node(topic_index=1, topic="A", children_index=[3]),
        _topic_node(topic_index=2, topic="B"),
    ]
    doc_dir = await KnowledgeWriter.write(memories, tmp_path)

    fm, _ = _parse_md(doc_dir / "1_A.md")
    assert fm["children_node_ids"] == ["d_abc123_3"]


async def test_empty_slug_fallback_for_topic(tmp_path: Path) -> None:
    """Topic with all-special-chars name falls back to ``{idx}_topic_{idx}.md``."""
    memories = [
        _root_node(),
        _topic_node(topic_index=5, topic="!!!@@@###"),
    ]
    doc_dir = await KnowledgeWriter.write(memories, tmp_path)

    assert (doc_dir / "5_topic_5.md").is_file()


async def test_empty_slug_fallback_for_title(tmp_path: Path) -> None:
    """Document title with all-special-chars falls back to ``doc_{doc_id}``."""
    doc_id = "d_abcdef1234567890"
    memories = [
        _root_node(
            doc_id=doc_id,
            topic="$$$%%%^^^",
        ),
    ]
    doc_dir = await KnowledgeWriter.write(memories, tmp_path)

    assert doc_dir == tmp_path / "Sports" / f"doc_{doc_id}"
    assert (doc_dir / "index.md").is_file()


async def test_empty_category_id_fallback_to_others(tmp_path: Path) -> None:
    memories = [_root_node(category_id="")]
    doc_dir = await KnowledgeWriter.write(memories, tmp_path)

    assert doc_dir.parent.name == "Others"
    fm, _ = _parse_md(doc_dir / "index.md")
    assert fm["category_id"] == "Others"


async def test_overwrite_replaces_existing_directory(tmp_path: Path) -> None:
    """Second write deletes old files and creates new ones."""
    memories_v1 = [
        _root_node(),
        _topic_node(topic_index=1, topic="Old Topic"),
    ]
    doc_dir = await KnowledgeWriter.write(memories_v1, tmp_path)
    assert (doc_dir / "1_Old_Topic.md").is_file()

    memories_v2 = [
        _root_node(),
        _topic_node(topic_index=1, topic="New Topic"),
    ]
    doc_dir = await KnowledgeWriter.write(memories_v2, tmp_path)

    assert (doc_dir / "1_New_Topic.md").is_file()
    assert not (doc_dir / "1_Old_Topic.md").exists()


async def test_empty_memories_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        await KnowledgeWriter.write([], tmp_path)


async def test_no_root_node_raises_value_error(tmp_path: Path) -> None:
    memories = [_topic_node(topic_index=1)]
    with pytest.raises(ValueError, match="root node"):
        await KnowledgeWriter.write(memories, tmp_path)


async def test_source_name_and_type_in_index_frontmatter(
    tmp_path: Path,
) -> None:
    memories = [_root_node()]
    doc_dir = await KnowledgeWriter.write(
        memories,
        tmp_path,
        source_name="https://example.com/doc",
        source_type="url",
    )
    fm, _ = _parse_md(doc_dir / "index.md")
    assert fm["source_name"] == "https://example.com/doc"
    assert fm["source_type"] == "url"


async def test_source_fields_omitted_when_none(tmp_path: Path) -> None:
    memories = [_root_node()]
    doc_dir = await KnowledgeWriter.write(memories, tmp_path)
    fm, _ = _parse_md(doc_dir / "index.md")
    assert "source_name" not in fm
    assert "source_type" not in fm


async def test_content_labels_preserved(tmp_path: Path) -> None:
    memories = [
        _root_node(),
        _topic_node(
            topic_index=1,
            content_labels=["sports", "ceremony"],
        ),
    ]
    doc_dir = await KnowledgeWriter.write(memories, tmp_path)
    fm, _ = _parse_md(doc_dir / "1_Opening_Ceremony.md")
    assert fm["content_labels"] == ["sports", "ceremony"]


async def test_topic_path_preserved(tmp_path: Path) -> None:
    memories = [
        _root_node(),
        _topic_node(
            topic_index=1,
            topic_path="Olympics Plan > Opening Ceremony",
        ),
    ]
    doc_dir = await KnowledgeWriter.write(memories, tmp_path)
    fm, _ = _parse_md(doc_dir / "1_Opening_Ceremony.md")
    assert fm["topic_path"] == "Olympics Plan > Opening Ceremony"
