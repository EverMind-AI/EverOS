"""KnowledgeWriter — write knowledge document + topic markdown files.

Knowledge storage uses a **directory per document** layout::

    knowledge/{category_id}/{title_dirname}/index.md           ← document root
    knowledge/{category_id}/{title_dirname}/1_topic_slug.md    ← topic node
    knowledge/{category_id}/{title_dirname}/2_topic_slug.md    ← topic node

Each call to :meth:`write` replaces the entire document directory
(delete-then-recreate) so that stale topic files from a prior extraction
do not linger.

The writer is intentionally **static** — it takes a ``knowledge_dir``
path rather than binding to :class:`MemoryRoot`. The service layer
resolves ``knowledge_dir`` from ``MemoryRoot.knowledge_dir(app, project)``
and passes it in. This keeps the writer decoupled from the root-resolution
logic and easier to test.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import anyio
import yaml
from everalgo.types import KnowledgeMemory

from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

_MAX_DIRNAME_LEN = 50
_SAFE_CHARS = re.compile(r"[^\w\-.]", re.UNICODE)


# ── Writer ────────────────────────────────────────────────────────────────


class KnowledgeWriter:
    """Convert ``KnowledgeMemory`` list to markdown files."""

    @staticmethod
    async def write(
        memories: list[KnowledgeMemory],
        knowledge_dir: Path,
        *,
        source_name: str | None = None,
        source_type: str | None = None,
    ) -> Path:
        """Write md files and return the document directory path.

        Args:
            memories: Flat list of nodes produced by everalgo extraction.
                Must contain exactly one root node (``topic_index=0``).
            knowledge_dir: Base ``knowledge/`` directory (from
                ``MemoryRoot.knowledge_dir``).
            source_name: Optional provenance label (e.g. URL, filename).
            source_type: Optional provenance type (e.g. ``"url"``,
                ``"file"``).

        Returns:
            Absolute path of the written document directory.

        Raises:
            ValueError: If *memories* is empty or has no root node.
        """
        root_node, topic_nodes = _split_root_and_topics(memories)
        doc_dir = _resolve_doc_dir(knowledge_dir, root_node)

        # Overwrite: remove existing directory, then recreate.
        await _remove_dir(doc_dir)
        await anyio.Path(doc_dir).mkdir(parents=True, exist_ok=True)

        await _write_index(doc_dir, root_node, source_name, source_type)
        for node in topic_nodes:
            await _write_topic(doc_dir, node, root_node.doc_id)

        logger.info(
            "knowledge document written",
            doc_id=root_node.doc_id,
            category_id=root_node.category_id or "Others",
            topic_count=len(topic_nodes),
        )
        return doc_dir


# ── Internals ─────────────────────────────────────────────────────────────


def _split_root_and_topics(
    memories: list[KnowledgeMemory],
) -> tuple[KnowledgeMemory, list[KnowledgeMemory]]:
    """Separate the root node from topic nodes.

    Raises:
        ValueError: If *memories* is empty or contains no root node.
    """
    if not memories:
        raise ValueError("memories must not be empty")

    root: KnowledgeMemory | None = None
    topics: list[KnowledgeMemory] = []
    for m in memories:
        if m.topic_index == 0:
            root = m
        else:
            topics.append(m)

    if root is None:
        raise ValueError("memories must contain a root node (topic_index=0)")
    return root, topics


def _sanitize_dirname(raw: str, fallback: str) -> str:
    """Produce a safe directory/file name segment.

    * Replace spaces with underscores.
    * Strip characters outside ``[a-zA-Z0-9_\\-.]``.
    * Truncate to 50 characters.
    * Fall back to *fallback* if the result is empty.
    """
    slug = raw.replace(" ", "_")
    slug = _SAFE_CHARS.sub("", slug)
    slug = slug[:_MAX_DIRNAME_LEN]
    return slug if slug else fallback


def _resolve_doc_dir(knowledge_dir: Path, root: KnowledgeMemory) -> Path:
    """Build the document directory path from category, title, and doc_id."""
    category = _sanitize_dirname(
        root.category_id if root.category_id else "Others", "Others"
    )
    title_slug = _sanitize_dirname(root.topic, "doc")
    dir_name = f"{title_slug}_{root.doc_id}"
    return knowledge_dir / category / dir_name


def _build_index_frontmatter(
    root: KnowledgeMemory,
    source_name: str | None,
    source_type: str | None,
) -> dict[str, object]:
    """Build YAML frontmatter dict for the document index file."""
    category = root.category_id if root.category_id else "Others"
    fm: dict[str, object] = {
        "type": "knowledge_document",
        "id": root.doc_id,
        "doc_id": root.doc_id,
        "category_id": category,
        "title": root.topic,
        "schema_version": 1,
    }
    if source_name is not None:
        fm["source_name"] = source_name
    if source_type is not None:
        fm["source_type"] = source_type
    return fm


def _build_topic_frontmatter(
    node: KnowledgeMemory,
    doc_id: str,
) -> dict[str, object]:
    """Build YAML frontmatter dict for a single topic file."""
    category = node.category_id if node.category_id else "Others"
    node_id = f"{doc_id}_{node.topic_index}"

    parent_node_id = None if node.depth <= 1 else f"{doc_id}_{node.parent_index}"

    children_node_ids = [f"{doc_id}_{ci}" for ci in node.children_index]

    return {
        "type": "knowledge_topic",
        "id": node_id,
        "node_id": node_id,
        "doc_id": doc_id,
        "category_id": category,
        "topic_index": node.topic_index,
        "topic_name": node.topic,
        "topic_path": node.topic_path,
        "summary": node.summary,
        "depth": node.depth,
        "parent_node_id": parent_node_id,
        "children_node_ids": children_node_ids,
        "content_labels": node.content_labels,
        "schema_version": 1,
    }


def _dump_yaml_frontmatter(meta: dict[str, object]) -> str:
    """Render a YAML frontmatter block with ``---`` delimiters."""
    yaml_block = yaml.safe_dump(
        meta,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return f"---\n{yaml_block}---\n"


def _ensure_trailing_newline(text: str) -> str:
    """Append a newline if *text* does not already end with one."""
    if not text:
        return ""
    return text if text.endswith("\n") else text + "\n"


async def _write_file(path: Path, content: str) -> None:
    """Write content to *path* via anyio (async)."""
    await anyio.Path(path.parent).mkdir(parents=True, exist_ok=True)
    await anyio.Path(path).write_text(content, encoding="utf-8")


async def _write_index(
    doc_dir: Path,
    root: KnowledgeMemory,
    source_name: str | None,
    source_type: str | None,
) -> None:
    """Write index.md with frontmatter and summary body."""
    fm = _build_index_frontmatter(root, source_name, source_type)
    body = _ensure_trailing_newline(root.summary)
    content = _dump_yaml_frontmatter(fm) + body
    await _write_file(doc_dir / "index.md", content)


async def _write_topic(
    doc_dir: Path,
    node: KnowledgeMemory,
    doc_id: str,
) -> None:
    """Write a numbered topic md file with frontmatter and content body."""
    slug = _sanitize_dirname(node.topic, f"topic_{node.topic_index}")
    filename = f"{node.topic_index}_{slug}.md"
    fm = _build_topic_frontmatter(node, doc_id)
    body = _ensure_trailing_newline(node.content)
    content = _dump_yaml_frontmatter(fm) + body
    await _write_file(doc_dir / filename, content)


async def _remove_dir(path: Path) -> None:
    """Remove directory tree if it exists (sync shutil offloaded)."""
    if await anyio.Path(path).is_dir():
        await anyio.to_thread.run_sync(shutil.rmtree, path)
