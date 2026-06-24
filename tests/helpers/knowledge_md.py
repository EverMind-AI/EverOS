"""Helpers for reading knowledge md files (truth layer) in tests.

Provides functions to parse index.md and topic md files so tests can
assert against the truth layer rather than only the derived SQLite/API
responses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_document_md(doc_dir: Path) -> dict[str, Any]:
    """Read ``index.md`` from *doc_dir*, return frontmatter + body.

    Returns:
        ``{"frontmatter": dict, "body": str}``

    Raises:
        FileNotFoundError: If ``index.md`` does not exist.
    """
    index = doc_dir / "index.md"
    text = index.read_text(encoding="utf-8")
    return _parse_md(text)


def read_topic_mds(doc_dir: Path) -> list[dict[str, Any]]:
    """Read all ``N_*.md`` topic files, return list of frontmatter + body.

    Sorted by filename (i.e. by topic_index prefix).
    """
    results = []
    for f in sorted(doc_dir.iterdir()):
        if f.name == "index.md" or f.suffix != ".md":
            continue
        text = f.read_text(encoding="utf-8")
        parsed = _parse_md(text)
        parsed["filename"] = f.name
        results.append(parsed)
    return results


def find_doc_dir(knowledge_dir: Path, doc_id: str) -> Path | None:
    """Scan *knowledge_dir* tree to find the directory containing *doc_id*'s index.md.

    Returns the directory Path, or ``None`` if not found.
    """
    if not knowledge_dir.exists():
        return None
    for index_md in knowledge_dir.rglob("index.md"):
        text = index_md.read_text(encoding="utf-8")
        parsed = _parse_md(text)
        if parsed["frontmatter"].get("doc_id") == doc_id:
            return index_md.parent
    return None


def _parse_md(text: str) -> dict[str, Any]:
    """Split YAML frontmatter from body."""
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {"frontmatter": {}, "body": text}
    fm: dict[str, Any] = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    return {"frontmatter": fm, "body": body}
