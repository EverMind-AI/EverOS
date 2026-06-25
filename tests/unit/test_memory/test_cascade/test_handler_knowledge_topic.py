"""Tests for :class:`KnowledgeTopicHandler` — cross-storage cascade.

KnowledgeTopic is the first handler to write to **both** LanceDB and
SQLite. The handler reads ``<n>_<name>.md``, extracts frontmatter,
computes a content digest, embeds the summary, and upserts to both
stores. These tests build the md file on disk and verify:

- upsert to both stores on first pass
- skip when digest unchanged
- skip when ``type`` frontmatter is wrong
- delete from both stores on ``handle_deleted``
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb import KnowledgeTopic
from everos.infra.persistence.sqlite import TopicUpsertPayload
from everos.memory.cascade.handlers import HandlerDeps, KnowledgeTopicHandler

# ── Stubs ──────────────────────────────────────────────────────────────


class _StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _StubEmbedder(EmbeddingProvider):
    dim = 1024

    async def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [await self.embed(t) for t in texts]


# ── Fake repos ─────────────────────────────────────────────────────────


class _FakeLanceRepo:
    """In-memory stand-in for the LanceDB knowledge_topic_repo."""

    def __init__(self) -> None:
        self.rows: dict[str, KnowledgeTopic] = {}
        self.upserts: list[list[KnowledgeTopic]] = []
        self.deletes: list[str] = []

    async def get_by_id(self, row_id: str) -> KnowledgeTopic | None:
        return self.rows.get(row_id)

    async def upsert(self, rows: list[KnowledgeTopic]) -> None:
        self.upserts.append(list(rows))
        for row in rows:
            self.rows[row.id] = row

    async def delete_by_md_path(self, md_path: str) -> int:
        self.deletes.append(md_path)
        before = len(self.rows)
        self.rows = {k: v for k, v in self.rows.items() if v.md_path != md_path}
        return before - len(self.rows)


class _FakeSqliteRepo:
    """In-memory stand-in for knowledge_topic_sqlite_repo."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.upserts: list[dict] = []
        self.deletes: list[str] = []

    async def upsert_from_handler(self, payload: TopicUpsertPayload) -> None:
        data = dataclasses.asdict(payload)
        self.upserts.append(data)
        self.rows[payload.node_id] = data

    async def delete_by_md_path(self, md_path: str) -> int:
        self.deletes.append(md_path)
        before = len(self.rows)
        self.rows = {k: v for k, v in self.rows.items() if v.get("md_path") != md_path}
        return before - len(self.rows)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


@pytest.fixture
def fake_lance(monkeypatch: pytest.MonkeyPatch) -> _FakeLanceRepo:
    from everos.memory.cascade.handlers import knowledge_topic as mod

    repo = _FakeLanceRepo()
    monkeypatch.setattr(mod, "knowledge_topic_repo", repo)
    return repo


@pytest.fixture
def fake_sqlite(monkeypatch: pytest.MonkeyPatch) -> _FakeSqliteRepo:
    from everos.memory.cascade.handlers import knowledge_topic as mod

    repo = _FakeSqliteRepo()
    monkeypatch.setattr(mod, "knowledge_topic_sqlite_repo", repo)
    return repo


# ── Helpers ────────────────────────────────────────────────────────────

_SAMPLE_FRONTMATTER = {
    "type": "knowledge_topic",
    "id": "node_001",
    "node_id": "node_001",
    "doc_id": "doc_budget",
    "category_id": "finance",
    "topic_index": 1,
    "topic_name": "Budget Planning",
    "topic_path": "finance/Budget_Planning",
    "summary": "Overview of budget planning practices.",
    "depth": 0,
    "parent_node_id": None,
    "children_node_ids": ["node_002", "node_003"],
    "content_labels": ["budget", "planning"],
    "schema_version": 1,
}


def _write_topic_md(
    memory_root: MemoryRoot,
    *,
    frontmatter: dict | None = None,
    body: str = "Budget planning involves setting goals and tracking expenses.\n",
) -> str:
    """Write a knowledge topic md file on disk, return relative md_path."""
    fm = frontmatter or dict(_SAMPLE_FRONTMATTER)
    # Build the YAML frontmatter string.
    lines = ["---"]
    for key, value in fm.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                rendered = f"  - {item!r}" if isinstance(item, str) else f"  - {item}"
                lines.append(rendered)
        elif value is None:
            lines.append(f"{key}: null")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append(body)
    content = "\n".join(lines)

    rel_dir = "default_app/default_project/knowledge/finance/Budget_Planning"
    abs_dir = memory_root.root / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)
    filename = "1_Budget_Planning.md"
    (abs_dir / filename).write_text(content, encoding="utf-8")
    return f"{rel_dir}/{filename}"


def _handler(memory_root: MemoryRoot) -> KnowledgeTopicHandler:
    return KnowledgeTopicHandler(
        HandlerDeps(
            memory_root=memory_root,
            embedder=_StubEmbedder(),
            tokenizer=_StubTokenizer(),
        )
    )


# ── Tests ──────────────────────────────────────────────────────────────


async def test_handle_added_or_modified_upserts_to_both_stores(
    memory_root: MemoryRoot,
    fake_lance: _FakeLanceRepo,
    fake_sqlite: _FakeSqliteRepo,
) -> None:
    md_path = _write_topic_md(memory_root)
    outcome = await _handler(memory_root).handle_added_or_modified(md_path)

    assert outcome.upserted == 1
    assert outcome.deleted == 0
    assert outcome.skipped == 0

    # LanceDB row assertions.
    assert len(fake_lance.upserts) == 1
    row = fake_lance.upserts[0][0]
    assert row.id == "node_001"
    assert row.doc_id == "doc_budget"
    assert row.category_id == "finance"
    assert row.topic_name == "Budget Planning"
    assert row.topic_path == "finance/Budget_Planning"
    assert row.depth == 0
    assert row.parent_node_id == ""
    assert row.summary == "Overview of budget planning practices."
    assert "budget" in row.summary_tokens.lower()
    assert "planning" in row.content_tokens.lower()
    assert row.content_labels == ["budget", "planning"]
    assert row.md_path == md_path
    assert len(row.vector) == 1024
    assert row.content_sha256  # non-empty digest

    # SQLite row assertions.
    assert len(fake_sqlite.upserts) == 1
    sq = fake_sqlite.upserts[0]
    assert sq["node_id"] == "node_001"
    assert sq["doc_id"] == "doc_budget"
    assert sq["topic_index"] == 1
    assert sq["topic_name"] == "Budget Planning"
    assert sq["summary"] == "Overview of budget planning practices."
    assert "Budget planning involves" in sq["content"]
    assert sq["children_node_ids"] == json.dumps(["node_002", "node_003"])
    assert sq["content_labels"] == json.dumps(["budget", "planning"])
    assert sq["md_path"] == md_path


async def test_same_digest_skips(
    memory_root: MemoryRoot,
    fake_lance: _FakeLanceRepo,
    fake_sqlite: _FakeSqliteRepo,
) -> None:
    """Second pass with identical content skips both stores."""
    md_path = _write_topic_md(memory_root)
    handler = _handler(memory_root)

    first = await handler.handle_added_or_modified(md_path)
    assert first.upserted == 1

    second = await handler.handle_added_or_modified(md_path)
    assert second.upserted == 0
    assert second.skipped == 1
    # Only one upsert batch total.
    assert len(fake_lance.upserts) == 1
    assert len(fake_sqlite.upserts) == 1


async def test_wrong_type_skips(
    memory_root: MemoryRoot,
    fake_lance: _FakeLanceRepo,
    fake_sqlite: _FakeSqliteRepo,
) -> None:
    """A file whose ``type`` is not ``knowledge_topic`` is skipped."""
    fm = dict(_SAMPLE_FRONTMATTER, type="knowledge_document")
    md_path = _write_topic_md(memory_root, frontmatter=fm)
    outcome = await _handler(memory_root).handle_added_or_modified(md_path)

    assert outcome.skipped == 1
    assert outcome.upserted == 0
    assert len(fake_lance.upserts) == 0
    assert len(fake_sqlite.upserts) == 0


async def test_handle_deleted_removes_from_both_stores(
    memory_root: MemoryRoot,
    fake_lance: _FakeLanceRepo,
    fake_sqlite: _FakeSqliteRepo,
) -> None:
    """``handle_deleted`` calls ``delete_by_md_path`` on both repos."""
    # Seed a row so delete_by_md_path has something to find.
    md_path = _write_topic_md(memory_root)
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(md_path)

    outcome = await handler.handle_deleted(md_path)

    assert outcome.deleted == 1
    assert outcome.upserted == 0
    assert md_path in fake_lance.deletes
    assert md_path in fake_sqlite.deletes


async def test_content_edit_triggers_upsert(
    memory_root: MemoryRoot,
    fake_lance: _FakeLanceRepo,
    fake_sqlite: _FakeSqliteRepo,
) -> None:
    """Editing the body changes the digest and triggers a re-upsert."""
    md_path = _write_topic_md(memory_root, body="Original content.\n")
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(md_path)

    # Edit the body on disk.
    abs_path = memory_root.root / md_path
    text = abs_path.read_text(encoding="utf-8")
    abs_path.write_text(text.replace("Original content.", "Revised content."))

    outcome = await handler.handle_added_or_modified(md_path)
    assert outcome.upserted == 1
    assert len(fake_lance.upserts) == 2
    assert len(fake_sqlite.upserts) == 2


async def test_handle_deleted_on_unknown_path_returns_zero(
    memory_root: MemoryRoot,
    fake_lance: _FakeLanceRepo,
    fake_sqlite: _FakeSqliteRepo,
) -> None:
    """Deleting a path that was never indexed returns deleted=0."""
    handler = _handler(memory_root)
    outcome = await handler.handle_deleted("nonexistent/path.md")
    assert outcome.deleted == 0
    assert outcome.upserted == 0
