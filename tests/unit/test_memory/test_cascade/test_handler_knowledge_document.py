"""Tests for :class:`KnowledgeDocumentHandler` — SQLite-only cascade.

KnowledgeDocumentHandler writes to **SQLite only** — no LanceDB, no
embedding, no tokenization.  The handler reads ``index.md``, extracts
frontmatter + body (summary), and upserts to ``knowledge_documents``.

Coverage:

- ``handle_added_or_modified`` with valid frontmatter → upserts,
  returns ``upserted=1``
- ``handle_added_or_modified`` with wrong ``type`` → returns
  ``skipped=1``, no upsert
- ``handle_deleted`` on an indexed path → calls
  ``delete_by_md_path``, returns ``deleted=1``
- ``handle_deleted`` on an unknown path → returns ``deleted=0``
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.sqlite import DocumentUpsertPayload
from everos.memory.cascade.handlers import (
    HandlerDeps,
    KnowledgeDocumentHandler,
)

# ── Stubs ──────────────────────────────────────────────────────────────


class _StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _StubEmbedder(EmbeddingProvider):
    dim = 1024

    async def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [await self.embed(t) for t in texts]


# ── Fake repo ──────────────────────────────────────────────────────────


class _FakeSqliteRepo:
    """In-memory stand-in for ``knowledge_document_repo``."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self.upserts: list[dict] = []
        self.deletes: list[str] = []

    async def upsert_from_handler(self, payload: DocumentUpsertPayload) -> None:
        data = dataclasses.asdict(payload)
        self.upserts.append(data)
        self.rows[payload.doc_id] = data

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
def fake_sqlite(monkeypatch: pytest.MonkeyPatch) -> _FakeSqliteRepo:
    from everos.memory.cascade.handlers import knowledge_document as mod

    repo = _FakeSqliteRepo()
    monkeypatch.setattr(mod, "knowledge_document_repo", repo)
    return repo


# ── Helpers ────────────────────────────────────────────────────────────

_SAMPLE_FRONTMATTER = {
    "type": "knowledge_document",
    "id": "doc_budget",
    "category_id": "finance",
    "title": "Budget Planning Guide",
    "source_name": "Internal Wiki",
    "source_type": "wiki",
    "schema_version": 1,
}

_SAMPLE_BODY = "An overview of budget planning practices for Q4."


def _write_document_md(
    memory_root: MemoryRoot,
    *,
    frontmatter: dict | None = None,
    body: str = _SAMPLE_BODY,
) -> str:
    """Write a knowledge document ``index.md`` on disk; return relative path."""
    fm = frontmatter or dict(_SAMPLE_FRONTMATTER)
    lines = ["---"]
    for key, value in fm.items():
        if value is None:
            lines.append(f"{key}: null")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    lines.append(body)
    content = "\n".join(lines)

    rel_dir = "default_app/default_project/knowledge/finance/Budget_Planning"
    abs_dir = memory_root.root / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)
    (abs_dir / "index.md").write_text(content, encoding="utf-8")
    return f"{rel_dir}/index.md"


def _handler(memory_root: MemoryRoot) -> KnowledgeDocumentHandler:
    return KnowledgeDocumentHandler(
        HandlerDeps(
            memory_root=memory_root,
            embedder=_StubEmbedder(),
            tokenizer=_StubTokenizer(),
        )
    )


# ── Tests ──────────────────────────────────────────────────────────────


async def test_handle_added_or_modified_upserts_to_sqlite(
    memory_root: MemoryRoot,
    fake_sqlite: _FakeSqliteRepo,
) -> None:
    md_path = _write_document_md(memory_root)
    outcome = await _handler(memory_root).handle_added_or_modified(md_path)

    assert outcome.upserted == 1
    assert outcome.deleted == 0
    assert outcome.skipped == 0

    assert len(fake_sqlite.upserts) == 1
    row = fake_sqlite.upserts[0]
    assert row["doc_id"] == "doc_budget"
    assert row["category_id"] == "finance"
    assert row["title"] == "Budget Planning Guide"
    assert row["source_name"] == "Internal Wiki"
    assert row["source_type"] == "wiki"
    assert row["summary"] == _SAMPLE_BODY
    assert row["app_id"] == "default"
    assert row["project_id"] == "default"
    assert row["md_path"] == md_path


async def test_handle_added_or_modified_wrong_type_skips(
    memory_root: MemoryRoot,
    fake_sqlite: _FakeSqliteRepo,
) -> None:
    """A file whose ``type`` is not ``knowledge_document`` is skipped."""
    fm = dict(_SAMPLE_FRONTMATTER, type="knowledge_topic")
    md_path = _write_document_md(memory_root, frontmatter=fm)
    outcome = await _handler(memory_root).handle_added_or_modified(md_path)

    assert outcome.skipped == 1
    assert outcome.upserted == 0
    assert len(fake_sqlite.upserts) == 0


async def test_handle_deleted_removes_row(
    memory_root: MemoryRoot,
    fake_sqlite: _FakeSqliteRepo,
) -> None:
    """``handle_deleted`` calls ``delete_by_md_path`` and returns ``deleted=1``."""
    md_path = _write_document_md(memory_root)
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(md_path)

    outcome = await handler.handle_deleted(md_path)

    assert outcome.deleted == 1
    assert outcome.upserted == 0
    assert md_path in fake_sqlite.deletes


async def test_handle_deleted_unknown_path_returns_zero(
    memory_root: MemoryRoot,
    fake_sqlite: _FakeSqliteRepo,
) -> None:
    """Deleting a path that was never indexed returns ``deleted=0``."""
    handler = _handler(memory_root)
    outcome = await handler.handle_deleted("nonexistent/index.md")

    assert outcome.deleted == 0
    assert outcome.upserted == 0
