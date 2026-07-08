"""Tests for :class:`EpisodeHandler` — md → LanceDB row reconcile.

Uses a real on-disk md file (via :class:`EpisodeWriter`) to exercise
the parse → diff → upsert path. The lancedb repo is faked since the
production singleton would need a live LanceDB connection; this keeps
the test in-memory while still validating row construction and the
3-way diff branch behaviour.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import pytest

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb import Episode
from everos.infra.persistence.markdown import EpisodeWriter
from everos.memory.cascade.handlers import HandlerDeps
from everos.memory.cascade.handlers.episode import EpisodeHandler


class _StubTokenizer(Tokenizer):
    """Returns the input split on whitespace — deterministic for assertions."""

    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _StubEmbedder(EmbeddingProvider):
    """Returns a fixed 1024-dim vector; records call count."""

    dim = 1024

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [0.1] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [await self.embed(t) for t in texts]


class _FakeEpisodeRepo:
    """Recording repo — captures upserts / deletes the handler issues."""

    def __init__(self) -> None:
        self.upserts: list[list[Episode]] = []
        self.deletes: list[str] = []
        self.rows: list[Episode] = []

    async def find_where(self, where: str, *, limit: int = 100) -> list[Episode]:
        # Honour only the md_path = '...' filter the handler emits.
        prefix = "md_path = '"
        if where.startswith(prefix):
            md_path = where[len(prefix) :].rstrip("'")
            return [r for r in self.rows if r.md_path == md_path]
        return []

    async def upsert(self, rows: list[Episode]) -> None:
        self.upserts.append(list(rows))
        # Reflect into ``self.rows`` so a follow-up find_where sees the state.
        by_id = {r.id: r for r in self.rows}
        for r in rows:
            by_id[r.id] = r
        self.rows = list(by_id.values())

    async def delete(self, predicate: str) -> None:
        self.deletes.append(predicate)

    async def delete_by_md_path(self, md_path: str) -> int:
        before = len(self.rows)
        self.rows = [r for r in self.rows if r.md_path != md_path]
        return before - len(self.rows)


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


@pytest.fixture
def fake_repo(monkeypatch: pytest.MonkeyPatch) -> _FakeEpisodeRepo:
    """Swap the class-level ``lance_repo`` on EpisodeHandler.

    After the BaseDailyLogHandler refactor, the repo binding is a
    ClassVar resolved at class-definition time; patching the module
    attribute would no longer reach the handler's call sites.
    """
    from everos.memory.cascade.handlers.episode import EpisodeHandler

    repo = _FakeEpisodeRepo()
    monkeypatch.setattr(EpisodeHandler, "lance_repo", repo)
    return repo


async def _write_one_entry(writer: EpisodeWriter, owner_id: str, body: str) -> str:
    """Append a single episode entry, return the md path (relative)."""
    today = _dt.date(2026, 5, 14)
    await writer.append_entry(
        owner_id,
        inline={
            "owner_id": owner_id,
            "session_id": "s1",
            "timestamp": "2026-05-14T10:00:00+00:00",
            "parent_type": "memcell",
            "parent_id": "mc_test_parent",
            "sender_ids": [owner_id],
        },
        sections={"Subject": "Test", "Summary": "Stub", "Content": body},
        date=today,
    )
    return (
        f"default_app/default_project/users/{owner_id}/episodes/episode-2026-05-14.md"
    )


def _build_handler(
    memory_root: MemoryRoot,
) -> tuple[EpisodeHandler, _StubEmbedder]:
    embedder = _StubEmbedder()
    deps = HandlerDeps(
        memory_root=memory_root,
        embedder=embedder,
        tokenizer=_StubTokenizer(),
    )
    return EpisodeHandler(deps), embedder


# ── happy path ───────────────────────────────────────────────────────────


async def test_added_entry_upserts_typed_row(
    memory_root: MemoryRoot, fake_repo: _FakeEpisodeRepo
) -> None:
    writer = EpisodeWriter(memory_root)
    rel = await _write_one_entry(writer, "u1", "hello world")

    handler, embedder = _build_handler(memory_root)
    outcome = await handler.handle_added_or_modified(rel)

    assert outcome.upserted == 1
    assert outcome.deleted == 0
    assert outcome.skipped == 0
    # Subject "Test" is present → two embed calls (content + subject).
    assert embedder.calls == 2
    assert len(fake_repo.upserts) == 1
    row = fake_repo.upserts[0][0]
    assert row.owner_id == "u1"
    assert row.owner_type == "user"
    # Scope is parsed back from the md path's <app>/<project> prefix.
    assert row.app_id == "default"
    assert row.project_id == "default"
    assert row.session_id == "s1"
    assert row.parent_id == "mc_test_parent"
    assert row.parent_type == "memcell"
    assert row.episode == "hello world"
    # episode_tokens includes subject keywords appended after content tokens.
    assert row.episode_tokens == "hello world Test"
    assert row.subject == "Test"
    assert row.md_path == rel
    assert row.entry_id.startswith("ep_")
    assert row.id == f"u1_{row.entry_id}"
    assert len(row.vector) == 1024
    assert row.subject_vector is not None
    assert len(row.subject_vector) == 1024


async def test_unchanged_entry_is_skipped_no_embed_call(
    memory_root: MemoryRoot, fake_repo: _FakeEpisodeRepo
) -> None:
    """Second handle run with no md change → skipped + no embed call."""
    writer = EpisodeWriter(memory_root)
    rel = await _write_one_entry(writer, "u1", "hello world")

    handler, embedder = _build_handler(memory_root)
    await handler.handle_added_or_modified(rel)  # first pass populates fake repo
    fake_repo.upserts.clear()
    embedder.calls = 0

    outcome = await handler.handle_added_or_modified(rel)
    assert outcome.skipped == 1
    assert outcome.upserted == 0
    assert embedder.calls == 0
    assert fake_repo.upserts == []


async def test_modified_entry_reembeds(
    memory_root: MemoryRoot, fake_repo: _FakeEpisodeRepo
) -> None:
    """Changing the entry body bumps the sha → re-embed + upsert."""
    writer = EpisodeWriter(memory_root)
    rel = await _write_one_entry(writer, "u1", "original content")

    handler, embedder = _build_handler(memory_root)
    await handler.handle_added_or_modified(rel)
    # Tamper with the row's stored sha so the next pass sees a mismatch.
    fake_repo.rows[0] = fake_repo.rows[0].model_copy(
        update={"content_sha256": "0" * 64}
    )
    fake_repo.upserts.clear()
    embedder.calls = 0

    outcome = await handler.handle_added_or_modified(rel)
    assert outcome.upserted == 1
    assert outcome.skipped == 0
    # Subject "Test" is present → two embed calls (content + subject).
    assert embedder.calls == 2


async def test_no_subject_skips_subject_embed(
    memory_root: MemoryRoot, fake_repo: _FakeEpisodeRepo
) -> None:
    """When Subject is absent, subject_vector is None; only one embed call is made."""
    today = _dt.date(2026, 5, 14)
    writer = EpisodeWriter(memory_root)
    await writer.append_entry(
        "u2",
        inline={
            "owner_id": "u2",
            "session_id": "s2",
            "timestamp": "2026-05-14T10:00:00+00:00",
            "parent_type": "memcell",
            "parent_id": "mc_no_subject",
            "sender_ids": ["u2"],
        },
        sections={"Content": "content only"},
        date=today,
    )
    rel = "default_app/default_project/users/u2/episodes/episode-2026-05-14.md"

    handler, embedder = _build_handler(memory_root)
    outcome = await handler.handle_added_or_modified(rel)

    assert outcome.upserted == 1
    # No Subject → single embed call for content only.
    assert embedder.calls == 1
    row = fake_repo.upserts[0][0]
    assert row.subject_vector is None
    assert row.episode_tokens == "content only"


# ── deletion paths ───────────────────────────────────────────────────────


async def test_handle_deleted_wipes_md_path_rows(
    memory_root: MemoryRoot, fake_repo: _FakeEpisodeRepo
) -> None:
    writer = EpisodeWriter(memory_root)
    rel = await _write_one_entry(writer, "u1", "hello")
    handler, _embedder = _build_handler(memory_root)
    await handler.handle_added_or_modified(rel)
    assert fake_repo.rows  # populated

    outcome = await handler.handle_deleted(rel)
    assert outcome.deleted == 1
    assert fake_repo.rows == []


# ── error path ───────────────────────────────────────────────────────────


async def test_missing_timestamp_raises_value_error(
    memory_root: MemoryRoot, fake_repo: _FakeEpisodeRepo
) -> None:
    """Malformed inline surfaces as ValueError — worker treats unrecoverable."""
    writer = EpisodeWriter(memory_root)
    # Manually bypass the writer to drop timestamp.
    today = _dt.date(2026, 5, 14)
    await writer.append_entry(
        "u1",
        inline={"owner_id": "u1", "session_id": "s1"},  # no timestamp
        sections={"Content": "x"},
        date=today,
    )
    rel = "default_app/default_project/users/u1/episodes/episode-2026-05-14.md"

    handler, _embedder = _build_handler(memory_root)
    with pytest.raises(ValueError, match="timestamp"):
        await handler.handle_added_or_modified(rel)


# ── unused noqa suppressor (keep imports tidy) ──────────────────────────


_: Any = None
