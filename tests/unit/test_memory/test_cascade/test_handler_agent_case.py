"""Tests for :class:`AgentCaseHandler` — md -> LanceDB ``agent_case`` row.

The kind this file covers had no handler test of its own, and the storage soak
never writes it either (four of the seven business tables stay at zero rows
there), so the md -> row contract for agent cases was going unexercised end to
end. It is a daily-log handler like episode, but with three differences worth
pinning: it lives on the agent track (``agents/<id>/.cases/``), it has no
``sender_ids`` column, and it embeds ``task_intent`` only — ``approach`` is
BM25-indexed but deliberately never sent to the embedder.

Uses a real on-disk md file via :class:`AgentCaseWriter`; the LanceDB repo is
faked so the test stays in-memory while still checking row construction and the
diff branches.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingCapability, EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb import AgentCase
from everos.infra.persistence.markdown import AgentCaseWriter
from everos.memory.cascade.handlers import HandlerDeps
from everos.memory.cascade.handlers.agent_case import AgentCaseHandler

_AGENT = "a_ops"
_DAY = _dt.date(2026, 5, 14)
_MD = f"default_app/default_project/agents/{_AGENT}/.cases/agent_case-2026-05-14.md"


class _StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _StubEmbedder(EmbeddingProvider):
    dim = 1024

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [await self.embed(t) for t in texts]


class _FakeAgentCaseRepo:
    def __init__(self) -> None:
        self.upserts: list[list[AgentCase]] = []
        self.deletes: list[str] = []
        self.rows: list[AgentCase] = []

    async def find_where(self, where: str, *, limit: int = 100) -> list[AgentCase]:
        prefix = "md_path = '"
        if where.startswith(prefix):
            md_path = where[len(prefix) :].rstrip("'")
            return [r for r in self.rows if r.md_path == md_path]
        return []

    async def upsert(self, rows: list[AgentCase]) -> None:
        self.upserts.append(list(rows))
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
def stub_embedder(monkeypatch: pytest.MonkeyPatch) -> _StubEmbedder:
    import everos.component.embedding.accessor as acc

    embedder = _StubEmbedder()
    monkeypatch.setattr(acc, "_capability", EmbeddingCapability(provider=embedder))
    return embedder


@pytest.fixture
def no_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding unavailable — the soft-dependency path."""
    import everos.component.embedding.accessor as acc

    monkeypatch.setattr(acc, "_capability", EmbeddingCapability(provider=None))


@pytest.fixture
def fake_repo(monkeypatch: pytest.MonkeyPatch) -> _FakeAgentCaseRepo:
    repo = _FakeAgentCaseRepo()
    monkeypatch.setattr(AgentCaseHandler, "lance_repo", repo)
    return repo


async def _write_entry(
    writer: AgentCaseWriter,
    *,
    intent: str = "restart the ingest worker",
    approach: str = "drain the queue then bounce the unit",
    key_insight: str | None = "check the lock first",
    quality: float = 0.8,
) -> str:
    sections: dict[str, str] = {"TaskIntent": intent, "Approach": approach}
    if key_insight is not None:
        sections["KeyInsight"] = key_insight
    await writer.append_entry(
        _AGENT,
        inline={
            "owner_id": _AGENT,
            "session_id": "s1",
            "timestamp": "2026-05-14T10:00:00+00:00",
            "parent_type": "memcell",
            "parent_id": "mc_case_parent",
            "quality_score": quality,
        },
        sections=sections,
        date=_DAY,
    )
    return _MD


def _handler(memory_root: MemoryRoot) -> AgentCaseHandler:
    return AgentCaseHandler(
        HandlerDeps(memory_root=memory_root, tokenizer=_StubTokenizer())
    )


async def test_added_entry_builds_the_agent_track_row(
    memory_root: MemoryRoot,
    fake_repo: _FakeAgentCaseRepo,
    stub_embedder: _StubEmbedder,
) -> None:
    """One md entry -> one typed row, with the agent-track fields set."""
    await _write_entry(AgentCaseWriter(memory_root))

    outcome = await _handler(memory_root).handle_added_or_modified(_MD)

    assert (outcome.upserted, outcome.deleted, outcome.skipped) == (1, 0, 0)
    row = fake_repo.upserts[0][0]
    assert row.owner_id == _AGENT
    assert row.owner_type == "agent"
    assert row.id.startswith(f"{_AGENT}_")
    assert row.session_id == "s1"
    assert row.parent_type == "memcell"
    assert row.parent_id == "mc_case_parent"
    assert row.quality_score == pytest.approx(0.8)
    assert row.task_intent == "restart the ingest worker"
    assert row.approach == "drain the queue then bounce the unit"
    assert row.key_insight == "check the lock first"
    assert row.md_path == _MD
    assert row.content_sha256


async def test_only_task_intent_is_embedded(
    memory_root: MemoryRoot,
    fake_repo: _FakeAgentCaseRepo,
    stub_embedder: _StubEmbedder,
) -> None:
    """``approach`` is BM25-indexed but must never reach the embedder.

    The retrieval anchor for a case is its intent; the approach text is long
    step-by-step prose, so embedding it would both cost tokens and blur the
    vector. Both fields still get tokenised for keyword recall — asserting on
    the tokens as well keeps this from passing for the wrong reason (a field
    that stopped being indexed at all would also stop being embedded).
    """
    await _write_entry(AgentCaseWriter(memory_root))

    await _handler(memory_root).handle_added_or_modified(_MD)

    assert stub_embedder.calls == ["restart the ingest worker"]
    row = fake_repo.upserts[0][0]
    assert row.task_intent_tokens == "restart the ingest worker"
    assert row.approach_tokens == "drain the queue then bounce the unit"


async def test_missing_embedding_still_writes_a_searchable_row(
    memory_root: MemoryRoot,
    fake_repo: _FakeAgentCaseRepo,
    no_embedder: None,
) -> None:
    """Embedding is a soft dependency: no provider -> ``vector=None``, row kept.

    This is the tier-1 (keyword-only) deployment, so the row must still land
    with its BM25 columns populated rather than the entry being dropped.
    """
    await _write_entry(AgentCaseWriter(memory_root))

    outcome = await _handler(memory_root).handle_added_or_modified(_MD)

    assert outcome.upserted == 1
    row = fake_repo.upserts[0][0]
    assert row.vector is None
    assert row.task_intent_tokens


async def test_optional_key_insight_may_be_absent(
    memory_root: MemoryRoot,
    fake_repo: _FakeAgentCaseRepo,
    stub_embedder: _StubEmbedder,
) -> None:
    """``KeyInsight`` is optional in the md contract — absence is not an error."""
    await _write_entry(AgentCaseWriter(memory_root), key_insight=None)

    outcome = await _handler(memory_root).handle_added_or_modified(_MD)

    assert outcome.upserted == 1
    assert fake_repo.upserts[0][0].key_insight is None


async def test_unchanged_entry_is_skipped_not_re_embedded(
    memory_root: MemoryRoot,
    fake_repo: _FakeAgentCaseRepo,
    stub_embedder: _StubEmbedder,
) -> None:
    """Re-processing an untouched file must short-circuit on ``content_sha256``.

    Without this the 30s scanner sweep would re-embed every case on every pass.
    """
    await _write_entry(AgentCaseWriter(memory_root))
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(_MD)
    embeds_after_first = len(stub_embedder.calls)

    outcome = await handler.handle_added_or_modified(_MD)

    assert (outcome.upserted, outcome.skipped) == (0, 1)
    assert len(stub_embedder.calls) == embeds_after_first, "must not re-embed"


async def test_edited_entry_re_upserts(
    memory_root: MemoryRoot,
    fake_repo: _FakeAgentCaseRepo,
    stub_embedder: _StubEmbedder,
) -> None:
    """A content edit must flip the digest and re-upsert, not skip."""
    writer = AgentCaseWriter(memory_root)
    await _write_entry(writer)
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(_MD)
    first_sha = fake_repo.upserts[0][0].content_sha256

    md_file = memory_root.root / _MD
    md_file.write_text(
        md_file.read_text().replace(
            "restart the ingest worker", "restart the ingest worker twice"
        )
    )
    outcome = await handler.handle_added_or_modified(_MD)

    assert outcome.upserted == 1
    row = fake_repo.upserts[-1][0]
    assert row.task_intent == "restart the ingest worker twice"
    assert row.content_sha256 != first_sha


async def test_deleted_md_removes_every_row_for_that_path(
    memory_root: MemoryRoot,
    fake_repo: _FakeAgentCaseRepo,
    stub_embedder: _StubEmbedder,
) -> None:
    """Deleting the md must clear its rows — otherwise search serves ghosts."""
    await _write_entry(AgentCaseWriter(memory_root))
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(_MD)
    assert fake_repo.rows

    outcome = await handler.handle_deleted(_MD)

    assert outcome.deleted == 1
    assert not fake_repo.rows
