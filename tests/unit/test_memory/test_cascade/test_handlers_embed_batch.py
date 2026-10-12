"""Contract (issue #486): one cascade pass over a daily-log file embeds every
changed entry's texts in a single ``embed_batch`` call, never one ``embed``
call per entry.

The base handler collects the texts each kind declares via ``_embed_texts``,
embeds them all at once and hands each entry's vectors back to ``_build_row``.
The per-entry fan-out this replaces put N single-input requests on the
process-wide embedding semaphore and starved the interactive query embedding.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pytest

from everos.component.embedding import EmbeddingCapability, EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.core.persistence import MemoryRoot, StructuredEntry
from everos.infra.persistence.index import Comparison, Predicate
from everos.infra.persistence.markdown import AtomicFactWriter, EpisodeWriter
from everos.memory.cascade.handlers import (
    AgentCaseHandler,
    AtomicFactHandler,
    ForesightHandler,
    HandlerDeps,
)
from everos.memory.cascade.handlers._daily_log_base import ParsedEntry
from everos.memory.cascade.handlers.episode import EpisodeHandler

_DIM = 1024


def _vec(text: str) -> list[float]:
    """A vector that identifies the text it was made from (by its length)."""
    return [float(len(text))] + [0.0] * (_DIM - 1)


class _RecordingEmbedder(EmbeddingProvider):
    dim = _DIM

    def __init__(self) -> None:
        self.single: list[str] = []
        self.batches: list[list[str]] = []

    async def embed(self, text: str) -> list[float]:
        self.single.append(text)
        return _vec(text)

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        texts = list(texts)
        self.batches.append(texts)
        return [_vec(t) for t in texts]


class _StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _FakeRepo:
    def __init__(self) -> None:
        self.rows: list[Any] = []

    async def find_where(
        self, where: Predicate, *, limit: int = 100, skip_vectors: bool = False
    ) -> list[Any]:
        if isinstance(where, Comparison) and where.field == "md_path":
            return [r for r in self.rows if r.md_path == where.value]
        return []

    async def upsert(self, rows: list[Any]) -> None:
        self.rows.extend(rows)

    async def delete(self, predicate: Predicate) -> None:
        pass

    async def delete_by_md_path(self, md_path: str) -> int:
        return 0


def _set_capability(
    monkeypatch: pytest.MonkeyPatch, provider: EmbeddingProvider | None
) -> None:
    import everos.component.embedding.accessor as acc

    monkeypatch.setattr(acc, "_capability", EmbeddingCapability(provider=provider))


def _deps(tmp_path) -> HandlerDeps:  # type: ignore[no-untyped-def]
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return HandlerDeps(memory_root=mr, tokenizer=_StubTokenizer())


_INLINE = {
    "owner_id": "u1",
    "session_id": "s1",
    "timestamp": "2026-05-14T10:00:00+00:00",
    "parent_id": "mc_1",
    "quality_score": "0.5",
}


def _entry(sections: dict[str, str]) -> ParsedEntry:
    return ParsedEntry(
        entry_id="x_20260514_0001",
        structured=StructuredEntry(
            id="x_20260514_0001",
            body="",
            start=0,
            end=0,
            header=None,
            inline=_INLINE,
            sections=sections,
        ),
        content_sha256="f" * 64,
    )


@pytest.mark.parametrize(
    ("handler_cls", "sections", "expected_texts"),
    [
        (AtomicFactHandler, {"Fact": " the fact "}, ("the fact",)),
        (
            ForesightHandler,
            {"Foresight": "will book lunch", "Evidence": "said so"},
            ("will book lunch",),
        ),
        (
            AgentCaseHandler,
            {"TaskIntent": "scan contract", "Approach": "read pages"},
            ("scan contract",),
        ),
        (
            EpisodeHandler,
            {"Subject": "Sub", "Summary": "s", "Content": "hello world"},
            ("hello world", "Sub"),
        ),
        (EpisodeHandler, {"Summary": "s", "Content": "hello world"}, ("hello world",)),
    ],
)
async def test_embed_texts_and_vectors_round_trip(  # type: ignore[no-untyped-def]
    tmp_path, handler_cls, sections, expected_texts
) -> None:
    """Each kind declares exactly the texts it embeds, and the vectors the
    base hands back land on the matching row columns."""
    handler = handler_cls(_deps(tmp_path))
    entry = _entry(sections)

    assert handler._embed_texts(entry) == expected_texts

    vectors = [_vec(t) for t in expected_texts]
    row = await handler._build_row(
        owner_id="u1",
        owner_type="user",
        md_path="x.md",
        entry=entry,
        vectors=vectors,
    )
    assert list(row.vector) == vectors[0]
    if handler_cls is EpisodeHandler:
        subject_vector = vectors[1] if len(vectors) > 1 else None
        assert (
            None if row.subject_vector is None else list(row.subject_vector)
        ) == subject_vector


async def _write_episodes(
    memory_root: MemoryRoot, entries: list[tuple[str, str | None]]
) -> str:
    writer = EpisodeWriter(memory_root)
    for content, subject in entries:
        sections = {"Summary": "s", "Content": content}
        if subject is not None:
            sections["Subject"] = subject
        await writer.append_entry(
            "u1",
            inline={
                "owner_id": "u1",
                "session_id": "s1",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_type": "memcell",
                "parent_id": "mc_1",
                "sender_ids": ["u1"],
            },
            sections=sections,
            date=_dt.date(2026, 5, 14),
        )
    return "default_app/default_project/users/u1/episodes/episode-2026-05-14.md"


async def test_one_file_pass_issues_one_embed_batch(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder = _RecordingEmbedder()
    _set_capability(monkeypatch, embedder)
    repo = _FakeRepo()
    monkeypatch.setattr(EpisodeHandler, "index_repo", repo)
    deps = _deps(tmp_path)
    rel = await _write_episodes(
        deps.memory_root, [("aa", "S1"), ("bbbb", None), ("cccccc", "S3xyz")]
    )

    outcome = await EpisodeHandler(deps).handle_added_or_modified(rel)

    assert outcome.upserted == 3
    assert embedder.single == []
    assert embedder.batches == [["aa", "S1", "bbbb", "cccccc", "S3xyz"]]
    by_text = {row.episode: row for row in repo.rows}
    assert by_text["aa"].vector[0] == 2.0
    assert by_text["aa"].subject_vector[0] == 2.0
    assert by_text["bbbb"].vector[0] == 4.0
    assert by_text["bbbb"].subject_vector is None
    assert by_text["cccccc"].vector[0] == 6.0
    assert by_text["cccccc"].subject_vector[0] == 5.0


async def test_missing_provider_writes_null_vectors_through_batch_path(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_capability(monkeypatch, None)
    repo = _FakeRepo()
    monkeypatch.setattr(EpisodeHandler, "index_repo", repo)
    deps = _deps(tmp_path)
    rel = await _write_episodes(deps.memory_root, [("aa", "S1"), ("bbbb", None)])

    outcome = await EpisodeHandler(deps).handle_added_or_modified(rel)

    assert outcome.upserted == 2
    assert all(r.vector is None and r.subject_vector is None for r in repo.rows)
    assert {r.episode_tokens for r in repo.rows} == {"aa S1", "bbbb"}


async def test_atomic_facts_of_one_file_go_out_in_one_batch(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The issue's shape: ~20 atomic facts per turn, one request chunked by
    the provider instead of twenty. Second witness for the batching contract
    next to the episode test above."""
    embedder = _RecordingEmbedder()
    _set_capability(monkeypatch, embedder)
    repo = _FakeRepo()
    monkeypatch.setattr(AtomicFactHandler, "index_repo", repo)
    deps = _deps(tmp_path)
    writer = AtomicFactWriter(deps.memory_root)
    facts = [f"fact number {i}" for i in range(20)]
    for fact in facts:
        await writer.append_entry(
            "u1",
            inline={
                "owner_id": "u1",
                "session_id": "s1",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_id": "mc_1",
                "sender_ids": ["u1"],
            },
            sections={"Fact": fact},
            date=_dt.date(2026, 5, 14),
        )
    path = (
        deps.memory_root.users_dir()
        / "u1"
        / ".atomic_facts"
        / "atomic_fact-2026-05-14.md"
    )
    rel = path.relative_to(deps.memory_root.root).as_posix()

    outcome = await AtomicFactHandler(deps).handle_added_or_modified(rel)

    assert outcome.upserted == 20
    assert embedder.single == []
    assert embedder.batches == [facts]
    assert [row.vector[0] for row in repo.rows] == [float(len(f)) for f in facts]
