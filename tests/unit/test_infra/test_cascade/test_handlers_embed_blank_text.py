"""One blank section must not fail the whole daily-log file's embed batch.

`BaseDailyLogHandler._embed_entries` collects every changed entry's texts
into one ``flat`` list and sends it in a single ``embed_batch`` call. Before
this test's fix it did not separate texts whose content is empty from the
ones that carry content, and it had no per-entry isolation: the call is
all-or-nothing.

Text-embeddings-inference rejects an empty string inside the input array:

    POST /v1/embeddings {"input": ["a", ""]}
    -> 400 {"message": "Input validation error: `inputs` cannot be empty"}

so one empty section failed the embed call, the handler raised, and the row
for the *entire file* was marked failed (``EmbeddingInputError`` is not
retryable, see ``_PERMANENT_STATUSES``). Every other entry in that file —
already extracted, already on disk — never reached the index.

An atomic_fact entry can legitimately end up with an empty ``Fact`` section:
the extractor writes the entry skeleton first (md is the source of truth) and
the sentence is filled in later; a crash between the two leaves the scaffold
on disk.

Expected: the empty text is not sent to the provider at all — the row is
written with ``vector = None`` (the same soft-degrade the handler already
uses when embedding is unavailable), the remaining entries embed normally,
and the file converges. ``vector_backfill`` already treats a NULL vector as
"re-embed me later", so the row is repaired once the md is filled in.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from everos.component.embedding import EmbeddingProvider, EmbeddingServiceError
from everos.component.tokenizer import Tokenizer
from everos.infra.cascade._projections import AtomicFactHandler, HandlerDeps
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.index import Comparison, Predicate
from everos.infra.persistence.markdown import AtomicFactWriter
from tests.fakes import handler_deps

_DIM = 1024


def _vec(text: str) -> list[float]:
    return [float(len(text))] + [0.0] * (_DIM - 1)


class _StrictEmbedder(EmbeddingProvider):
    """Rejects an empty input the way TEI does (HTTP 400, not retryable)."""

    dim = _DIM

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    async def embed(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingServiceError("Input validation error: cannot be empty")
        return _vec(text)

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        texts = list(texts)
        self.batches.append(texts)
        if any(not t.strip() for t in texts):
            raise EmbeddingServiceError(
                "Input validation error: `inputs` cannot be empty"
            )
        return [_vec(t) for t in texts]


class _StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _FakeRepo:
    def __init__(self) -> None:
        self.rows: list[object] = []

    async def find_where(
        self, where: Predicate, *, limit: int = 100, skip_vectors: bool = False
    ) -> list[object]:
        if isinstance(where, Comparison) and where.field == "md_path":
            return [r for r in self.rows if r.md_path == where.value]
        return []

    async def upsert(self, rows: list[object]) -> None:
        self.rows.extend(rows)

    async def delete(self, predicate: Predicate) -> None:
        pass

    async def delete_by_md_path(self, md_path: str) -> int:
        return 0


def _deps(tmp_path, embedding: EmbeddingProvider | None) -> HandlerDeps:  # type: ignore[no-untyped-def]
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return handler_deps(memory_root=mr, tokenizer=_StubTokenizer(), embedding=embedding)


async def test_empty_fact_does_not_fail_the_files_other_entries(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder = _StrictEmbedder()
    repo = _FakeRepo()
    monkeypatch.setattr(AtomicFactHandler, "index_repo", repo)
    deps = _deps(tmp_path, embedder)

    writer = AtomicFactWriter(deps.memory_root)
    for fact in ["fact one", "fact two"]:
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
    # The scaffold the extractor leaves behind when it dies mid-write.
    await writer.append_entry(
        "u1",
        inline={
            "owner_id": "u1",
            "session_id": "s1",
            "timestamp": "2026-05-14T10:00:01+00:00",
            "parent_id": "mc_1",
            "sender_ids": ["u1"],
        },
        sections={"Fact": ""},
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

    # Two good facts land, the empty one does not poison the batch.
    assert outcome.upserted == 3
    embedded = [t for batch in embedder.batches for t in batch]
    assert embedded == ["fact one", "fact two"]
    by_fact = {row.fact: row for row in repo.rows}
    assert by_fact["fact one"].vector[0] == 8.0
    assert by_fact["fact two"].vector[0] == 8.0
    # The empty row is written with no vector, and stays repair-able.
    assert by_fact[""].vector is None
