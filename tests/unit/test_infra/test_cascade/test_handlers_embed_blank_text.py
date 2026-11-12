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

An entry can legitimately carry an empty embedded section: the md is the
source of truth and stays editable by hand, so a blank section (here a picture
memory's ``Description``) is a state the cascade has to accept from disk. The
guard lives in the shared base, so the picture memory stands in for every
daily-log kind.

Expected: the empty text is not sent to the provider at all — the row is
written with ``vector = None`` (the same soft-degrade the handler already
uses when embedding is unavailable), the remaining entries embed normally,
and the file converges. ``vector_backfill`` already treats a NULL vector as
"re-embed me later", so the row is repaired once the md is filled in.

Shape notes, because three of these assertions exist to kill a specific
wrong version of the fix:

- the blank entry sits in the **middle**, not last, so a result list that is
  not realigned with ``flat`` cannot accidentally line up;
- the non-blank descriptions have **different lengths** and ``_vec`` encodes
  length in its first component, so a swapped vector is visible rather than
  equal;
- one description is **whitespace-only** (``"   "``), because TEI accepts that and
  only a ``.strip()`` guard keeps it out — a truthiness guard looks right and
  embeds garbage.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from everos.component.embedding import EmbeddingProvider, EmbeddingServiceError
from everos.component.tokenizer import Tokenizer
from everos.infra.cascade._projections import HandlerDeps, ImageMemoryHandler
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.index import Comparison, Predicate
from everos.infra.persistence.markdown import ImageMemoryWriter
from tests.fakes import handler_deps

_DIM = 1024


def _vec(text: str) -> list[float]:
    """First component = text length, so a swap between entries is visible."""
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


class _ShortReturningEmbedder(_StrictEmbedder):
    """Returns one vector fewer than asked — a provider contract violation.

    The batch must fail loudly rather than pair vectors with the wrong texts:
    a silently short result would leave the last entry with ``vector = None``
    and every earlier one holding its neighbour's vector.
    """

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        vectors = await super().embed_batch(texts)
        return vectors[:-1]


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


class _UnstrippingImageMemoryHandler(ImageMemoryHandler):
    """A subclass whose ``_embed_texts`` hands back unstripped text.

    The guard lives in the base class precisely so a subclass cannot poison a
    batch by returning a whitespace-only string; TEI would accept ``"   "`` and
    quietly embed nothing useful. The markdown writer stores a whitespace-only
    section as an empty one, so the subclass hands the whitespace back itself.
    """

    def _embed_texts(self, entry):  # type: ignore[no-untyped-def]
        description = entry.structured.sections.get("Description", "")
        return (description or "   ",)  # no .strip()


def _deps(tmp_path, embedding: EmbeddingProvider | None) -> HandlerDeps:  # type: ignore[no-untyped-def]
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return handler_deps(memory_root=mr, tokenizer=_StubTokenizer(), embedding=embedding)


async def _write_pictures(deps: HandlerDeps, descriptions: list[str]) -> str:
    writer = ImageMemoryWriter(deps.memory_root)
    for i, description in enumerate(descriptions):
        image_id = f"{i:064x}"
        await writer.append_entry(
            "u1",
            inline={
                "owner_id": "u1",
                "session_id": "s1",
                "timestamp": f"2026-05-14T10:00:0{i}+00:00",
                "parent_type": "memcell",
                "parent_id": "mc_1",
                "image_id": image_id,
                "original_path": f"figures/{image_id}.png",
                "media_type": "png",
                "width": 640,
                "height": 480,
            },
            sections={"Description": description},
            date=_dt.date(2026, 5, 14),
        )
    return "default_app/default_project/users/u1/images/image-2026-05-14.md"


async def test_blank_section_does_not_fail_the_files_other_entries(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedder = _StrictEmbedder()
    repo = _FakeRepo()
    monkeypatch.setattr(ImageMemoryHandler, "index_repo", repo)
    deps = _deps(tmp_path, embedder)
    # blank in the MIDDLE; the two non-blank descriptions differ in length.
    rel = await _write_pictures(deps, ["aa", "", "cccc"])

    outcome = await ImageMemoryHandler(deps).handle_added_or_modified(rel)

    # Every entry is written, including the blank one.
    assert outcome.upserted == 3
    # The blank text never reached the provider; the rest went in one call.
    assert embedder.batches == [["aa", "cccc"]]
    by_description = {row.description: row for row in repo.rows}
    assert set(by_description) == {"aa", "", "cccc"}
    # Each row carries ITS OWN vector: len("aa")=2, len("cccc")=4.
    assert by_description["aa"].vector[0] == 2.0
    assert by_description["cccc"].vector[0] == 4.0
    # The blank row is written with no vector, and stays repair-able.
    assert by_description[""].vector is None


async def test_whitespace_only_text_from_a_subclass_is_not_sent(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``"   "`` is accepted by TEI, so a truthiness guard would embed it."""
    embedder = _StrictEmbedder()
    repo = _FakeRepo()
    monkeypatch.setattr(_UnstrippingImageMemoryHandler, "index_repo", repo)
    deps = _deps(tmp_path, embedder)
    rel = await _write_pictures(deps, ["aa", "   ", "cccc"])

    handler = _UnstrippingImageMemoryHandler(deps)
    outcome = await handler.handle_added_or_modified(rel)

    assert outcome.upserted == 3
    assert embedder.batches == [["aa", "cccc"]]
    vectors = {row.description.strip(): row.vector for row in repo.rows}
    assert vectors["aa"][0] == 2.0
    assert vectors["cccc"][0] == 4.0
    assert vectors[""] is None


async def test_a_wholly_blank_file_costs_no_embedding_call(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every entry blank: no request at all, and the rows are still written."""
    embedder = _StrictEmbedder()
    repo = _FakeRepo()
    monkeypatch.setattr(ImageMemoryHandler, "index_repo", repo)
    deps = _deps(tmp_path, embedder)
    rel = await _write_pictures(deps, ["", ""])

    outcome = await ImageMemoryHandler(deps).handle_added_or_modified(rel)

    assert outcome.upserted == 2
    assert embedder.batches == []
    assert all(row.vector is None for row in repo.rows)


async def test_a_short_embed_batch_fails_instead_of_misaligning(  # type: ignore[no-untyped-def]
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider returning fewer vectors must raise, not pair them up wrongly."""
    embedder = _ShortReturningEmbedder()
    repo = _FakeRepo()
    monkeypatch.setattr(ImageMemoryHandler, "index_repo", repo)
    deps = _deps(tmp_path, embedder)
    rel = await _write_pictures(deps, ["aa", "cccc"])

    with pytest.raises(ValueError):
        await ImageMemoryHandler(deps).handle_added_or_modified(rel)
    # Nothing half-written: the misaligned rows never reached the repo.
    assert repo.rows == []
