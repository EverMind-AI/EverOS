"""FTS-layer normalisation contract tests.

``BaseLanceTable.ensure_fts_indexes`` builds the LanceDB FTS index with
the following configuration::

    base_tokenizer="whitespace"
    lower_case=True
    stem=True
    remove_stop_words=True
    ascii_folding=True
    language="English" (tantivy default)

The app-layer ``JiebaTokenizer`` already handles segmentation +
stopword filtering, so these FTS-layer settings act as a *belt-and-
braces* layer of normalisation. These tests probe the FTS layer
*directly* (bypassing jieba) to verify each setting actually behaves
as the docstring claims:

- lower_case=True  → query case-insensitive against the raw-cased text
- stem=True        → query for the word root hits inflected forms
- remove_stop_words=False → FTS layer does NOT drop stop-words; the
  app-layer JiebaTokenizer is the single source of truth for
  stop-word filtering (English + Chinese)
- ascii_folding=True → diacritics on Latin chars normalised (café → cafe)
- CJK pass-through → no stemming applied to CJK

Tests build a fresh in-memory-ish LanceDB store under ``tmp_path``,
declare a minimal schema with one ``body`` column, and inspect query
hits against handcrafted rows.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import ClassVar

import lancedb
import pytest
from lancedb import AsyncTable

from everos.core.persistence.lancedb import BaseLanceTable


class _FtsSpec(BaseLanceTable):
    """Minimal schema with one BM25-indexed column for FTS-layer probes."""

    TABLE_NAME: ClassVar[str] = "fts_probe"
    BM25_FIELDS: ClassVar[list[str]] = ["body"]

    id: str
    body: str


@pytest.fixture
async def fts_table(tmp_path: Path) -> AsyncIterator[AsyncTable]:
    """Build a fresh tmp LanceDB store + ``_FtsSpec`` table; index gets
    built on first ``ensure_fts_indexes`` call by each test (FTS index
    requires data first to materialise sensibly).
    """
    conn = await lancedb.connect_async(str(tmp_path / "lancedb"))
    table = await conn.create_table(_FtsSpec.TABLE_NAME, schema=_FtsSpec)
    yield table


async def _seed_and_index(table: AsyncTable, rows: list[dict]) -> None:
    """Insert rows, then (re)build the FTS index over the full table."""
    await table.add([_FtsSpec(**r) for r in rows])
    await _FtsSpec.ensure_fts_indexes(table)


async def _query_ids(table: AsyncTable, text: str) -> set[str]:
    """Run a BM25 keyword query over the ``body`` column, return matched ids."""
    rows = await table.query().nearest_to_text(text, columns="body").limit(10).to_list()
    return {r["id"] for r in rows}


# ── lower_case=True ────────────────────────────────────────────────────


async def test_lower_case_query_matches_uppercase_index(
    fts_table: AsyncTable,
) -> None:
    """Document indexed as ``HELLO`` is found by query ``hello``."""
    await _seed_and_index(
        fts_table,
        [
            {"id": "1", "body": "HELLO world"},
            {"id": "2", "body": "GOODBYE world"},
        ],
    )
    hits = await _query_ids(fts_table, "hello")
    assert hits == {"1"}


# ── stem=True ──────────────────────────────────────────────────────────


async def test_stem_query_root_matches_inflected_forms(
    fts_table: AsyncTable,
) -> None:
    """Query ``counsel`` hits documents containing ``counseling`` / ``counseled``."""
    await _seed_and_index(
        fts_table,
        [
            {"id": "1", "body": "counseling session happened"},
            {"id": "2", "body": "counseled patient yesterday"},
            {"id": "3", "body": "unrelated content"},
        ],
    )
    hits = await _query_ids(fts_table, "counsel")
    assert hits == {"1", "2"}


# ── remove_stop_words=False (app layer owns stop-words) ────────────────


async def test_fts_layer_does_not_filter_stopwords(
    fts_table: AsyncTable,
) -> None:
    """FTS layer is configured ``remove_stop_words=False`` — app layer owns it.

    The FTS index does NOT strip English stop-words. A query ``the``
    reaches BM25 unfiltered and hits a document that contains it.
    In production, :class:`JiebaTokenizer` removes ``the`` before
    tokens reach this layer; this test bypasses jieba to probe the
    FTS layer's behaviour in isolation.
    """
    await _seed_and_index(
        fts_table,
        [
            {"id": "1", "body": "the cat sat on the mat"},
            {"id": "2", "body": "unrelated body text"},
        ],
    )
    hits = await _query_ids(fts_table, "the")
    assert hits == {"1"}


# ── ascii_folding=True ─────────────────────────────────────────────────


async def test_ascii_folding_strips_diacritics(fts_table: AsyncTable) -> None:
    """``café`` is indexed/queried as ``cafe`` once diacritics are folded."""
    await _seed_and_index(
        fts_table,
        [
            {"id": "1", "body": "café latte"},
            {"id": "2", "body": "tea house"},
        ],
    )
    hits = await _query_ids(fts_table, "cafe")
    assert hits == {"1"}


# ── CJK pass-through ───────────────────────────────────────────────────


async def test_cjk_terms_pass_through_untouched(fts_table: AsyncTable) -> None:
    """CJK tokens are not stemmed or stop-word-filtered (English-only rules).

    Note: ``base_tokenizer="whitespace"`` means CJK substrings are split
    only on whitespace. The app-layer tokenizer (``JiebaTokenizer``)
    normally inserts spaces between CJK words before they reach this
    layer; here we simulate that by pre-spacing the body text.
    """
    await _seed_and_index(
        fts_table,
        [
            {"id": "1", "body": "北京 天安门"},
            {"id": "2", "body": "上海 外滩"},
        ],
    )
    hits = await _query_ids(fts_table, "北京")
    assert hits == {"1"}
