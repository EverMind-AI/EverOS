"""Unit tests for ``KnowledgeTopicRecaller``.

Verifies dual-column BM25 + cosine ANN recall, using ``unittest.mock``
to patch ``get_table`` so no real LanceDB connection is needed.

White-box surfaces touched:
  - ``everos.memory.search.recall.knowledge_topic.get_table`` (patched)
  - ``KnowledgeTopicRecaller.sparse_recall`` — queries both BM25 columns
  - ``KnowledgeTopicRecaller.dense_recall`` — cosine ANN with distance→score
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from everos.component.tokenizer import Tokenizer
from everos.memory.search.recall.base import RecallerDeps
from everos.memory.search.recall.knowledge_topic import KnowledgeTopicRecaller

_MODULE = "everos.memory.search.recall.knowledge_topic"


class _WhitespaceTokenizer(Tokenizer):
    """Splits on whitespace — predictable token output for assertions."""

    def tokenize(self, text: str) -> list[str]:
        return text.split()


def _make_row(
    rid: str, *, score: float = 1.0, distance: float | None = None
) -> dict[str, Any]:
    """Build a minimal LanceDB row dict."""
    row: dict[str, Any] = {
        "id": rid,
        "app_id": "app",
        "project_id": "proj",
        "doc_id": "doc_1",
        "category_id": "cat_1",
        "topic_name": f"Topic {rid}",
        "topic_path": f"/root/{rid}",
        "depth": 1,
        "parent_node_id": "",
        "summary": f"Summary of {rid}",
        "summary_tokens": f"summary {rid}",
        "content_tokens": f"content {rid}",
        "content_labels": [],
        "md_path": f"knowledge/default/{rid}.md",
        "content_sha256": "a" * 64,
    }
    if distance is not None:
        row["_distance"] = distance
    else:
        row["_score"] = score
    return row


def _mock_bm25_table(
    summary_rows: list[dict[str, Any]],
    content_rows: list[dict[str, Any]],
) -> MagicMock:
    """Build a table mock whose BM25 results differ per column.

    The first ``nearest_to_text`` call (summary_tokens) returns
    ``summary_rows``; the second (content_tokens) returns ``content_rows``.
    ``asyncio.gather`` fires both concurrently, so we use ``side_effect``
    on the chain rather than recording call order.
    """
    summary_chain = MagicMock()
    summary_chain.where.return_value.limit.return_value.to_list = AsyncMock(
        return_value=summary_rows
    )

    content_chain = MagicMock()
    content_chain.where.return_value.limit.return_value.to_list = AsyncMock(
        return_value=content_rows
    )

    tbl = MagicMock()
    tbl.query.return_value.nearest_to_text.side_effect = [summary_chain, content_chain]
    return tbl


def _mock_ann_table(rows: list[dict[str, Any]]) -> MagicMock:
    """Build a table mock for ANN (dense) queries."""
    tbl = MagicMock()
    ann = tbl.query.return_value.nearest_to.return_value
    chain = ann.distance_type.return_value.where.return_value.limit.return_value
    chain.to_list = AsyncMock(return_value=rows)
    return tbl


@pytest.fixture()
def recaller() -> KnowledgeTopicRecaller:
    return KnowledgeTopicRecaller(RecallerDeps(tokenizer=_WhitespaceTokenizer()))


_WHERE = "app_id = 'app' AND project_id = 'proj'"


# ---------------------------------------------------------------------------
# sparse_recall — dual-column BM25
# ---------------------------------------------------------------------------


async def test_sparse_recall_queries_both_columns(
    recaller: KnowledgeTopicRecaller,
) -> None:
    """``nearest_to_text`` must be called once per BM25 column."""
    tbl = _mock_bm25_table(
        summary_rows=[_make_row("t1", score=0.9)],
        content_rows=[_make_row("t2", score=0.7)],
    )
    with patch(f"{_MODULE}.get_table", new_callable=AsyncMock, return_value=tbl):
        result = await recaller.sparse_recall("topic query", _WHERE, limit=10)

    # nearest_to_text called twice (once per column)
    assert tbl.query.return_value.nearest_to_text.call_count == 2
    ids = {c.id for c in result}
    assert ids == {"t1", "t2"}


async def test_sparse_recall_merges_by_max_score(
    recaller: KnowledgeTopicRecaller,
) -> None:
    """When the same id appears in both columns, keep the higher score."""
    shared_id = "topic_shared"
    summary_rows = [_make_row(shared_id, score=0.5)]
    content_rows = [_make_row(shared_id, score=0.9)]

    tbl = _mock_bm25_table(summary_rows, content_rows)
    with patch(f"{_MODULE}.get_table", new_callable=AsyncMock, return_value=tbl):
        result = await recaller.sparse_recall("overlap", _WHERE, limit=10)

    assert len(result) == 1
    assert result[0].id == shared_id
    assert result[0].score == pytest.approx(0.9)
    assert result[0].source == "keyword"


async def test_sparse_recall_returns_sorted_by_score(
    recaller: KnowledgeTopicRecaller,
) -> None:
    """Merged results must be sorted descending by score, truncated to limit."""
    summary_rows = [
        _make_row("a", score=0.3),
        _make_row("b", score=0.8),
    ]
    content_rows = [
        _make_row("c", score=0.6),
    ]
    tbl = _mock_bm25_table(summary_rows, content_rows)
    with patch(f"{_MODULE}.get_table", new_callable=AsyncMock, return_value=tbl):
        result = await recaller.sparse_recall("query", _WHERE, limit=2)

    assert len(result) == 2
    assert result[0].id == "b"
    assert result[1].id == "c"


async def test_sparse_recall_empty_query_returns_empty(
    recaller: KnowledgeTopicRecaller,
) -> None:
    """Empty tokenisation short-circuits — no LanceDB query is issued."""
    tok = MagicMock(spec=Tokenizer)
    tok.tokenize.return_value = []
    r = KnowledgeTopicRecaller(RecallerDeps(tokenizer=tok))

    with patch(f"{_MODULE}.get_table", new_callable=AsyncMock) as mock_gt:
        result = await r.sparse_recall("", _WHERE, limit=10)

    assert result == []
    mock_gt.assert_not_called()


# ---------------------------------------------------------------------------
# dense_recall — cosine ANN
# ---------------------------------------------------------------------------


async def test_dense_recall_cosine_conversion(
    recaller: KnowledgeTopicRecaller,
) -> None:
    """``_distance`` is converted to similarity: score = 1.0 - distance."""
    rows = [
        _make_row("t1", distance=0.2),
        _make_row("t2", distance=0.5),
    ]
    tbl = _mock_ann_table(rows)
    with patch(f"{_MODULE}.get_table", new_callable=AsyncMock, return_value=tbl):
        result = await recaller.dense_recall([0.1] * 1024, _WHERE, limit=10)

    assert len(result) == 2
    scores = {c.id: c.score for c in result}
    assert scores["t1"] == pytest.approx(0.8)
    assert scores["t2"] == pytest.approx(0.5)
    assert all(c.source == "vector" for c in result)


async def test_dense_recall_empty_vector_returns_empty(
    recaller: KnowledgeTopicRecaller,
) -> None:
    """Empty vector short-circuits — no LanceDB query is issued."""
    with patch(f"{_MODULE}.get_table", new_callable=AsyncMock) as mock_gt:
        result = await recaller.dense_recall([], _WHERE, limit=10)

    assert result == []
    mock_gt.assert_not_called()


async def test_dense_recall_metadata_excludes_noise_columns(
    recaller: KnowledgeTopicRecaller,
) -> None:
    """``vector`` and ``_distance`` must not appear in ``Candidate.metadata``."""
    row = _make_row("t1", distance=0.3)
    row["vector"] = [0.0] * 1024

    tbl = _mock_ann_table([row])
    with patch(f"{_MODULE}.get_table", new_callable=AsyncMock, return_value=tbl):
        result = await recaller.dense_recall([0.1] * 1024, _WHERE, limit=5)

    assert len(result) == 1
    assert "vector" not in result[0].metadata
    assert "_distance" not in result[0].metadata
