"""Unit tests for ``KnowledgeTopicRecaller``.

Verifies dual-column BM25 delegation + cosine ANN recall, using
``unittest.mock`` to patch the backend-neutral index repo.

White-box surfaces touched:
  - ``everos.memory.search.recall.knowledge_topic.knowledge_topic_repo`` (patched)
  - ``KnowledgeTopicRecaller.sparse_recall`` — passes both BM25 columns
  - ``KnowledgeTopicRecaller.dense_recall`` — cosine ANN with distance→score
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from everos.component.tokenizer import Tokenizer
from everos.infra.persistence.index import KnowledgeTopic, all_of, eq
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


@pytest.fixture()
def recaller() -> KnowledgeTopicRecaller:
    return KnowledgeTopicRecaller(RecallerDeps(tokenizer=_WhitespaceTokenizer()))


_WHERE = all_of(eq("app_id", "app"), eq("project_id", "proj"))


# ---------------------------------------------------------------------------
# sparse_recall — dual-column BM25
# ---------------------------------------------------------------------------


async def test_sparse_recall_queries_both_columns(
    recaller: KnowledgeTopicRecaller,
) -> None:
    """Both BM25 columns must be delegated to the index repo."""
    rows = [_make_row("t1", score=0.9), _make_row("t2", score=0.7)]
    with patch(
        f"{_MODULE}.knowledge_topic_repo.sparse_search",
        new_callable=AsyncMock,
        return_value=rows,
    ) as mock_sparse:
        result = await recaller.sparse_recall("topic query", _WHERE, limit=10)

    mock_sparse.assert_awaited_once()
    assert list(mock_sparse.await_args.args[0]) == ["topic", "query"]
    assert mock_sparse.await_args.args[1] == _WHERE
    assert mock_sparse.await_args.kwargs["columns"] == KnowledgeTopic.BM25_FIELDS
    assert mock_sparse.await_args.kwargs["limit"] == 10
    ids = {c.id for c in result}
    assert ids == {"t1", "t2"}


async def test_sparse_recall_merges_by_max_score(
    recaller: KnowledgeTopicRecaller,
) -> None:
    """Scores returned by the repo are preserved on keyword candidates."""
    shared_id = "topic_shared"

    with patch(
        f"{_MODULE}.knowledge_topic_repo.sparse_search",
        new_callable=AsyncMock,
        return_value=[_make_row(shared_id, score=0.9)],
    ):
        result = await recaller.sparse_recall("overlap", _WHERE, limit=10)

    assert len(result) == 1
    assert result[0].id == shared_id
    assert result[0].score == pytest.approx(0.9)
    assert result[0].source == "keyword"


async def test_sparse_recall_returns_sorted_by_score(
    recaller: KnowledgeTopicRecaller,
) -> None:
    """The recaller preserves repo ordering and maps rows to candidates."""
    rows = [
        _make_row("b", score=0.8),
        _make_row("c", score=0.6),
    ]
    with patch(
        f"{_MODULE}.knowledge_topic_repo.sparse_search",
        new_callable=AsyncMock,
        return_value=rows,
    ):
        result = await recaller.sparse_recall("query", _WHERE, limit=2)

    assert len(result) == 2
    assert result[0].id == "b"
    assert result[1].id == "c"


async def test_sparse_recall_empty_query_returns_empty(
    recaller: KnowledgeTopicRecaller,
) -> None:
    """Empty tokenisation short-circuits — no repo query is issued."""
    tok = MagicMock(spec=Tokenizer)
    tok.tokenize.return_value = []
    r = KnowledgeTopicRecaller(RecallerDeps(tokenizer=tok))

    with patch(
        f"{_MODULE}.knowledge_topic_repo.sparse_search",
        new_callable=AsyncMock,
    ) as mock_sparse:
        result = await r.sparse_recall("", _WHERE, limit=10)

    assert result == []
    mock_sparse.assert_not_called()


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
    with patch(
        f"{_MODULE}.knowledge_topic_repo.dense_search",
        new_callable=AsyncMock,
        return_value=rows,
    ):
        result = await recaller.dense_recall([0.1] * 1024, _WHERE, limit=10)

    assert len(result) == 2
    scores = {c.id: c.score for c in result}
    assert scores["t1"] == pytest.approx(0.8)
    assert scores["t2"] == pytest.approx(0.5)
    assert all(c.source == "vector" for c in result)


async def test_dense_recall_empty_vector_returns_empty(
    recaller: KnowledgeTopicRecaller,
) -> None:
    """Empty vector short-circuits — no repo query is issued."""
    with patch(
        f"{_MODULE}.knowledge_topic_repo.dense_search",
        new_callable=AsyncMock,
    ) as mock_dense:
        result = await recaller.dense_recall([], _WHERE, limit=10)

    assert result == []
    mock_dense.assert_not_called()


async def test_dense_recall_metadata_excludes_noise_columns(
    recaller: KnowledgeTopicRecaller,
) -> None:
    """``vector`` and ``_distance`` must not appear in ``Candidate.metadata``."""
    row = _make_row("t1", distance=0.3)
    row["vector"] = [0.0] * 1024

    with patch(
        f"{_MODULE}.knowledge_topic_repo.dense_search",
        new_callable=AsyncMock,
        return_value=[row],
    ):
        result = await recaller.dense_recall([0.1] * 1024, _WHERE, limit=5)

    assert len(result) == 1
    assert "vector" not in result[0].metadata
    assert "_distance" not in result[0].metadata
