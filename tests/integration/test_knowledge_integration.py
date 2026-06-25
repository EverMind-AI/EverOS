"""End-to-end integration tests for the knowledge module.

Drives the full pipeline with real components except the embedding
provider (stubbed) and the knowledge extractor (mocked):

    create_document  ->  KnowledgeWriter  ->  md files on disk
    watchdog FSEvents ->  CascadeWatcher   ->  md_change_state
    CascadeWorker    ->  KnowledgeDocumentHandler + KnowledgeTopicHandler
                     ->  SQLite rows + LanceDB rows
    search_knowledge ->  BM25 / vector recall  ->  SearchKnowledgeResult

Validates that the cascade pipeline correctly indexes knowledge
documents for both document-level and topic-level storage, and that
search retrieval works against the indexed data.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from everalgo.types import KnowledgeMemory, ParsedContent
from sqlmodel import SQLModel

from everos.component.embedding import EmbeddingProvider
from everos.component.rerank import RerankResult
from everos.component.tokenizer import build_tokenizer
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb import (
    KnowledgeTopic,
    dispose_connection,
    ensure_business_indexes,
)
from everos.infra.persistence.lancedb.lancedb_manager import get_table
from everos.infra.persistence.sqlite import (
    DocumentUpsertPayload,
    dispose_engine,
    get_engine,
    knowledge_document_repo,
    knowledge_topic_sqlite_repo,
    md_change_state_repo,
)
from everos.memory.cascade import CascadeConfig, CascadeOrchestrator
from everos.service.knowledge import (
    ExtractionEmptyError,
    create_document,
    delete_document,
    patch_document,
    replace_document,
    search_knowledge,
)
from tests.helpers.knowledge_md import find_doc_dir, read_document_md, read_topic_mds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubEmbedder(EmbeddingProvider):
    """1024-dim deterministic vector; counts calls."""

    dim = 1024

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [float(i % 7) / 7.0 for i in range(self.dim)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class _StubReranker:
    """Deterministic reranker — returns candidates in original order."""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
    ) -> list[RerankResult]:
        return [
            RerankResult(index=i, score=1.0 - i * 0.01) for i in range(len(documents))
        ]


def _build_mock_extractor(
    memories: list[KnowledgeMemory],
) -> AsyncMock:
    """Return a mock ``KnowledgeExtractor`` whose ``aextract`` returns *memories*."""
    extractor = AsyncMock()
    extractor.aextract.return_value = memories
    return extractor


def _make_memories(
    doc_id: str,
    category_id: str = "Sports",
) -> list[KnowledgeMemory]:
    """Build a 3-node knowledge tree: root + 2 topic nodes."""
    return [
        KnowledgeMemory(
            doc_id=doc_id,
            topic_index=0,
            topic="Olympics Plan",
            summary="Overview of the 2028 Olympics plan.",
            content="",
            depth=0,
            category_id=category_id,
            topic_path="Olympics Plan",
        ),
        KnowledgeMemory(
            doc_id=doc_id,
            topic_index=1,
            topic="Budget",
            summary="Budget overview for the Games.",
            content="Total budget is $50B allocated across venues and operations.",
            depth=1,
            parent_index=0,
            children_index=[],
            topic_path="Olympics Plan > Budget",
            content_labels=["finance", "planning"],
            category_id=category_id,
        ),
        KnowledgeMemory(
            doc_id=doc_id,
            topic_index=2,
            topic="Venue",
            summary="Venue plans for the Games.",
            content="Three new stadiums will be constructed in downtown LA.",
            depth=1,
            parent_index=0,
            children_index=[],
            topic_path="Olympics Plan > Venue",
            content_labels=["infrastructure"],
            category_id=category_id,
        ),
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_lancedb_write_locks() -> None:
    """Drop per-table asyncio.Lock objects between tests."""
    from everos.core.persistence.lancedb.repository import LanceRepoBase

    LanceRepoBase._reset_locks_for_tests()


@pytest.fixture(autouse=True)
def _reset_knowledge_embedding_singleton() -> None:
    """Reset the lazy embedding and reranker singletons in service.knowledge."""
    import everos.service.knowledge as _kmod

    for attr in ("_embedding", "_reranker"):
        setattr(_kmod, attr, None)
    for attr in ("_embedding_resolved", "_reranker_resolved"):
        setattr(_kmod, attr, False)


@pytest.fixture
async def cascade_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[MemoryRoot]:
    """Boot sqlite + lancedb against a tmp memory_root; dispose at teardown."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    monkeypatch.setenv("EVEROS_EMBEDDING__MODEL", "stub-model")
    monkeypatch.setenv("EVEROS_EMBEDDING__BASE_URL", "http://stub.invalid/v1")
    monkeypatch.setenv("EVEROS_EMBEDDING__API_KEY", "stub-key")

    await dispose_connection()
    await dispose_engine()

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    await ensure_business_indexes()
    (tmp_path / "ome.toml").write_text("# test\n")

    yield MemoryRoot.default()

    await dispose_connection()
    await dispose_engine()


def _build_orchestrator(
    memory_root: MemoryRoot,
    embedder: _StubEmbedder,
    *,
    scan_interval: float = 60.0,
) -> CascadeOrchestrator:
    """Factory for a tight-polling cascade orchestrator."""
    return CascadeOrchestrator(
        memory_root=memory_root,
        embedder=embedder,
        tokenizer=build_tokenizer(),
        config=CascadeConfig(
            scan_interval_seconds=scan_interval,
            worker_batch_size=20,
            worker_max_retry=2,
            worker_poll_interval_seconds=0.05,
            worker_retry_backoff_seconds=0.0,
        ),
    )


async def _wait_drain(*, deadline: float = 20.0) -> None:
    """Poll until the cascade queue has no pending items."""
    async with asyncio.timeout(deadline):
        while True:
            summary = await md_change_state_repo.queue_summary()
            if summary.pending == 0:
                return
            await asyncio.sleep(0.05)


async def _wait_lance_rows(
    doc_id: str,
    expected: int,
    *,
    deadline: float = 20.0,
) -> None:
    """Poll until LanceDB has exactly *expected* rows for *doc_id*."""
    table = await get_table(KnowledgeTopic.TABLE_NAME, KnowledgeTopic)
    async with asyncio.timeout(deadline):
        while True:
            count = await table.count_rows(
                filter=f"doc_id = '{doc_id}'",
            )
            if count == expected:
                return
            await asyncio.sleep(0.05)


async def _create_test_document(
    memory_root: MemoryRoot,
    *,
    doc_id: str = "d_test12345678",
    category_id: str = "Sports",
    app_id: str = "default",
    project_id: str = "default",
):
    """Convenience: create a document using the standard 3-node fixture."""
    memories = _make_memories(doc_id, category_id)
    extractor = _build_mock_extractor(memories)
    knowledge_dir = memory_root.knowledge_dir(app_id, project_id)

    result = await create_document(
        extractor=extractor,
        parsed=ParsedContent(text="Full document text about the Olympics."),
        title="Olympics Plan",
        knowledge_dir=knowledge_dir,
        doc_id=doc_id,
        category_id=category_id,
    )
    return result


# ---------------------------------------------------------------------------
# A. Document Creation
# ---------------------------------------------------------------------------


async def test_create_document_end_to_end(
    cascade_runtime: MemoryRoot,
) -> None:
    """Full pipeline: create -> md -> cascade -> SQLite + LanceDB."""
    memory_root = cascade_runtime
    embedder = _StubEmbedder()
    orchestrator = _build_orchestrator(memory_root, embedder)
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        doc_id = "d_test12345678"
        result = await _create_test_document(memory_root, doc_id=doc_id)
        assert result.doc_id == doc_id
        assert result.category_id == "Sports"
        assert result.topic_count == 2

        # Wait for cascade to process all files.
        await _wait_lance_rows(doc_id, expected=2, deadline=20.0)
        await _wait_drain(deadline=20.0)

        # -- Assert md files --
        knowledge_dir = memory_root.knowledge_dir()
        doc_dir = knowledge_dir / "Sports" / f"Olympics_Plan_{doc_id}"
        assert doc_dir.is_dir()
        assert (doc_dir / "index.md").is_file()
        topic_files = sorted(f.name for f in doc_dir.iterdir() if f.name != "index.md")
        assert len(topic_files) == 2
        assert any("Budget" in f for f in topic_files)
        assert any("Venue" in f for f in topic_files)

        # -- Assert SQLite: knowledge_documents --
        doc_row = await knowledge_document_repo.get_by_doc_id(doc_id)
        assert doc_row is not None
        assert doc_row.title == "Olympics Plan"
        assert doc_row.category_id == "Sports"

        # -- Assert SQLite: knowledge_topics --
        topic_rows = await knowledge_topic_sqlite_repo.get_topics_by_doc_id(
            doc_id,
        )
        assert len(topic_rows) == 2
        topic_names = {r.topic_name for r in topic_rows}
        assert "Budget" in topic_names
        assert "Venue" in topic_names

        # -- Assert LanceDB: knowledge_topic --
        table = await get_table(KnowledgeTopic.TABLE_NAME, KnowledgeTopic)
        lance_count = await table.count_rows(
            filter=f"doc_id = '{doc_id}'",
        )
        assert lance_count == 2

        # Verify vector dimension and token fields.
        lance_rows = (
            await table.query().where(f"doc_id = '{doc_id}'").limit(10).to_list()
        )
        for row in lance_rows:
            assert len(row["vector"]) == 1024
            assert row["summary_tokens"]
            assert row["content_tokens"]

        assert embedder.calls >= 2

        # ── Truth layer (md files) verification ──
        doc_dir = find_doc_dir(memory_root.knowledge_dir(), result.doc_id)
        assert doc_dir is not None, "Document directory should exist"

        index = read_document_md(doc_dir)
        assert index["frontmatter"]["type"] == "knowledge_document"
        assert index["frontmatter"]["doc_id"] == result.doc_id
        assert index["frontmatter"]["title"] == "Olympics Plan"
        assert len(index["body"]) > 10, "Document summary should be in body"

        topics = read_topic_mds(doc_dir)
        assert len(topics) == 2, "Should have 2 topic md files (Budget + Venue)"
        for t in topics:
            fm = t["frontmatter"]
            assert fm["type"] == "knowledge_topic"
            assert fm["doc_id"] == result.doc_id
            assert fm["node_id"].startswith(result.doc_id + "_")
            assert len(t["body"]) > 10, "Topic should have content body"

    finally:
        await orchestrator.stop()


# ---------------------------------------------------------------------------
# B. Search
# ---------------------------------------------------------------------------


async def test_search_finds_ingested_topic(
    cascade_runtime: MemoryRoot,
) -> None:
    """Keyword search finds topics after cascade indexing."""
    import everos.service.knowledge as _kmod

    memory_root = cascade_runtime
    embedder = _StubEmbedder()
    _kmod._embedding = embedder
    _kmod._embedding_resolved = True
    _kmod._reranker = _StubReranker()
    _kmod._reranker_resolved = True
    orchestrator = _build_orchestrator(memory_root, embedder)
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        doc_id = "d_search001"
        await _create_test_document(memory_root, doc_id=doc_id)
        await _wait_lance_rows(doc_id, expected=2, deadline=20.0)
        await _wait_drain(deadline=20.0)

        result = await search_knowledge(
            query="budget",
            method="keyword",
            top_k=10,
        )
        assert result.hits, "Expected at least one search hit"
        budget_hits = [h for h in result.hits if "Budget" in h.topic_name]
        assert budget_hits, "Expected a hit with topic_name containing Budget"
        assert budget_hits[0].score > 0
        assert budget_hits[0].document.title == "Olympics Plan"
        assert result.took_ms > 0

    finally:
        await orchestrator.stop()


async def test_search_include_content(
    cascade_runtime: MemoryRoot,
) -> None:
    """include_content flag controls whether content is populated."""
    import everos.service.knowledge as _kmod

    memory_root = cascade_runtime
    embedder = _StubEmbedder()
    _kmod._embedding = embedder
    _kmod._embedding_resolved = True
    _kmod._reranker = _StubReranker()
    _kmod._reranker_resolved = True
    orchestrator = _build_orchestrator(memory_root, embedder)
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        doc_id = "d_content01"
        await _create_test_document(memory_root, doc_id=doc_id)
        await _wait_lance_rows(doc_id, expected=2, deadline=20.0)
        await _wait_drain(deadline=20.0)

        # Without content.
        r_no = await search_knowledge(
            query="budget",
            method="keyword",
            top_k=10,
            include_content=False,
        )
        assert r_no.hits
        assert r_no.hits[0].content is None

        # With content.
        r_yes = await search_knowledge(
            query="budget",
            method="keyword",
            top_k=10,
            include_content=True,
        )
        assert r_yes.hits
        assert r_yes.hits[0].content
        assert len(r_yes.hits[0].content) > 0

    finally:
        await orchestrator.stop()


async def test_search_score_threshold_filters(
    cascade_runtime: MemoryRoot,
) -> None:
    """score_threshold filters out low-scoring results."""
    import everos.service.knowledge as _kmod

    memory_root = cascade_runtime
    embedder = _StubEmbedder()
    _kmod._embedding = embedder
    _kmod._embedding_resolved = True
    _kmod._reranker = _StubReranker()
    _kmod._reranker_resolved = True
    orchestrator = _build_orchestrator(memory_root, embedder)
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        doc_id = "d_thresh01"
        await _create_test_document(memory_root, doc_id=doc_id)
        await _wait_lance_rows(doc_id, expected=2, deadline=20.0)
        await _wait_drain(deadline=20.0)

        r_none = await search_knowledge(
            query="budget",
            method="keyword",
            top_k=10,
            score_threshold=None,
        )
        assert r_none.hits

        r_high = await search_knowledge(
            query="budget",
            method="keyword",
            top_k=10,
            score_threshold=0.99,
        )
        assert len(r_high.hits) <= len(r_none.hits)

    finally:
        await orchestrator.stop()


async def test_search_app_project_isolation(
    cascade_runtime: MemoryRoot,
) -> None:
    """Documents in different app/project scopes are isolated in search."""
    import everos.service.knowledge as _kmod

    memory_root = cascade_runtime
    embedder = _StubEmbedder()
    _kmod._embedding = embedder
    _kmod._embedding_resolved = True
    _kmod._reranker = _StubReranker()
    _kmod._reranker_resolved = True
    orchestrator = _build_orchestrator(memory_root, embedder)
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        doc_a = "d_iso_a00001"
        doc_b = "d_iso_b00001"

        # Create doc A in (app1, proj1).
        memories_a = _make_memories(doc_a)
        ext_a = _build_mock_extractor(memories_a)
        await create_document(
            extractor=ext_a,
            parsed=ParsedContent(text="Doc A about Olympics."),
            title="Olympics Plan",
            knowledge_dir=memory_root.knowledge_dir("app1", "proj1"),
            doc_id=doc_a,
            category_id="Sports",
        )

        # Create doc B in (app2, proj2).
        memories_b = [
            KnowledgeMemory(
                doc_id=doc_b,
                topic_index=0,
                topic="Quantum Computing",
                summary="Overview of quantum computing.",
                content="",
                depth=0,
                category_id="Technology",
                topic_path="Quantum Computing",
            ),
            KnowledgeMemory(
                doc_id=doc_b,
                topic_index=1,
                topic="Qubits",
                summary="Qubit fundamentals.",
                content="A qubit is the basic unit of quantum information.",
                depth=1,
                parent_index=0,
                children_index=[],
                topic_path="Quantum Computing > Qubits",
                category_id="Technology",
            ),
        ]
        ext_b = _build_mock_extractor(memories_b)
        await create_document(
            extractor=ext_b,
            parsed=ParsedContent(text="Doc B about quantum computing."),
            title="Quantum Computing",
            knowledge_dir=memory_root.knowledge_dir("app2", "proj2"),
            doc_id=doc_b,
            category_id="Technology",
        )

        await _wait_lance_rows(doc_a, expected=2, deadline=20.0)
        await _wait_lance_rows(doc_b, expected=1, deadline=20.0)
        await _wait_drain(deadline=20.0)

        # Search in (app1, proj1) scope.
        r1 = await search_knowledge(
            query="budget stadium qubit",
            method="keyword",
            top_k=10,
            app_id="app1",
            project_id="proj1",
        )
        r1_doc_ids = {h.document.doc_id for h in r1.hits}
        if r1.hits:
            assert doc_b not in r1_doc_ids, "app1/proj1 must not see doc B"

        # Search in (app2, proj2) scope.
        r2 = await search_knowledge(
            query="budget stadium qubit",
            method="keyword",
            top_k=10,
            app_id="app2",
            project_id="proj2",
        )
        r2_doc_ids = {h.document.doc_id for h in r2.hits}
        if r2.hits:
            assert doc_a not in r2_doc_ids, "app2/proj2 must not see doc A"

    finally:
        await orchestrator.stop()


# ---------------------------------------------------------------------------
# C. Delete
# ---------------------------------------------------------------------------


async def test_delete_document_end_to_end(
    cascade_runtime: MemoryRoot,
) -> None:
    """Manual directory removal + scanner detects deletion -> cleanup.

    ``delete_document`` reports the correct topic count. We then
    manually remove the md directory and let the scanner (short
    interval) detect the missing files and cascade-delete SQLite +
    LanceDB rows.
    """
    memory_root = cascade_runtime
    embedder = _StubEmbedder()
    orchestrator = _build_orchestrator(
        memory_root,
        embedder,
        scan_interval=2.0,
    )
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        doc_id = "d_del0000001"
        await _create_test_document(memory_root, doc_id=doc_id)
        await _wait_lance_rows(doc_id, expected=2, deadline=20.0)
        await _wait_drain(deadline=20.0)

        # Confirm pre-delete state.
        assert await knowledge_document_repo.get_by_doc_id(doc_id) is not None
        topic_count_before = await knowledge_topic_sqlite_repo.count_by_doc_id(
            doc_id,
        )
        assert topic_count_before == 2

        # Service-level delete reports correct counts.
        del_result = await delete_document(
            doc_id=doc_id,
            app_id="default",
            project_id="default",
        )
        assert del_result.doc_id == doc_id
        assert del_result.deleted_topics == 2

        # Manually remove the md directory to trigger cascade cleanup.
        knowledge_dir = memory_root.knowledge_dir()
        doc_dir = knowledge_dir / "Sports" / f"Olympics_Plan_{doc_id}"
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
        assert not doc_dir.exists()

        # Wait for cascade scanner to detect + process deletions.
        # LanceDB topic rows should be cleared first.
        await _wait_lance_rows(doc_id, expected=0, deadline=20.0)
        await _wait_drain(deadline=20.0)

        # LanceDB: no topic rows.
        table = await get_table(KnowledgeTopic.TABLE_NAME, KnowledgeTopic)
        lance_count = await table.count_rows(
            filter=f"doc_id = '{doc_id}'",
        )
        assert lance_count == 0

        # SQLite: topic rows gone.
        topic_count_after = await knowledge_topic_sqlite_repo.count_by_doc_id(
            doc_id,
        )
        assert topic_count_after == 0

        # The document row may need a second scanner pass if the FK
        # constraint prevented deletion on the first attempt (index.md
        # processed before topic files).  Wait for the retry.
        async with asyncio.timeout(15.0):
            while True:
                row = await knowledge_document_repo.get_by_doc_id(doc_id)
                if row is None:
                    break
                await asyncio.sleep(0.2)
        assert await knowledge_document_repo.get_by_doc_id(doc_id) is None

    finally:
        await orchestrator.stop()


async def test_delete_idempotent(
    cascade_runtime: MemoryRoot,
) -> None:
    """Deleting an already-deleted (or nonexistent) document is a no-op."""
    memory_root = cascade_runtime
    embedder = _StubEmbedder()
    orchestrator = _build_orchestrator(
        memory_root,
        embedder,
        scan_interval=2.0,
    )
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        doc_id = "d_idemp00001"
        await _create_test_document(memory_root, doc_id=doc_id)
        await _wait_lance_rows(doc_id, expected=2, deadline=20.0)
        await _wait_drain(deadline=20.0)

        # First delete (service) + manual rmtree + cascade cleanup.
        await delete_document(
            doc_id=doc_id,
            app_id="default",
            project_id="default",
        )
        knowledge_dir = memory_root.knowledge_dir()
        doc_dir = knowledge_dir / "Sports" / f"Olympics_Plan_{doc_id}"
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
        await _wait_lance_rows(doc_id, expected=0, deadline=20.0)
        await _wait_drain(deadline=20.0)

        # Second delete: must not raise, reports 0.
        result = await delete_document(
            doc_id=doc_id,
            app_id="default",
            project_id="default",
        )
        assert result.deleted_topics == 0

    finally:
        await orchestrator.stop()


# ---------------------------------------------------------------------------
# D. Replace (PUT)
# ---------------------------------------------------------------------------


async def test_replace_document_end_to_end(
    cascade_runtime: MemoryRoot,
) -> None:
    """Replace = delete old + create new with same doc_id."""
    memory_root = cascade_runtime
    embedder = _StubEmbedder()
    orchestrator = _build_orchestrator(
        memory_root,
        embedder,
        scan_interval=2.0,
    )
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        doc_id = "d_repl000001"
        knowledge_dir = memory_root.knowledge_dir()

        # V1: 2 topics.
        await _create_test_document(memory_root, doc_id=doc_id)
        await _wait_lance_rows(doc_id, expected=2, deadline=20.0)
        await _wait_drain(deadline=20.0)

        # Delete V1 (service + manual rmtree + cascade cleanup).
        await delete_document(
            doc_id=doc_id,
            app_id="default",
            project_id="default",
        )
        doc_dir_v1 = knowledge_dir / "Sports" / f"Olympics_Plan_{doc_id}"
        if doc_dir_v1.exists():
            shutil.rmtree(doc_dir_v1)
        await _wait_lance_rows(doc_id, expected=0, deadline=20.0)
        await _wait_drain(deadline=20.0)

        # V2: 3 topics (same doc_id).
        memories_v2 = [
            KnowledgeMemory(
                doc_id=doc_id,
                topic_index=0,
                topic="Olympics Plan V2",
                summary="Updated overview.",
                content="",
                depth=0,
                category_id="Sports",
                topic_path="Olympics Plan V2",
            ),
            KnowledgeMemory(
                doc_id=doc_id,
                topic_index=1,
                topic="Budget V2",
                summary="Updated budget overview.",
                content="Revised budget is $60B.",
                depth=1,
                parent_index=0,
                children_index=[],
                topic_path="Olympics Plan V2 > Budget V2",
                category_id="Sports",
            ),
            KnowledgeMemory(
                doc_id=doc_id,
                topic_index=2,
                topic="Venue V2",
                summary="Updated venue plans.",
                content="Four stadiums now planned.",
                depth=1,
                parent_index=0,
                children_index=[],
                topic_path="Olympics Plan V2 > Venue V2",
                category_id="Sports",
            ),
            KnowledgeMemory(
                doc_id=doc_id,
                topic_index=3,
                topic="Transport",
                summary="Transport infrastructure.",
                content="New metro line connecting all venues.",
                depth=1,
                parent_index=0,
                children_index=[],
                topic_path="Olympics Plan V2 > Transport",
                category_id="Sports",
            ),
        ]
        ext_v2 = _build_mock_extractor(memories_v2)
        await create_document(
            extractor=ext_v2,
            parsed=ParsedContent(text="Updated Olympic document."),
            title="Olympics Plan V2",
            knowledge_dir=knowledge_dir,
            doc_id=doc_id,
            category_id="Sports",
        )
        await _wait_lance_rows(doc_id, expected=3, deadline=20.0)
        await _wait_drain(deadline=20.0)

        # SQLite: 1 document row, title changed.
        doc_row = await knowledge_document_repo.get_by_doc_id(doc_id)
        assert doc_row is not None
        assert doc_row.title == "Olympics Plan V2"

        # SQLite: 3 topic rows.
        topic_rows = await knowledge_topic_sqlite_repo.get_topics_by_doc_id(
            doc_id,
        )
        assert len(topic_rows) == 3

        # LanceDB: 3 rows.
        table = await get_table(KnowledgeTopic.TABLE_NAME, KnowledgeTopic)
        lance_count = await table.count_rows(
            filter=f"doc_id = '{doc_id}'",
        )
        assert lance_count == 3

    finally:
        await orchestrator.stop()


# ---------------------------------------------------------------------------
# E. Patch
# ---------------------------------------------------------------------------


async def test_patch_title_updates_metadata(
    cascade_runtime: MemoryRoot,
) -> None:
    """patch_document updates the title in SQLite without touching topics."""
    memory_root = cascade_runtime
    embedder = _StubEmbedder()
    orchestrator = _build_orchestrator(memory_root, embedder)
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        doc_id = "d_patch00001"
        await _create_test_document(memory_root, doc_id=doc_id)
        await _wait_lance_rows(doc_id, expected=2, deadline=20.0)
        await _wait_drain(deadline=20.0)

        # Patch title.
        p_result = await patch_document(
            doc_id=doc_id,
            app_id="default",
            project_id="default",
            title="New Title",
        )
        assert "title" in p_result.updated_fields

        # SQLite: title updated.
        doc_row = await knowledge_document_repo.get_by_doc_id(doc_id)
        assert doc_row is not None
        assert doc_row.title == "New Title"

        # SQLite: topic count unchanged.
        topic_count = await knowledge_topic_sqlite_repo.count_by_doc_id(doc_id)
        assert topic_count == 2

    finally:
        await orchestrator.stop()


async def test_patch_category_updates_sqlite(
    cascade_runtime: MemoryRoot,
) -> None:
    """patch_document with category_id updates the document row (SQLite-only MVP)."""
    memory_root = cascade_runtime
    embedder = _StubEmbedder()
    orchestrator = _build_orchestrator(memory_root, embedder)
    await orchestrator.start()
    await asyncio.sleep(0.3)

    try:
        doc_id = "d_patchcat01"
        await _create_test_document(memory_root, doc_id=doc_id)
        await _wait_lance_rows(doc_id, expected=2, deadline=20.0)
        await _wait_drain(deadline=20.0)

        p_result = await patch_document(
            doc_id=doc_id,
            app_id="default",
            project_id="default",
            category_id="Technology",
        )
        assert "category_id" in p_result.updated_fields

        doc_row = await knowledge_document_repo.get_by_doc_id(doc_id)
        assert doc_row is not None
        assert doc_row.category_id == "Technology"

    finally:
        await orchestrator.stop()


# ---------------------------------------------------------------------------
# F. API Errors (service-level)
# ---------------------------------------------------------------------------


async def test_get_nonexistent_document_raises(
    cascade_runtime: MemoryRoot,
) -> None:
    """get_document for a missing doc_id raises DocumentNotFoundError."""
    from everos.service.knowledge import DocumentNotFoundError, get_document

    with pytest.raises(DocumentNotFoundError):
        await get_document(
            doc_id="d_nonexistent",
            app_id="default",
            project_id="default",
        )


async def test_search_empty_query_returns_empty(
    cascade_runtime: MemoryRoot,
) -> None:
    """An empty query string returns empty results (no crash)."""
    import everos.service.knowledge as _kmod

    _kmod._embedding = _StubEmbedder()
    _kmod._embedding_resolved = True
    _kmod._reranker = _StubReranker()
    _kmod._reranker_resolved = True

    result = await search_knowledge(
        query="",
        method="keyword",
        top_k=10,
    )
    assert result.total == 0
    assert result.hits == []


async def test_patch_nonexistent_raises(
    cascade_runtime: MemoryRoot,
) -> None:
    """Patching a non-existent document raises DocumentNotFoundError."""
    from everos.service.knowledge import DocumentNotFoundError

    with pytest.raises(DocumentNotFoundError):
        await patch_document(
            doc_id="d_nonexistent",
            app_id="default",
            project_id="default",
            title="Nope",
        )


# ---------------------------------------------------------------------------
# G. Edge Cases
# ---------------------------------------------------------------------------


async def test_create_document_empty_result(
    cascade_runtime: MemoryRoot,
) -> None:
    """Extractor returning [] raises ExtractionEmptyError; no files created."""
    memory_root = cascade_runtime
    knowledge_dir = memory_root.knowledge_dir()

    extractor = _build_mock_extractor([])

    with pytest.raises(ExtractionEmptyError):
        await create_document(
            extractor=extractor,
            parsed=ParsedContent(text="Some content"),
            title="Empty Doc",
            knowledge_dir=knowledge_dir,
            doc_id="d_empty00001",
            category_id="Sports",
        )

    # No document directory should have been created for this doc.
    doc_dir = knowledge_dir / "Sports" / "Empty_Doc_d_empty00001"
    assert not doc_dir.exists()


# ---------------------------------------------------------------------------
# H. Truth-layer bug-exposing tests (xfail)
# ---------------------------------------------------------------------------


async def test_patch_title_updates_md(cascade_runtime: MemoryRoot) -> None:
    """PATCH title must update index.md frontmatter (truth layer)."""
    memory_root = cascade_runtime
    await _create_test_document(memory_root, doc_id="d_patch_title1")

    doc_dir = find_doc_dir(memory_root.knowledge_dir(), "d_patch_title1")
    assert doc_dir is not None

    old_fm = read_document_md(doc_dir)["frontmatter"]
    assert old_fm["title"] == "Olympics Plan"

    await patch_document("d_patch_title1", "default", "default", title="New Title")

    new_fm = read_document_md(doc_dir)["frontmatter"]
    assert new_fm["title"] == "New Title"


async def test_patch_category_moves_directory(cascade_runtime: MemoryRoot) -> None:
    """PATCH category_id must move the document directory to the new category."""
    memory_root = cascade_runtime
    await _create_test_document(
        memory_root, doc_id="d_patch_cat01", category_id="Sports"
    )

    knowledge_dir = memory_root.knowledge_dir()
    old_dir = find_doc_dir(knowledge_dir, "d_patch_cat01")
    assert old_dir is not None
    assert "Sports" in str(old_dir)

    await patch_document("d_patch_cat01", "default", "default", category_id="Finance")

    new_dir = find_doc_dir(knowledge_dir, "d_patch_cat01")
    assert new_dir is not None, (
        "Document directory should still exist after category change"
    )
    assert "Finance" in str(new_dir), (
        f"Directory should be under Finance/, got {new_dir}"
    )
    assert not old_dir.exists(), "Old Sports/ directory should be gone"


async def test_replace_failure_preserves_old_document(
    cascade_runtime: MemoryRoot,
) -> None:
    """replace_document restores the original md directory when extraction fails.

    White-box surfaces: md directory on disk, SQLite knowledge_documents row.
    """
    memory_root = cascade_runtime
    doc_id = "d_replace_fail"
    result = await _create_test_document(memory_root, doc_id=doc_id)

    knowledge_dir = memory_root.knowledge_dir()
    doc_dir = find_doc_dir(knowledge_dir, doc_id)
    assert doc_dir is not None, "Setup: original document directory must exist"

    # replace_document checks SQLite before proceeding; simulate cascade sync.
    await knowledge_document_repo.upsert_from_handler(
        DocumentUpsertPayload(
            doc_id=doc_id,
            app_id="default",
            project_id="default",
            category_id=result.category_id,
            title="Olympics Plan",
            summary="test",
            source_name=None,
            source_type=None,
            md_path=result.md_path,
        )
    )

    failing_extractor = AsyncMock()
    failing_extractor.aextract.return_value = []

    with pytest.raises(ExtractionEmptyError):
        await replace_document(
            extractor=failing_extractor,
            parsed=ParsedContent(text=""),
            title="Should Fail",
            doc_id=doc_id,
            knowledge_dir=knowledge_dir,
        )

    # After failure, the original md directory must be restored.
    assert doc_dir.exists(), "Original md must survive a failed PUT replacement"


async def test_dirname_collision_different_docs(cascade_runtime: MemoryRoot) -> None:
    """Two docs with titles that sanitize to the same dirname must not collide."""
    memory_root = cascade_runtime
    knowledge_dir = memory_root.knowledge_dir()

    # Both titles sanitize to "Hello_World" after stripping punctuation.
    memories1 = _make_memories("d_collision01", "Sports")
    memories1[0] = memories1[0].model_copy(update={"topic": "Hello World!"})
    ext1 = _build_mock_extractor(memories1)
    await create_document(
        extractor=ext1,
        parsed=ParsedContent(text="doc1"),
        title="Hello World!",
        knowledge_dir=knowledge_dir,
        doc_id="d_collision01",
        category_id="Sports",
    )

    memories2 = _make_memories("d_collision02", "Sports")
    memories2[0] = memories2[0].model_copy(update={"topic": "Hello World?"})
    ext2 = _build_mock_extractor(memories2)
    await create_document(
        extractor=ext2,
        parsed=ParsedContent(text="doc2"),
        title="Hello World?",
        knowledge_dir=knowledge_dir,
        doc_id="d_collision02",
        category_id="Sports",
    )

    dir1 = find_doc_dir(knowledge_dir, "d_collision01")
    dir2 = find_doc_dir(knowledge_dir, "d_collision02")
    assert dir1 is not None, "First doc directory should exist"
    assert dir2 is not None, "Second doc directory should exist"
    assert dir1 != dir2, "Different docs must have different directories"
