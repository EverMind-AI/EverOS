"""SQLite adapter of the knowledge catalog port, against a real system DB."""

from __future__ import annotations

import json

import pytest

from everos.infra.persistence.sqlite import SqliteKnowledgeCatalog
from everos.infra.persistence.sqlite.repos.knowledge import TopicUpsertPayload
from everos.memory.models import KnowledgeDocument
from everos.memory.ports import KnowledgeCatalog, Page
from tests.fakes.storage import Storage


@pytest.fixture(autouse=True)
async def _schema(storage: Storage) -> None:
    await storage.create_schema()


def _catalog(storage: Storage) -> SqliteKnowledgeCatalog:
    return SqliteKnowledgeCatalog(
        storage.sqlite.knowledge_document, storage.sqlite.knowledge_topic
    )


def _document(doc_id: str, *, category_id: str = "Tech") -> KnowledgeDocument:
    return KnowledgeDocument(
        doc_id=doc_id,
        app_id="app",
        project_id="proj",
        category_id=category_id,
        title=f"Title {doc_id}",
        summary="summary",
        source_name="a.pdf",
        source_type="file",
        md_path=f"app/proj/knowledge/{category_id}/t_{doc_id}/index.md",
    )


def test_adapter_satisfies_the_port(storage: Storage) -> None:
    assert isinstance(_catalog(storage), KnowledgeCatalog)


async def test_documents_round_trip(storage: Storage) -> None:
    catalog = _catalog(storage)
    await catalog.record_document(_document("d_1"))
    await catalog.record_document(_document("d_2", category_id="Science"))

    got = await catalog.get_document("d_1")

    assert got is not None
    assert got.model_dump(exclude={"created_at", "updated_at"}) == _document(
        "d_1"
    ).model_dump(exclude={"created_at", "updated_at"})
    assert got.created_at is not None
    assert await catalog.document_exists("d_2")
    assert not await catalog.document_exists("d_missing")
    assert {d.doc_id for d in await catalog.get_documents({"d_1", "d_2"})} == {
        "d_1",
        "d_2",
    }
    assert await catalog.count_by_category("app", "proj") == {"Tech": 1, "Science": 1}
    page = await catalog.list_documents(
        "app",
        "proj",
        category_id=None,
        page=Page(sort_by="title", number=1, size=1, descending=False),
    )
    assert page.total == 2 and [d.doc_id for d in page.items] == ["d_1"]


async def test_topics_decode_their_list_columns(storage: Storage) -> None:
    await storage.sqlite.knowledge_topic.upsert_from_handler(
        TopicUpsertPayload(
            node_id="d_1_1",
            doc_id="d_1",
            app_id="app",
            project_id="proj",
            category_id="Tech",
            topic_index=1,
            topic_name="Intro",
            topic_path="Doc > Intro",
            depth=1,
            parent_node_id=None,
            children_node_ids=json.dumps(["d_1_2"]),
            summary="s",
            content="body",
            content_labels=json.dumps(["safety"]),
            md_path="app/proj/knowledge/Tech/t_d_1/1_intro.md",
        )
    )
    catalog = _catalog(storage)

    (topic,) = await catalog.get_topics(["d_1_1"])

    assert topic.topic_id == "d_1_1" and topic.content == "body"
    assert topic.children_topic_ids == ["d_1_2"]
    assert topic.content_labels == ["safety"]
    assert [t.topic_id for t in await catalog.topics_of("d_1")] == ["d_1_1"]
    assert await catalog.count_topics("d_1") == 1
