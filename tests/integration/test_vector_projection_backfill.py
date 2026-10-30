"""Normal projections and NULL-vector repair use the same embedding sources."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import pytest
from pydantic import BaseModel

from everos.component.tokenizer import build_tokenizer
from everos.infra.cascade._projections import (
    AgentCaseHandler,
    AgentSkillHandler,
    AtomicFactHandler,
    EpisodeHandler,
    ForesightHandler,
    HandlerDeps,
    KnowledgeTopicHandler,
)
from everos.infra.cascade._projections._daily_log_base import ParsedEntry
from everos.infra.persistence.index import VECTOR_PROJECTIONS
from everos.infra.persistence.index.vector_backfill import (
    _backfill_table,
    _extract_row,
    _NullVectorRow,
    _TableBacklog,
)
from everos.infra.persistence.markdown import StructuredEntry
from everos.memory.ports import primary_key
from tests.fakes import handler_deps
from tests.fakes.storage import Storage


def _ignore_progress(table: str, done: int, total: int) -> None:
    return None


class RecordingEmbedding:
    """Deterministic vectors with input capture for projection comparisons."""

    dim = 1024

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.texts.append(text)
        return [float(sum(text.encode()) % 97)] * self.dim

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


async def _normal_row(
    kind: str, deps: HandlerDeps, *, subject: str = "episode subject"
) -> BaseModel:
    if kind == "agent_skill":
        path = deps.memory_root.root / "agents/a/skills/s/SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "---\nagent_id: a\nname: s\ndescription: useful skill\n---\nBody.\n"
        )
        await AgentSkillHandler(deps).handle_added_or_modified(
            str(path.relative_to(deps.memory_root.root))
        )
        return (await deps.index.agent_skill.scan())[0]
    if kind == "knowledge_topic":
        return await KnowledgeTopicHandler(deps)._build_lance_row(
            {
                "node_id": "n",
                "doc_id": "d",
                "category_id": "c",
                "app_id": "default",
                "project_id": "default",
                "topic_name": "topic",
                "topic_path": "topic",
                "depth": 0,
                "parent_node_id": None,
                "summary": "topic summary",
                "content": "not the embedding source",
                "content_labels": [],
            },
            "a" * 64,
            "knowledge/topic.md",
        )
    handlers = {
        "episode": EpisodeHandler,
        "atomic_fact": AtomicFactHandler,
        "foresight": ForesightHandler,
        "agent_case": AgentCaseHandler,
    }
    entry = ParsedEntry(
        entry_id="entry",
        content_sha256="a" * 64,
        structured=StructuredEntry(
            id="entry",
            body="",
            start=0,
            end=0,
            header=None,
            inline={"timestamp": "2026-10-27T00:00:00+00:00", "quality_score": "0.9"},
            sections={
                "Content": "episode body",
                "Subject": subject,
                "Fact": "atomic fact",
                "Foresight": "future plan",
                "TaskIntent": "case intent",
                "Approach": "not the embedding source",
            },
        ),
    )
    [row] = await handlers[kind](deps)._embed_entries(
        [entry], "owner", "user", "default", "default", "memory.md"
    )
    return row


@pytest.mark.parametrize("kind", tuple(VECTOR_PROJECTIONS))
async def test_normal_projection_and_repair_match(storage: Storage, kind: str) -> None:
    """Erase persisted vectors, repair them, and compare against normal indexing."""
    embedding = RecordingEmbedding()
    tokenizer = build_tokenizer()
    deps = handler_deps(
        memory_root=storage.root,
        index=storage.repos,
        embedding=embedding,
        tokenizer=tokenizer,
    )
    normal = await _normal_row(kind, deps)
    original_texts = list(embedding.texts)
    repo = getattr(storage.repos, kind)
    projection = VECTOR_PROJECTIONS[kind]
    await repo.upsert([normal])
    raw = normal.model_dump()
    key = primary_key(raw["app_id"], raw["project_id"], raw["id"])
    columns = projection.values([])
    await repo.update(columns, where=key)
    [incomplete] = await repo.scan(projection.null_filter())
    pending = _extract_row(incomplete.model_dump(), projection, tokenizer)
    assert isinstance(pending, _NullVectorRow)
    embedding.texts.clear()
    result = await _backfill_table(
        _TableBacklog(projection, repo, [pending]),
        embedding,
        prune_retention=dt.timedelta(days=7),
        on_progress=_ignore_progress,
    )
    assert result.rows_processed == 1
    assert result.rows_failed == 0
    assert embedding.texts == original_texts
    [restored] = await repo.scan(key)
    for column in columns:
        assert getattr(restored, column) == getattr(normal, column)


@pytest.mark.parametrize("subject", [" padded subject ", "   ", ""])
async def test_episode_subject_normalization_remains_source_specific(
    storage: Storage, subject: str
) -> None:
    """Keep the existing raw-index repair behavior for historical subject text."""
    embedding = RecordingEmbedding()
    tokenizer = build_tokenizer()
    deps = handler_deps(
        memory_root=storage.root,
        index=storage.repos,
        embedding=embedding,
        tokenizer=tokenizer,
    )
    normal = await _normal_row("episode", deps, subject=subject)
    assert embedding.texts == (
        ["episode body", subject.strip()] if subject.strip() else ["episode body"]
    )
    raw = normal.model_dump()
    raw["vector"] = None
    raw["subject_vector"] = None
    pending = _extract_row(raw, VECTOR_PROJECTIONS["episode"], tokenizer)
    assert isinstance(pending, _NullVectorRow)
    assert pending.subject_text == (subject or None)
    assert pending.needs_subject == bool(subject)
