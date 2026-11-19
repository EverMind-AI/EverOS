"""Contract: every memory handler writes ``vector=None`` when embedding is
unavailable.

Handlers receive the embedder through :class:`HandlerDeps` and embed every
changed entry of a file in one batch. When it is ``None`` (embedding not
configured), each kind must still build
its row — BM25 tokenization and scalar
columns are populated as usual — but with ``vector`` (and, for Episode,
``subject_vector`` too) left as ``None`` rather than raising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from everos.component.tokenizer import Tokenizer
from everos.infra.cascade._projections import (
    AgentCaseHandler,
    AgentSkillHandler,
    AtomicFactHandler,
    ForesightHandler,
    HandlerDeps,
)
from everos.infra.cascade._projections._daily_log_base import ParsedEntry
from everos.infra.cascade._projections.episode import EpisodeHandler
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    AgentSkillFrontmatter,
    AgentSkillWriter,
    StructuredEntry,
)
from tests.fakes import handler_deps, index_repositories


class _StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


def _deps(memory_root: MemoryRoot, **index: Any) -> HandlerDeps:
    """No embedder: the exact condition each handler must degrade against."""
    return handler_deps(
        memory_root=memory_root,
        tokenizer=_StubTokenizer(),
        index=index_repositories(**index),
        embedding=None,
    )


def _entry(
    entry_id: str,
    inline: dict[str, str],
    sections: dict[str, str],
    *,
    sha: str = "f" * 64,
) -> ParsedEntry:
    return ParsedEntry(
        entry_id=entry_id,
        structured=StructuredEntry(
            id=entry_id,
            body="",
            start=0,
            end=0,
            header=None,
            inline=inline,
            sections=sections,
        ),
        content_sha256=sha,
    )


async def _embed_one(handler, **kw):  # type: ignore[no-untyped-def]
    """Run one entry through the batch embed path (the production route)."""
    [row] = await handler._embed_entries(
        [kw["entry"]],
        kw["owner_id"],
        kw["owner_type"],
        kw.get("app_id", "default"),
        kw.get("project_id", "default"),
        kw["md_path"],
    )
    return row


async def test_atomic_fact_handler_writes_null_vector_when_embed_unavailable(
    memory_root: MemoryRoot,
) -> None:
    handler = AtomicFactHandler(_deps(memory_root))
    row = await _embed_one(
        handler,
        owner_id="u1",
        owner_type="user",
        md_path="x.md",
        entry=_entry(
            "af_20260514_0001",
            inline={
                "owner_id": "u1",
                "session_id": "s1",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_id": "mc_1",
            },
            sections={"Fact": "the user prefers dark mode"},
        ),
    )
    assert row.vector is None
    # BM25 tokenization still runs — the row stays keyword-searchable.
    assert row.fact_tokens == "the user prefers dark mode"


async def test_foresight_handler_writes_null_vector_when_embed_unavailable(
    memory_root: MemoryRoot,
) -> None:
    handler = ForesightHandler(_deps(memory_root))
    row = await _embed_one(
        handler,
        owner_id="u1",
        owner_type="user",
        md_path="x.md",
        entry=_entry(
            "fs_20260514_0001",
            inline={
                "owner_id": "u1",
                "session_id": "s1",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_id": "mc_1",
            },
            sections={"Foresight": "user will book lunch"},
        ),
    )
    assert row.vector is None
    assert row.foresight_tokens == "user will book lunch"


async def test_agent_case_handler_writes_null_vector_when_embed_unavailable(
    memory_root: MemoryRoot,
) -> None:
    handler = AgentCaseHandler(_deps(memory_root))
    row = await _embed_one(
        handler,
        owner_id="a1",
        owner_type="agent",
        md_path="x.md",
        entry=_entry(
            "ac_20260514_0001",
            inline={
                "owner_id": "a1",
                "session_id": "s1",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_id": "mc_1",
                "quality_score": "0.5",
            },
            sections={"TaskIntent": "scan contract", "Approach": "read pages"},
        ),
    )
    assert row.vector is None
    assert row.task_intent_tokens == "scan contract"


async def test_episode_handler_writes_null_vectors_when_embed_unavailable(
    memory_root: MemoryRoot,
) -> None:
    """Episode has two vector columns — both must degrade to ``None``."""
    handler = EpisodeHandler(_deps(memory_root))
    row = await _embed_one(
        handler,
        owner_id="u1",
        owner_type="user",
        md_path="x.md",
        entry=_entry(
            "ep_20260514_0001",
            inline={
                "owner_id": "u1",
                "session_id": "s1",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_id": "mc_1",
            },
            sections={"Subject": "Test", "Summary": "Stub", "Content": "hello world"},
        ),
    )
    assert row.vector is None
    assert row.subject_vector is None
    assert row.episode_tokens == "hello world Test"


class _FakeSkillRepo:
    def __init__(self) -> None:
        self.rows: dict = {}
        self.upserts: list = []

    async def find_one_where(self, where: Any) -> Any | None:
        wanted = {c.field: c.value for c in where.children}
        row = self.rows.get(wanted["id"])
        if row is None or (row.app_id, row.project_id) != (
            wanted["app_id"],
            wanted["project_id"],
        ):
            return None
        return row

    async def upsert(self, rows) -> None:  # type: ignore[no-untyped-def]
        self.upserts.append(list(rows))
        for row in rows:
            self.rows[row.id] = row

    async def find_where(
        self, predicate: str, *, limit: int, skip_vectors: bool = False
    ):  # type: ignore[no-untyped-def]
        return []

    async def delete(self, predicate: str) -> None:  # type: ignore[no-untyped-def]
        pass

    async def delete_by_md_path(self, md_path: str) -> int:
        return 0


async def test_agent_skill_handler_writes_null_vector_when_embed_unavailable(
    memory_root: MemoryRoot,
) -> None:
    repo = _FakeSkillRepo()

    writer = AgentSkillWriter(memory_root)
    fm = AgentSkillFrontmatter(
        id="skill_contract_scan",
        agent_id="a1",
        name="contract_scan",
        description="Scan a contract draft for risk clauses.",
        confidence=0.8,
        maturity_score=0.6,
        source_case_ids=[],
    )
    await writer.write_main("a1", "contract_scan", frontmatter=fm, body="step one\n")
    md_path = (
        "default_app/default_project/agents/a1/skills/skill_contract_scan/SKILL.md"
    )

    handler = AgentSkillHandler(_deps(memory_root, agent_skill=repo))
    outcome = await handler.handle_added_or_modified(md_path)

    assert outcome.upserted == 1
    row = repo.upserts[0][0]
    assert row.vector is None
