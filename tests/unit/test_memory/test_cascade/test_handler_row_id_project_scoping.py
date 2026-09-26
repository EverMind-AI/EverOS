"""Regression test for EverOS #320 — cross-project LanceDB row-id collision.

Every daily-log cascade kind allocates ``entry_id`` per file
(``<kind>_<date>_NNNNNNNN``), so two projects with a same-date file
produce identical entry ids. Row ids were built as
``f"{owner_id}_{entry_id}"`` — identical across projects — and
``merge_insert("id")`` treated the other project's rows as matches and
replaced them: only the last-processed file per date survived.

The fix derives the row id from the file's relative path plus entry_id
(``f"{md_path}#{entry_id}"``) — unique by construction. This test
writes the same-date entry for two different projects through each
daily-log writer + handler pair and asserts both projects' rows
coexist under distinct ids. It fails on every version <= 1.2.3.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingCapability, EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    AgentCaseWriter,
    AtomicFactWriter,
    EpisodeWriter,
    ForesightWriter,
)
from everos.memory.cascade.handlers import HandlerDeps
from everos.memory.cascade.handlers.agent_case import AgentCaseHandler
from everos.memory.cascade.handlers.atomic_fact import AtomicFactHandler
from everos.memory.cascade.handlers.episode import EpisodeHandler
from everos.memory.cascade.handlers.foresight import ForesightHandler

_DATE = _dt.date(2026, 8, 27)
_OWNER = "alex"


class _StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _StubEmbedder(EmbeddingProvider):
    dim = 1024

    async def embed(self, text: str) -> list[float]:
        return [0.1] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [await self.embed(t) for t in texts]


class _FakeRepo:
    """In-memory repo with the merge-by-id semantics of merge_insert."""

    def __init__(self) -> None:
        self.rows: dict[str, object] = {}

    async def find_where(self, where: str, *, limit: int = 100) -> list:
        prefix = "md_path = '"
        if where.startswith(prefix):
            md_path = where[len(prefix) :].rstrip("'")
            return [r for r in self.rows.values() if r.md_path == md_path]
        return []

    async def upsert(self, rows: list) -> None:
        for r in rows:
            self.rows[r.id] = r  # merge_insert: same id => replace

    async def delete_by_md_path(self, md_path: str) -> int:
        before = len(self.rows)
        self.rows = {
            k: r for k, r in self.rows.items() if r.md_path != md_path
        }
        return before - len(self.rows)


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


@pytest.fixture(autouse=True)
def stub_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    import everos.component.embedding.accessor as acc

    monkeypatch.setattr(
        acc, "_capability", EmbeddingCapability(provider=_StubEmbedder())
    )


def _inline() -> dict:
    return {
        "owner_id": _OWNER,
        "session_id": "s1",
        "timestamp": "2026-08-27T10:00:00+00:00",
        "parent_type": "memcell",
        "parent_id": "mc_parent",
        "sender_ids": [_OWNER],
    }


# kind -> (writer cls, handler cls, sections, relative md path template)
_KINDS = {
    "episode": (
        EpisodeWriter,
        EpisodeHandler,
        {"Subject": "S", "Summary": "Stub", "Content": "body"},
        "dsh/{project}/users/alex/episodes/episode-2026-08-27.md",
    ),
    "atomic_fact": (
        AtomicFactWriter,
        AtomicFactHandler,
        {"Fact": "a fact"},
        "dsh/{project}/users/alex/.atomic_facts/atomic_fact-2026-08-27.md",
    ),
    "foresight": (
        ForesightWriter,
        ForesightHandler,
        {"Foresight": "a foresight"},
        "dsh/{project}/users/alex/.foresights/foresight-2026-08-27.md",
    ),
    "agent_case": (
        AgentCaseWriter,
        AgentCaseHandler,
        {"TaskIntent": "do a thing", "Approach": "carefully"},
        "dsh/{project}/agents/dsh/.cases/agent_case-2026-08-27.md",
    ),
}


@pytest.mark.parametrize("kind", sorted(_KINDS))
async def test_same_date_entries_coexist_across_projects(
    kind: str, memory_root: MemoryRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer_cls, handler_cls, sections, path_tpl = _KINDS[kind]
    repo = _FakeRepo()
    monkeypatch.setattr(handler_cls, "lance_repo", repo)

    writer = writer_cls(memory_root)
    handler = handler_cls(
        HandlerDeps(memory_root=memory_root, tokenizer=_StubTokenizer())
    )
    scope_id = _OWNER if kind != "agent_case" else "dsh"

    paths: list[str] = []
    for project in ("proj-a", "proj-b"):
        await writer.append_entry(
            scope_id,
            inline=_inline() if kind != "agent_case" else {**_inline(), "agent_id": "dsh", "quality_score": "0.9"},
            sections=sections,
            date=_DATE,
            app_id="dsh",
            project_id=project,
        )
        md_path = path_tpl.format(project=project)
        paths.append(md_path)
        await handler.handle_added_or_modified(md_path)

    # Both projects' rows must coexist (pre-fix: project B's upsert
    # matched project A's row id and replaced it — 1 row, proj-b's).
    rows_a = await repo.find_where(f"md_path = '{paths[0]}'")
    rows_b = await repo.find_where(f"md_path = '{paths[1]}'")
    assert len(rows_a) == 1, (
        f"{kind}: project A's row was replaced by project B's "
        f"(ids collide across projects — EverOS #320)"
    )
    assert len(rows_b) == 1
    assert rows_a[0].id != rows_b[0].id
    assert rows_a[0].project_id == "proj-a"
    assert rows_b[0].project_id == "proj-b"
