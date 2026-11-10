"""Deprecation marks reach the index with only the writes the file needs.

Pins what the daily-log projection writes to the ``deprecated_by`` column:

- Re-projecting a file the index already agrees with commits no table
  version. Every LanceDB ``update`` writes a version even when it changes
  nothing, and the projection used to issue one per entry per pass: a rebuilt
  MemLens store reached about 173k versions over three tables.
- A mark, the undo of a mark, and a mark lost to a re-embedding upsert are
  each written; entries sharing a value go in one update.
- A kind whose record has no ``deprecated_by`` column reads and writes
  nothing, and a failed read writes nothing.

The episode table is a real LanceDB table, because the version count is the
contract.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
from pathlib import Path
from typing import Any

import pytest

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.config import LanceDBSettings
from everos.infra.cascade._projections.agent_case import AgentCaseHandler
from everos.infra.cascade._projections.episode import EpisodeHandler
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.lancedb import LanceRepoBase, open_lancedb_connection
from everos.infra.persistence.lancedb._repos.episode import _EpisodeRepo
from everos.infra.persistence.markdown import EpisodeWriter
from everos.infra.persistence.records import AgentCase, Episode
from tests.fakes import TEST_DIM, arrow_schema, handler_deps

_REL = "default_app/default_project/users/u1/episodes/episode-2026-05-14.md"


class _Tokens(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _Vectors(EmbeddingProvider):
    dim = TEST_DIM

    async def embed(self, text: str) -> list[float]:
        return [0.1] * TEST_DIM

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [[0.1] * TEST_DIM for _ in texts]


@dataclasses.dataclass
class _Index:
    repo: _EpisodeRepo
    table: Any

    async def version(self) -> int:
        return await self.table.version()

    async def marks(self, entry_ids: list[str]) -> dict[str, str | None]:
        rows = await self.repo.find_by_owner_entries("u1", entry_ids, skip_vectors=True)
        return {row.entry_id: row.deprecated_by for row in rows}


@pytest.fixture(autouse=True)
def _reset_write_locks() -> None:
    LanceRepoBase._reset_locks_for_tests()


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    root = MemoryRoot(tmp_path)
    root.ensure()
    return root


@pytest.fixture
async def index(memory_root: MemoryRoot, monkeypatch: pytest.MonkeyPatch) -> _Index:
    conn = await open_lancedb_connection(memory_root.lancedb_dir, LanceDBSettings())
    table = await conn.create_table(
        "episode", schema=arrow_schema(Episode, dim=TEST_DIM)
    )
    repo = _EpisodeRepo(table=table, dim=TEST_DIM)
    monkeypatch.setattr(EpisodeHandler, "index_repo", repo)
    return _Index(repo=repo, table=table)


def _handler(root: MemoryRoot) -> EpisodeHandler:
    deps = handler_deps(memory_root=root, tokenizer=_Tokens(), embedding=_Vectors())
    return EpisodeHandler(deps)


async def _write_entries(root: MemoryRoot, n: int) -> list[str]:
    writer = EpisodeWriter(root)
    ids: list[str] = []
    for i in range(n):
        entry_id = await writer.append_entry(
            "u1",
            inline={
                "owner_id": "u1",
                "session_id": "s1",
                "timestamp": "2026-05-14T10:00:00+00:00",
                "parent_type": "memcell",
                "parent_id": f"mc_{i}",
                "sender_ids": ["u1"],
            },
            sections={"Subject": f"S{i}", "Summary": "Stub", "Content": f"body {i}"},
            date=_dt.date(2026, 5, 14),
        )
        ids.append(str(entry_id))
    return ids


async def _mark(root: MemoryRoot, marks: dict[str, str]) -> None:
    await EpisodeWriter(root).patch_frontmatter(
        root.root / _REL, {"deprecated_entries": marks}
    )


async def test_reprojecting_an_agreeing_file_commits_no_version(
    memory_root: MemoryRoot, index: _Index
) -> None:
    await _write_entries(memory_root, 3)
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(_REL)
    before = await index.version()

    await handler.handle_added_or_modified(_REL)

    assert await index.version() == before


async def test_a_mark_and_its_undo_are_one_write_each(
    memory_root: MemoryRoot, index: _Index
) -> None:
    ids = await _write_entries(memory_root, 3)
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(_REL)

    await _mark(memory_root, {ids[1]: "ep_new"})
    before = await index.version()
    await handler.handle_added_or_modified(_REL)

    assert await index.version() == before + 1
    assert await index.marks(ids) == {ids[0]: None, ids[1]: "ep_new", ids[2]: None}

    await EpisodeWriter(memory_root).drop_deprecations(
        memory_root.root / _REL, {ids[1]}
    )
    before = await index.version()
    await handler.handle_added_or_modified(_REL)

    assert await index.version() == before + 1
    assert await index.marks(ids) == dict.fromkeys(ids)


async def test_entries_sharing_a_value_go_in_one_write(
    memory_root: MemoryRoot, index: _Index
) -> None:
    ids = await _write_entries(memory_root, 4)
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(_REL)
    marks = {ids[0]: "ep_a", ids[1]: "ep_a", ids[2]: "ep_b"}
    await _mark(memory_root, marks)
    before = await index.version()

    await handler.handle_added_or_modified(_REL)

    assert await index.version() == before + 2  # one per distinct value
    assert await index.marks(ids) == {**marks, ids[3]: None}


async def test_a_mark_survives_a_reembedding_upsert(
    memory_root: MemoryRoot, index: _Index
) -> None:
    ids = await _write_entries(memory_root, 2)
    await _mark(memory_root, {ids[0]: "ep_new"})
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(_REL)
    assert (await index.marks(ids))[ids[0]] == "ep_new"
    # A stale hash sends the entry through embed + upsert, which writes the row
    # the handler builds from the entry alone, without the frontmatter's marks.
    await index.repo.update(
        {"content_sha256": "0" * 64}, where=f"entry_id = '{ids[0]}'"
    )

    await handler.handle_added_or_modified(_REL)

    assert await index.marks(ids) == {ids[0]: "ep_new", ids[1]: None}


async def test_a_kind_without_the_column_touches_nothing(
    memory_root: MemoryRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class _Repo:
        schema = AgentCase

        async def find_by_owner_entries(self, *args: Any, **kwargs: Any) -> list[Any]:
            calls.append("read")
            return []

        async def update(self, *args: Any, **kwargs: Any) -> None:
            calls.append("write")

    monkeypatch.setattr(AgentCaseHandler, "index_repo", _Repo())
    deps = handler_deps(memory_root=memory_root, tokenizer=_Tokens(), embedding=None)

    await AgentCaseHandler(deps)._propagate_deprecations(
        {"deprecated_entries": {"ac_1": "ac_2"}},
        "a1",
        "default",
        "default",
        entry_ids={"ac_1", "ac_3"},
    )

    assert calls == []


async def test_a_failed_read_writes_nothing(
    memory_root: MemoryRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes: list[Any] = []

    class _Repo:
        schema = Episode

        async def find_by_owner_entries(self, *args: Any, **kwargs: Any) -> list[Any]:
            raise RuntimeError("table busy")

        async def update(self, *args: Any, **kwargs: Any) -> None:
            writes.append(args)

    monkeypatch.setattr(EpisodeHandler, "index_repo", _Repo())

    await _handler(memory_root)._propagate_deprecations(
        {"deprecated_entries": {"ep_1": "ep_2"}},
        "u1",
        "default",
        "default",
        entry_ids={"ep_1", "ep_3"},
    )

    assert writes == []
