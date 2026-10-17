"""merge_insert keys on the declared PRIMARY_KEY, not on ``id`` alone.

Real temporary LanceDB, no fake repo: the fix lives in the merge key the
repository hands to ``merge_insert``, and a fake would model whichever key
semantics its author assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from everos.component.utils.datetime import get_utc_now
from everos.config import LanceDBSettings
from everos.core.persistence import BaseLanceTable, LanceRepoBase, MemoryRoot
from everos.core.persistence.lancedb import open_lancedb_connection
from everos.infra.persistence.index import KnowledgeTopic, primary_key
from everos.infra.persistence.lancedb import LanceIndexContract


class _Scoped(BaseLanceTable):
    TABLE_NAME: ClassVar[str] = "_scoped"
    PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("app_id", "project_id", "id")
    id: str
    app_id: str = "default"
    project_id: str = "default"
    text: str


class _ScopedRepo(LanceIndexContract[_Scoped], LanceRepoBase[_Scoped]):
    schema = _Scoped


class _Undeclared(BaseLanceTable):
    TABLE_NAME: ClassVar[str] = "_undeclared"
    id: str


class _UndeclaredRepo(LanceIndexContract[_Undeclared], LanceRepoBase[_Undeclared]):
    schema = _Undeclared


@pytest.fixture(autouse=True)
def _reset_locks() -> None:
    # pytest-asyncio makes a fresh loop per test; the class-level lock pool
    # must not carry a lock bound to the previous loop.
    LanceRepoBase._reset_locks_for_tests()


async def _repo(tmp_path: Path, schema, repo_cls):  # type: ignore[no-untyped-def]
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())
    table = await conn.create_table(schema.TABLE_NAME, schema=schema)
    return repo_cls(table=table)


async def test_same_id_in_two_projects_keeps_two_rows(tmp_path: Path) -> None:
    repo = await _repo(tmp_path, _Scoped, _ScopedRepo)
    await repo.upsert([_Scoped(id="u1_ep_1", project_id="alpha", text="rust")])
    await repo.upsert([_Scoped(id="u1_ep_1", project_id="beta", text="go")])
    assert await repo.count() == 2
    alpha = await repo.find_one_where(primary_key("default", "alpha", "u1_ep_1"))
    assert alpha is not None and alpha.text == "rust"


async def test_same_key_twice_updates_in_place(tmp_path: Path) -> None:
    repo = await _repo(tmp_path, _Scoped, _ScopedRepo)
    await repo.upsert([_Scoped(id="u1_ep_1", project_id="alpha", text="v1")])
    await repo.upsert([_Scoped(id="u1_ep_1", project_id="alpha", text="v2")])
    assert await repo.count() == 1
    row = await repo.find_one_where(primary_key("default", "alpha", "u1_ep_1"))
    assert row is not None and row.text == "v2"


async def test_upsert_rejects_undeclared_primary_key(tmp_path: Path) -> None:
    repo = await _repo(tmp_path, _Undeclared, _UndeclaredRepo)
    with pytest.raises(ValueError, match="_Undeclared has no PRIMARY_KEY"):
        await repo.upsert([_Undeclared(id="x")])


class _KnowledgeTopicRepo(
    LanceIndexContract[KnowledgeTopic], LanceRepoBase[KnowledgeTopic]
):
    schema = KnowledgeTopic


def _topic(project_id: str) -> KnowledgeTopic:
    return KnowledgeTopic(
        id="d_abc_0",
        doc_id="d_abc",
        category_id="cat",
        app_id="default",
        project_id=project_id,
        topic_name=f"topic in {project_id}",
        topic_path="root/topic",
        depth=1,
        summary=f"summary {project_id}",
        summary_tokens=f"summary {project_id}",
        content_tokens=f"content {project_id}",
        md_path=f"default_app/{project_id}/knowledge/cat/d_abc/0.md",
        content_sha256=project_id,
        created_at=get_utc_now(),
        updated_at=get_utc_now(),
    )


async def test_knowledge_topic_same_node_id_two_projects(tmp_path: Path) -> None:
    """A25: the declared key is a superset of node_id, so behaviour is unchanged
    when node_ids differ — and two spaces could even share one."""
    repo = await _repo(tmp_path, KnowledgeTopic, _KnowledgeTopicRepo)
    await repo.upsert([_topic("alpha")])
    await repo.upsert([_topic("beta")])
    assert await repo.count() == 2
    beta = await repo.find_one_where(primary_key("default", "beta", "d_abc_0"))
    assert beta is not None and beta.summary == "summary beta"
