"""Tests for :class:`PrincipleHandler` — explode one file into N rows."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.component.tokenizer import Tokenizer
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.lancedb import Principle
from everos.infra.persistence.markdown import (
    PrincipleFrontmatter,
    PrincipleItem,
    ProfileWriter,
    render_principles_body,
)
from everos.memory.cascade.handlers import HandlerDeps, PrincipleHandler


class _StubTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.split() if tok]

    def tokenize_batch(self, texts):  # type: ignore[no-untyped-def]
        return [self.tokenize(t) for t in texts]


class _FakePrincipleRepo:
    def __init__(self) -> None:
        self.rows: dict[str, Principle] = {}
        self.upserts: list[list[Principle]] = []
        self.delete_predicates: list[str] = []
        self.deletes_by_md: list[str] = []

    async def find_where(self, where: str, *, limit: int = 100) -> list[Principle]:
        md_path = where.split("md_path = '", 1)[1].rsplit("'", 1)[0]
        found = [r for r in self.rows.values() if r.md_path == md_path]
        return found[:limit]

    async def upsert(self, rows: list[Principle]) -> None:
        self.upserts.append(list(rows))
        for row in rows:
            self.rows[row.id] = row

    async def delete(self, predicate: str) -> None:
        self.delete_predicates.append(predicate)
        inside = predicate.split(" IN (", 1)[1].rstrip(")")
        ids = [tok.strip().strip("'") for tok in inside.split(",")]
        for row_id in ids:
            self.rows.pop(row_id, None)

    async def delete_by_md_path(self, md_path: str) -> int:
        self.deletes_by_md.append(md_path)
        before = len(self.rows)
        self.rows = {rid: r for rid, r in self.rows.items() if r.md_path != md_path}
        return before - len(self.rows)


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


@pytest.fixture
def fake_repo(monkeypatch: pytest.MonkeyPatch) -> _FakePrincipleRepo:
    from everos.memory.cascade.handlers import principle as pr_mod

    repo = _FakePrincipleRepo()
    monkeypatch.setattr(pr_mod, "principle_repo", repo)
    return repo


async def _write_principles(
    memory_root: MemoryRoot,
    user_id: str,
    items: list[PrincipleItem],
) -> str:
    writer = ProfileWriter(memory_root)
    fm = PrincipleFrontmatter(
        id=f"principle_{user_id}",
        user_id=user_id,
        principles=items,
    )
    await writer.write(
        user_id,
        frontmatter=fm,
        body=render_principles_body(items),
    )
    return f"default_app/default_project/users/{user_id}/principles.md"


def _handler(memory_root: MemoryRoot) -> PrincipleHandler:
    return PrincipleHandler(
        HandlerDeps(
            memory_root=memory_root,
            tokenizer=_StubTokenizer(),
        )
    )


def _item(
    principle_id: str,
    *,
    title: str = "Use Rust on device",
    statement: str = "Device Runtime uses Rust.",
    source_entry_ids: list[str] | None = None,
    timestamp_ms: int = 1_700_000_000_000,
) -> PrincipleItem:
    return PrincipleItem(
        id=principle_id,
        title=title,
        statement=statement,
        source_entry_ids=source_entry_ids or ["dc_20260517_0001"],
        timestamp_ms=timestamp_ms,
    )


async def test_first_pass_explodes_into_n_rows(
    memory_root: MemoryRoot, fake_repo: _FakePrincipleRepo
) -> None:
    md_path = await _write_principles(
        memory_root,
        "u_alice",
        [_item("pr_aaaaaaaaaaaa"), _item("pr_bbbbbbbbbbbb", title="Keep Python")],
    )
    outcome = await _handler(memory_root).handle_added_or_modified(md_path)

    assert outcome.kind == "principle"
    assert outcome.upserted == 2
    assert outcome.deleted == 0
    assert outcome.skipped == 0
    rows = fake_repo.upserts[0]
    assert {r.id for r in rows} == {
        "u_alice_pr_aaaaaaaaaaaa",
        "u_alice_pr_bbbbbbbbbbbb",
    }
    rust = next(r for r in rows if r.principle_id == "pr_aaaaaaaaaaaa")
    assert rust.owner_id == "u_alice"
    assert rust.owner_type == "user"
    assert rust.title == "Use Rust on device"
    assert rust.statement == "Device Runtime uses Rust."
    assert rust.source_entry_ids == ["dc_20260517_0001"]
    assert rust.md_path == md_path
    assert "vector" not in Principle.model_fields


async def test_second_pass_with_same_content_skips(
    memory_root: MemoryRoot, fake_repo: _FakePrincipleRepo
) -> None:
    md_path = await _write_principles(
        memory_root, "u_alice", [_item("pr_aaaaaaaaaaaa")]
    )
    handler = _handler(memory_root)
    first = await handler.handle_added_or_modified(md_path)
    assert first.upserted == 1

    second = await handler.handle_added_or_modified(md_path)
    assert second.upserted == 0
    assert second.skipped == 1
    assert second.deleted == 0
    assert len(fake_repo.upserts) == 1


async def test_timestamp_only_drift_skips(
    memory_root: MemoryRoot, fake_repo: _FakePrincipleRepo
) -> None:
    md_path = await _write_principles(
        memory_root,
        "u_alice",
        [_item("pr_aaaaaaaaaaaa", timestamp_ms=1_700_000_000_000)],
    )
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(md_path)

    absolute = memory_root.root / md_path
    absolute.write_text(
        absolute.read_text(encoding="utf-8").replace("1700000000000", "1800000000000"),
        encoding="utf-8",
    )
    outcome = await handler.handle_added_or_modified(md_path)
    assert outcome.upserted == 0
    assert outcome.skipped == 1


async def test_removed_item_deletes_row(
    memory_root: MemoryRoot, fake_repo: _FakePrincipleRepo
) -> None:
    md_path = await _write_principles(
        memory_root,
        "u_alice",
        [_item("pr_aaaaaaaaaaaa"), _item("pr_bbbbbbbbbbbb", title="Keep Python")],
    )
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(md_path)
    assert len(fake_repo.rows) == 2

    md_path = await _write_principles(
        memory_root, "u_alice", [_item("pr_aaaaaaaaaaaa")]
    )
    outcome = await handler.handle_added_or_modified(md_path)
    assert outcome.upserted == 0
    assert outcome.deleted == 1
    assert outcome.skipped == 1
    assert list(fake_repo.rows) == ["u_alice_pr_aaaaaaaaaaaa"]
    assert fake_repo.delete_predicates
    assert "u_alice_pr_bbbbbbbbbbbb" in fake_repo.delete_predicates[0]


async def test_statement_edit_triggers_upsert(
    memory_root: MemoryRoot, fake_repo: _FakePrincipleRepo
) -> None:
    md_path = await _write_principles(
        memory_root, "u_alice", [_item("pr_aaaaaaaaaaaa")]
    )
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(md_path)

    md_path = await _write_principles(
        memory_root,
        "u_alice",
        [_item("pr_aaaaaaaaaaaa", statement="Prefer Rust for device Runtime.")],
    )
    outcome = await handler.handle_added_or_modified(md_path)
    assert outcome.upserted == 1
    assert fake_repo.upserts[1][0].statement == "Prefer Rust for device Runtime."


async def test_missing_user_id_raises(
    memory_root: MemoryRoot, fake_repo: _FakePrincipleRepo
) -> None:
    bad_dir = memory_root.root / "default_app" / "default_project" / "users" / "u_x"
    bad_dir.mkdir(parents=True, exist_ok=True)
    (bad_dir / "principles.md").write_text(
        "---\nid: principle_u_x\ntype: principle\ntrack: user\nprinciples: []\n---\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="user_id"):
        await _handler(memory_root).handle_added_or_modified(
            "default_app/default_project/users/u_x/principles.md"
        )


async def test_duplicate_ids_raise(
    memory_root: MemoryRoot, fake_repo: _FakePrincipleRepo
) -> None:
    md_path = await _write_principles(
        memory_root,
        "u_alice",
        [_item("pr_aaaaaaaaaaaa"), _item("pr_aaaaaaaaaaaa", title="Dup")],
    )
    with pytest.raises(ValueError, match="duplicate id"):
        await _handler(memory_root).handle_added_or_modified(md_path)


async def test_handle_deleted_drops_all_rows(
    memory_root: MemoryRoot, fake_repo: _FakePrincipleRepo
) -> None:
    md_path = await _write_principles(
        memory_root,
        "u_alice",
        [_item("pr_aaaaaaaaaaaa"), _item("pr_bbbbbbbbbbbb", title="Keep Python")],
    )
    handler = _handler(memory_root)
    await handler.handle_added_or_modified(md_path)
    assert len(fake_repo.rows) == 2

    outcome = await handler.handle_deleted(md_path)
    assert outcome.deleted == 2
    assert fake_repo.deletes_by_md == [md_path]
    assert fake_repo.rows == {}


def test_lance_schema_has_no_vector_or_bm25() -> None:
    assert Principle.TABLE_NAME == "principle"
    assert Principle.BM25_FIELDS == []
    assert "vector" not in Principle.model_fields
