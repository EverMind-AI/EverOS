"""Tests for :class:`LanceRepoBase` + :class:`LanceDailyLogRepoBase`.

Exercises the chassis-level query helpers shared by every business
LanceDB repo: ``find_where`` / ``find_one_where`` / ``find_by_owner`` /
``find_by_md_path`` (on :class:`LanceRepoBase`), and the daily-log
slice ``find_by_owner_entry`` / ``find_by_session`` /
``find_by_parent`` (on :class:`LanceDailyLogRepoBase`). Also covers
``get_by_id`` + ``upsert`` so the chassis CRUD surface is end-to-end
verified.

Uses a tmp LanceDB connection + a locally-defined daily-log-shaped
table so the chassis can be exercised without depending on any
specific business schema (episode / atomic_fact / …).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from everos.config import LanceDBSettings
from everos.core.persistence import (
    BaseLanceTable,
    MemoryRoot,
    Vector,
    open_lancedb_connection,
)
from everos.core.persistence.lancedb import (
    LanceDailyLogRepoBase,
    LanceRepoBase,
)


class _Note(BaseLanceTable):
    """Minimal daily-log-shaped table for chassis tests."""

    TABLE_NAME: ClassVar[str] = "_note"

    id: str
    owner_id: str
    app_id: str = "default"
    project_id: str = "default"
    entry_id: str
    session_id: str
    parent_type: str
    parent_id: str
    md_path: str
    text: str
    vector: Vector(4)  # type: ignore[valid-type]


class _SearchNote(BaseLanceTable):
    """Schema with BM25_FIELDS declared — exercises FTS index setup."""

    TABLE_NAME: ClassVar[str] = "_search_note"
    BM25_FIELDS: ClassVar[list[str]] = ["tokens"]

    id: str
    text: str
    """Original surface form (display)."""

    tokens: str
    """Space-joined pre-tokenised text (BM25 index target)."""

    vector: Vector(4)  # type: ignore[valid-type]


class _NoteRepo(LanceDailyLogRepoBase[_Note]):
    schema = _Note


def _row(
    *,
    owner: str,
    entry: str,
    session: str = "sess_a",
    parent_type: str = "memcell",
    parent_id: str = "mc_1",
    md_path: str | None = None,
    text: str = "x",
) -> _Note:
    return _Note(
        id=f"{owner}_{entry}",
        owner_id=owner,
        entry_id=entry,
        session_id=session,
        parent_type=parent_type,
        parent_id=parent_id,
        md_path=md_path or f"users/{owner}/notes/{entry}.md",
        text=text,
        vector=[1.0, 0.0, 0.0, 0.0],
    )


@pytest.fixture(autouse=True)
def _reset_write_locks() -> None:
    """Drop the per-table write-lock pool between tests.

    ``LanceRepoBase`` lazily creates an ``asyncio.Lock`` per table name
    and stashes it in a class-level dict; without a reset the lock
    object outlives the pytest-asyncio function-scoped event loop and
    the next test fails with "bound to a different event loop".
    """
    LanceRepoBase._reset_locks_for_tests()


@pytest.fixture
async def repo(tmp_path: Path) -> _NoteRepo:
    """Open a tmp connection, create the ``_note`` table, return a repo."""
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())
    table = await conn.create_table("_note", schema=_Note)
    return _NoteRepo(table=table)


# ── add + get_by_id + count ──────────────────────────────────────────────


async def test_add_and_count(repo: _NoteRepo) -> None:
    await repo.add([_row(owner="u1", entry="ep_1"), _row(owner="u1", entry="ep_2")])
    assert await repo.count() == 2


async def test_get_by_id_returns_typed_instance(repo: _NoteRepo) -> None:
    await repo.add([_row(owner="u1", entry="ep_1", text="hello")])
    got = await repo.get_by_id("u1_ep_1")
    assert got is not None
    assert isinstance(got, _Note)
    assert got.text == "hello"


async def test_get_by_id_returns_none_when_missing(repo: _NoteRepo) -> None:
    assert await repo.get_by_id("ghost") is None


# ── upsert ──────────────────────────────────────────────────────────────


async def test_upsert_inserts_on_new(repo: _NoteRepo) -> None:
    await repo.upsert([_row(owner="u1", entry="ep_1", text="v1")])
    got = await repo.get_by_id("u1_ep_1")
    assert got is not None
    assert got.text == "v1"


async def test_upsert_updates_on_existing(repo: _NoteRepo) -> None:
    await repo.add([_row(owner="u1", entry="ep_1", text="v1")])
    await repo.upsert([_row(owner="u1", entry="ep_1", text="v2")])
    got = await repo.get_by_id("u1_ep_1")
    assert got is not None
    assert got.text == "v2"
    assert await repo.count() == 1  # update, not append


# ── find_where / find_one_where ─────────────────────────────────────────


async def test_find_where_returns_typed_list(repo: _NoteRepo) -> None:
    await repo.add(
        [
            _row(owner="u1", entry="ep_1"),
            _row(owner="u1", entry="ep_2"),
            _row(owner="u2", entry="ep_3"),
        ]
    )
    rows = await repo.find_where("owner_id = 'u1'")
    assert len(rows) == 2
    assert all(isinstance(r, _Note) for r in rows)
    assert {r.entry_id for r in rows} == {"ep_1", "ep_2"}


async def test_find_one_where_returns_first_match(repo: _NoteRepo) -> None:
    await repo.add([_row(owner="u1", entry="ep_1")])
    got = await repo.find_one_where("entry_id = 'ep_1'")
    assert got is not None
    assert got.entry_id == "ep_1"


async def test_find_one_where_returns_none(repo: _NoteRepo) -> None:
    assert await repo.find_one_where("entry_id = 'ghost'") is None


# ── find_where_paginated ────────────────────────────────────────────────


async def test_find_where_paginated_first_page(repo: _NoteRepo) -> None:
    """5 rows, page=1 size=2 → 2 rows, total=5, sorted DESC by entry_id."""
    await repo.add(
        [_row(owner="u1", entry=f"ep_{i}") for i in range(1, 6)],
    )
    rows, total = await repo.find_where_paginated(
        "owner_id = 'u1'",
        sort_by="entry_id",
        descending=True,
        page=1,
        page_size=2,
    )
    assert total == 5
    assert [r.entry_id for r in rows] == ["ep_5", "ep_4"]


async def test_find_where_paginated_last_page_partial(repo: _NoteRepo) -> None:
    """5 rows, page=3 size=2 → 1 row (the tail)."""
    await repo.add(
        [_row(owner="u1", entry=f"ep_{i}") for i in range(1, 6)],
    )
    rows, total = await repo.find_where_paginated(
        "owner_id = 'u1'",
        sort_by="entry_id",
        descending=True,
        page=3,
        page_size=2,
    )
    assert total == 5
    assert [r.entry_id for r in rows] == ["ep_1"]


async def test_find_where_paginated_ascending_sort(repo: _NoteRepo) -> None:
    """``descending=False`` flips order."""
    await repo.add(
        [_row(owner="u1", entry=f"ep_{i}") for i in range(1, 4)],
    )
    rows, total = await repo.find_where_paginated(
        "owner_id = 'u1'",
        sort_by="entry_id",
        descending=False,
        page=1,
        page_size=10,
    )
    assert total == 3
    assert [r.entry_id for r in rows] == ["ep_1", "ep_2", "ep_3"]


async def test_find_where_paginated_empty_predicate(repo: _NoteRepo) -> None:
    """Predicate that matches nothing → empty list + total=0."""
    rows, total = await repo.find_where_paginated(
        "owner_id = 'ghost'",
        sort_by="entry_id",
        page=1,
        page_size=20,
    )
    assert rows == []
    assert total == 0


async def test_find_where_paginated_filters_by_owner(repo: _NoteRepo) -> None:
    """Total is the predicate's true count, not the table's row count."""
    await repo.add(
        [
            _row(owner="u1", entry="ep_1"),
            _row(owner="u1", entry="ep_2"),
            _row(owner="u2", entry="ep_3"),
        ]
    )
    rows, total = await repo.find_where_paginated(
        "owner_id = 'u1'",
        sort_by="entry_id",
        page=1,
        page_size=10,
    )
    assert total == 2
    assert {r.entry_id for r in rows} == {"ep_1", "ep_2"}


async def test_find_where_paginated_truncates_above_max_fetch(
    repo: _NoteRepo,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When total > max_fetch the chassis warns and returns a prefix sort.

    Correctness contract: ``total`` is still the *true* row count from
    ``count_rows(filter=...)``, but the page contents are taken from
    only the first ``max_fetch`` rows the engine scanned. structlog now
    routes through stdlib's root logger (see
    ``core/observability/logging/factory.py``), so the standard
    ``caplog`` fixture is the right way to assert on the warning.
    """
    # Unit tests don't go through the CLI entry, so the structlog →
    # stdlib bridge is uninitialised — wire it up here so ``caplog``
    # can observe the warning.
    from everos.core.observability.logging import configure_logging

    configure_logging(level="WARNING")

    await repo.add(
        [_row(owner="u1", entry=f"ep_{i:03d}") for i in range(1, 11)],
    )
    with caplog.at_level("WARNING"):
        rows, total = await repo.find_where_paginated(
            "owner_id = 'u1'",
            sort_by="entry_id",
            page=1,
            page_size=3,
            max_fetch=5,
        )
    assert total == 10  # true match count
    assert len(rows) == 3
    assert "find_where_paginated truncated" in caplog.text


# ── 5-table shared: find_by_owner / find_by_md_path ─────────────────────


async def test_find_by_owner(repo: _NoteRepo) -> None:
    await repo.add(
        [
            _row(owner="u1", entry="ep_1"),
            _row(owner="u1", entry="ep_2"),
            _row(owner="u2", entry="ep_3"),
        ]
    )
    rows = await repo.find_by_owner("u1")
    assert {r.entry_id for r in rows} == {"ep_1", "ep_2"}


async def test_find_by_md_path_round_trip(repo: _NoteRepo) -> None:
    path = "users/u1/notes/ep_1.md"
    await repo.add([_row(owner="u1", entry="ep_1", md_path=path)])
    got = await repo.find_by_md_path(path)
    assert got is not None
    assert got.entry_id == "ep_1"


async def test_find_by_md_path_returns_none_when_missing(repo: _NoteRepo) -> None:
    assert await repo.find_by_md_path("users/u1/notes/ghost.md") is None


# ── daily-log: find_by_owner_entry / find_by_session / find_by_parent ───


async def test_find_by_owner_entry(repo: _NoteRepo) -> None:
    await repo.add([_row(owner="u1", entry="ep_7")])
    got = await repo.find_by_owner_entry("u1", "ep_7")
    assert got is not None
    assert got.entry_id == "ep_7"


async def test_find_by_owner_entry_returns_none_when_missing(
    repo: _NoteRepo,
) -> None:
    assert await repo.find_by_owner_entry("u1", "ghost") is None


async def test_find_by_owner_entries_returns_only_matching_rows(
    repo: _NoteRepo,
) -> None:
    """Bulk lookup keeps only rows whose ``entry_id`` is in the set."""
    await repo.add(
        [
            _row(owner="u1", entry="ep_1"),
            _row(owner="u1", entry="ep_2"),
            _row(owner="u1", entry="ep_3"),
            _row(owner="u2", entry="ep_1"),  # different owner — must not leak
        ]
    )
    rows = await repo.find_by_owner_entries("u1", ["ep_1", "ep_3"])
    assert {r.entry_id for r in rows} == {"ep_1", "ep_3"}
    assert all(r.owner_id == "u1" for r in rows)


async def test_find_by_owner_entries_empty_input_short_circuits(
    repo: _NoteRepo,
) -> None:
    """No ids → ``[]`` without emitting a ``WHERE entry_id IN ()`` predicate."""
    await repo.add([_row(owner="u1", entry="ep_1")])
    assert await repo.find_by_owner_entries("u1", []) == []


async def test_find_by_session(repo: _NoteRepo) -> None:
    await repo.add(
        [
            _row(owner="u1", entry="ep_1", session="sess_a"),
            _row(owner="u1", entry="ep_2", session="sess_a"),
            _row(owner="u1", entry="ep_3", session="sess_b"),
        ]
    )
    rows = await repo.find_by_session("u1", "sess_a")
    assert {r.entry_id for r in rows} == {"ep_1", "ep_2"}


async def test_find_by_parent(repo: _NoteRepo) -> None:
    await repo.add(
        [
            _row(owner="u1", entry="ep_1", parent_type="memcell", parent_id="mc_x"),
            _row(owner="u1", entry="ep_2", parent_type="memcell", parent_id="mc_x"),
            _row(owner="u1", entry="ep_3", parent_type="other", parent_id="mc_y"),
        ]
    )
    rows = await repo.find_by_parent("memcell", "mc_x")
    assert {r.entry_id for r in rows} == {"ep_1", "ep_2"}


# ── chassis fallback behaviour ──────────────────────────────────────────


async def test_table_lookup_not_implemented_when_no_override() -> None:
    """Repo with neither ``table=`` injection nor ``_table_lookup`` raises."""

    class _BareRepo(LanceRepoBase[_Note]):
        schema = _Note

    bare = _BareRepo()
    with pytest.raises(NotImplementedError, match="_table_lookup"):
        await bare.count()


async def test_table_name_derived_from_schema() -> None:
    """``repo.table_name`` reads off ``schema.TABLE_NAME`` (single source of truth)."""

    class _R(LanceRepoBase[_Note]):
        schema = _Note

    assert _R().table_name == "_note"  # equals _Note.TABLE_NAME


# ── SQL-quote escape defence ────────────────────────────────────────────


# ── BaseLanceTable.ensure_fts_indexes ───────────────────────────────────


async def test_ensure_fts_indexes_creates_index(tmp_path: Path) -> None:
    """Declared ``BM25_FIELDS`` becomes an FTS index after ensure."""
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())
    table = await conn.create_table("_search_note", schema=_SearchNote)
    await table.add(
        [
            _SearchNote(
                id="1",
                text="hello world",
                tokens="hello world",
                vector=[1, 0, 0, 0],
            )
        ]
    )

    await _SearchNote.ensure_fts_indexes(table)

    indices = await table.list_indices()
    indexed_cols = {col for idx in indices for col in (idx.columns or [])}
    assert "tokens" in indexed_cols
    conn.close()


async def test_ensure_fts_indexes_is_idempotent(tmp_path: Path) -> None:
    """Calling twice is safe — no error, no duplicate index."""
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())
    table = await conn.create_table("_search_note", schema=_SearchNote)
    await table.add([_SearchNote(id="1", text="hi", tokens="hi", vector=[1, 0, 0, 0])])

    await _SearchNote.ensure_fts_indexes(table)
    first = await table.list_indices()
    await _SearchNote.ensure_fts_indexes(table)
    second = await table.list_indices()

    assert len(first) == len(second)
    conn.close()


async def test_ensure_fts_indexes_noop_when_no_fields_declared(
    repo: _NoteRepo,
) -> None:
    """Schema without ``BM25_FIELDS`` is a no-op (no error)."""
    table = await repo._table()
    # _Note declares no BM25_FIELDS — calling the classmethod is a no-op.
    await _Note.ensure_fts_indexes(table)
    indices = await table.list_indices()
    # No FTS index was created; vector/scalar may exist by default but we
    # only assert no error path triggered.
    assert isinstance(indices, list) or hasattr(indices, "__iter__")


# ── SQL-quote escape defence ────────────────────────────────────────────


# ── delete_by_md_path ───────────────────────────────────────────────────


async def test_delete_by_md_path_removes_matching_row(repo: _NoteRepo) -> None:
    """Cascade md-deleted flow: rows for a path are wiped, count returned."""
    target = "users/u1/notes/ep_1.md"
    await repo.add(
        [
            _row(owner="u1", entry="ep_1", md_path=target),
            _row(owner="u1", entry="ep_2"),
        ]
    )
    deleted = await repo.delete_by_md_path(target)
    assert deleted == 1
    assert await repo.find_by_md_path(target) is None
    assert await repo.count() == 1  # the other row survived


async def test_delete_by_md_path_returns_zero_when_no_match(
    repo: _NoteRepo,
) -> None:
    await repo.add([_row(owner="u1", entry="ep_1")])
    assert await repo.delete_by_md_path("users/u1/notes/ghost.md") == 0
    assert await repo.count() == 1


async def test_delete_by_md_path_removes_multiple_entries_one_file(
    repo: _NoteRepo,
) -> None:
    """A daily-log md holds many entries → all rows for the path go."""
    shared = "users/u1/notes/episode-2026-05-12.md"
    await repo.add(
        [
            _row(owner="u1", entry="ep_1", md_path=shared),
            _row(owner="u1", entry="ep_2", md_path=shared),
            _row(owner="u1", entry="ep_3", md_path=shared),
            _row(owner="u2", entry="ep_4"),  # different path, untouched
        ]
    )
    deleted = await repo.delete_by_md_path(shared)
    assert deleted == 3
    assert await repo.count() == 1


async def test_delete_by_md_path_escapes_single_quotes(
    repo: _NoteRepo,
) -> None:
    """A path containing a single quote does not break the predicate."""
    tricky = "users/u1/notes/it's.md"
    await repo.add([_row(owner="u1", entry="ep_1", md_path=tricky)])
    assert await repo.delete_by_md_path(tricky) == 1


# ── SQL-quote escape defence (kept) ─────────────────────────────────────


async def test_get_by_id_escapes_single_quotes(repo: _NoteRepo) -> None:
    """An id containing a single quote does not break the predicate."""
    quoted_id = "u1_it's_fine"
    await repo.add(
        [
            _Note(
                id=quoted_id,
                owner_id="u1",
                entry_id="it's_fine",
                session_id="s",
                parent_type="memcell",
                parent_id="mc_1",
                md_path="x",
                text="t",
                vector=[1.0, 0.0, 0.0, 0.0],
            )
        ]
    )
    got = await repo.get_by_id(quoted_id)
    assert got is not None
    assert got.entry_id == "it's_fine"


# ── Concurrency: per-table write lock ───────────────────────────────────


async def test_concurrent_upsert_disjoint_ids_no_lost_update(
    repo: _NoteRepo,
) -> None:
    """Regression for Bug B: cascade ``asyncio.gather`` over rows of the
    same kind would race on ``merge_insert`` and drop a write (observed
    on ``user_profile`` — pk = owner_id, two disjoint INSERTs ending up
    with only one row in LanceDB). The per-table ``asyncio.Lock`` in
    :meth:`LanceRepoBase.upsert` must serialise those writes so every
    submitted row lands.
    """
    n = 16
    rows = [_row(owner=f"u_{i}", entry=f"ep_{i}") for i in range(n)]
    await asyncio.gather(*(repo.upsert([r]) for r in rows))
    assert await repo.count() == n
    for i in range(n):
        got = await repo.get_by_id(f"u_{i}_ep_{i}")
        assert got is not None, f"u_{i}_ep_{i} disappeared after concurrent upsert"


async def test_concurrent_upsert_same_id_last_writer_wins(
    repo: _NoteRepo,
) -> None:
    """Concurrent upserts on the *same* pk must converge: exactly one row,
    one of the texts wins. The lock makes the outcome deterministic per
    schedule (no torn state, no duplicate row)."""
    row_a = _row(owner="u1", entry="ep_1", text="A")
    row_b = _row(owner="u1", entry="ep_1", text="B")
    await asyncio.gather(repo.upsert([row_a]), repo.upsert([row_b]))
    assert await repo.count() == 1
    got = await repo.get_by_id("u1_ep_1")
    assert got is not None
    assert got.text in {"A", "B"}


async def test_read_not_blocked_by_write_lock(repo: _NoteRepo) -> None:
    """Search / count must remain available while a write lock is held —
    only write paths take the lock. Acquires the table lock manually,
    then verifies a read still resolves."""
    await repo.add([_row(owner="u1", entry="ep_1", text="seed")])
    lock = repo._write_lock(repo.table_name)
    async with lock:
        # Whilst the lock is held, reads should not block.
        got = await asyncio.wait_for(repo.get_by_id("u1_ep_1"), timeout=2.0)
    assert got is not None
    assert got.text == "seed"


async def test_write_lock_is_per_table(tmp_path: Path) -> None:
    """Distinct tables share no lock — writes on table A do not stall
    writes on table B."""
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())

    class _OtherNote(BaseLanceTable):
        TABLE_NAME: ClassVar[str] = "_other_note"
        id: str
        owner_id: str
        entry_id: str
        session_id: str
        parent_type: str
        parent_id: str
        md_path: str
        text: str
        vector: Vector(4)  # type: ignore[valid-type]

    class _OtherRepo(LanceDailyLogRepoBase[_OtherNote]):
        schema = _OtherNote

    table_a = await conn.create_table("_note_a", schema=_Note)
    table_b = await conn.create_table(_OtherNote.TABLE_NAME, schema=_OtherNote)

    class _NoteARepo(LanceDailyLogRepoBase[_Note]):
        schema = _Note

        @property
        def table_name(self) -> str:
            return "_note_a"

    repo_a = _NoteARepo(table=table_a)
    repo_b = _OtherRepo(table=table_b)
    assert repo_a._write_lock(repo_a.table_name) is not repo_b._write_lock(
        repo_b.table_name
    )


# ── migrate_fts_indexes (one-time rebuild of pre-fix with_position indexes) ──


async def test_migrate_fts_indexes_runs_once_and_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """migrate_fts_indexes rebuilds existing FTS indexes once, guarded by a
    version marker in the LanceDB dir (fix for lance-format/lance#7653).

    White-box surfaces: the ``.fts_index_version`` marker file and
    ``AsyncTable.list_indices`` on the global-connection table.
    """
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    import everos.infra.persistence.lancedb as lancedb_infra
    from everos.core.persistence import MemoryRoot
    from everos.infra.persistence.lancedb import (
        dispose_connection,
        get_table,
        migrate_fts_indexes,
    )

    monkeypatch.setattr(lancedb_infra, "_BUSINESS_SCHEMAS", (_SearchNote,))
    await dispose_connection()
    try:
        table = await get_table(_SearchNote.TABLE_NAME, _SearchNote)
        await table.add(
            [
                _SearchNote(
                    id="1",
                    text="hello world",
                    tokens="hello world",
                    vector=[1, 0, 0, 0],
                )
            ]
        )
        await _SearchNote.ensure_fts_indexes(table)
        assert any("tokens" in (i.columns or []) for i in await table.list_indices())

        marker = MemoryRoot.resolve().lancedb_dir / ".fts_index_version"
        assert not marker.exists()

        # First run: migrates + writes the marker, index still present.
        await migrate_fts_indexes()
        assert marker.read_text().strip() == "2"
        assert any("tokens" in (i.columns or []) for i in await table.list_indices())

        # Marker present → second run is a no-op. Drop the index, re-run,
        # and confirm it is NOT rebuilt (migration skipped, not re-executed).
        for i in await table.list_indices():
            await table.drop_index(i.name)
        await migrate_fts_indexes()
        assert not list(await table.list_indices())
    finally:
        await dispose_connection()


async def test_prune_holds_write_lock_and_is_cross_process_safe(
    tmp_path: Path,
) -> None:
    """``prune`` runs the underlying optimize **under the per-table write
    lock** (so concurrent churn in this process can't preempt its Rewrite)
    and passes ``delete_unverified=False``.

    The write lock is the fix for the bundled optimize+prune starving
    cleanup under churn (soak: 16 prune successes / 547 commit conflicts
    over 21h → unbounded index dir). ``delete_unverified=False`` is the
    cross-process guard: the in-process lock cannot fence a *second* process
    (a CLI ``cascade sync`` / ``backfill``), and lance warns that
    ``delete_unverified=True`` can corrupt the dataset if any other process
    is writing. ``False`` reclaims identically on churned tables (both
    collapse superseded versions ~97% — measured) because ordinary orphans
    are version-referenced and thus verifiable; ``True`` only additionally
    deletes in-flight/dangling files — exactly the corruption vector.
    """
    captured: dict = {}
    state = {"held": False}

    class _MockTable:
        async def optimize(self, *, cleanup_older_than=None, delete_unverified=False):
            state["held"] = repo._write_lock(repo.table_name).locked()
            captured["cleanup_older_than"] = cleanup_older_than
            captured["delete_unverified"] = delete_unverified

        async def uri(self) -> str:
            return str(tmp_path)

        async def list_indices(self):  # type: ignore[no-untyped-def]
            # Mirrors the real signature: prune reads live index UUIDs so the
            # husk sweep can spare them. A double that omits it hides a real
            # TypeError behind a green test.
            return []

    repo = _NoteRepo(table=_MockTable())  # type: ignore[arg-type]
    await repo.prune(dt.timedelta(seconds=42))

    assert state["held"], "prune must hold the write lock while optimizing"
    assert captured["delete_unverified"] is False
    assert captured["cleanup_older_than"] == dt.timedelta(seconds=42)


async def test_prune_times_out_and_releases_the_write_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hung lance cleanup must not hold the per-table write lock forever.

    On expiry the body is cancelled, the lock is released so writers on that
    table are not wedged, and the timeout surfaces as
    :class:`VectorStoreBusyError` — deliberately a *retryable* error, so the
    cascade worker retries the row instead of marking it permanently failed.
    """
    from everos.core.errors import ExternalServiceError, VectorStoreBusyError
    from everos.core.persistence.lancedb import repository as repo_mod

    monkeypatch.setattr(repo_mod, "_PRUNE_TIMEOUT_SECONDS", 0.05)

    class _HangingTable:
        async def optimize(self, **_kw):  # type: ignore[no-untyped-def]
            await asyncio.sleep(30)  # never returns within the timeout

        async def uri(self) -> str:
            return str(tmp_path)

    repo = _NoteRepo(table=_HangingTable())  # type: ignore[arg-type]
    with pytest.raises(VectorStoreBusyError) as excinfo:
        await repo.prune(dt.timedelta(seconds=60))

    assert isinstance(excinfo.value, ExternalServiceError), (
        "must be retryable — under VectorStoreError the worker would mark the "
        "row permanently failed and need a manual `cascade fix`"
    )
    assert not repo._write_lock(repo.table_name).locked(), (
        "the write lock must be released after the timeout, otherwise a hung "
        "cleanup wedges every writer on this table"
    )
    # And the lock is genuinely reusable afterwards.
    async with repo._write_lock(repo.table_name):
        pass


async def test_waiting_for_a_stuck_holder_also_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**No path may wait for this lock indefinitely.**

    The deadline covers acquisition, not just the body. Without that, one
    operation that hangs while holding the lock wedges the table for good:
    every writer blocks on acquire, and the maintenance scheduler skips a kind
    whose task never finishes, so that table stops reclaiming versions forever
    (observed in a soak run — 150 versions retained, disk 11x live size, and
    *no* error logged anywhere, because nothing failed; it simply never
    returned).
    """
    from everos.core.errors import VectorStoreBusyError
    from everos.core.persistence.lancedb import repository as repo_mod

    monkeypatch.setattr(repo_mod, "_WRITE_TIMEOUT_SECONDS", 0.05)

    class _NoopTable:
        async def add(self, _records):  # type: ignore[no-untyped-def]
            return None

    repo = _NoteRepo(table=_NoopTable())  # type: ignore[arg-type]

    # Simulate a holder that never gives the lock back.
    lock = repo._write_lock(repo.table_name)
    await lock.acquire()
    try:
        with pytest.raises(VectorStoreBusyError):
            await repo.add([_row(owner="u1", entry="n1")])
    finally:
        lock.release()

    # Once the stuck holder is gone, the table works again — the timeout
    # bounded the wait without breaking anything.
    await repo.add([_row(owner="u1", entry="n2")])


async def test_a_hanging_table_handle_still_hits_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolving the table handle must happen **inside** the deadline.

    With it outside, a hang there never returns, and the maintenance scheduler
    runs one task per kind and skips a kind whose task is still in flight — so
    that kind silently stops being maintained. A soak run hit exactly this: one
    table went 13 minutes without a prune, zero failure logs, while its two
    siblings pruned on schedule.
    """
    from everos.core.errors import VectorStoreBusyError
    from everos.core.persistence.lancedb import repository as repo_mod

    monkeypatch.setattr(repo_mod, "_PRUNE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(repo_mod, "_COMPACT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(repo_mod, "_WRITE_TIMEOUT_SECONDS", 0.05)

    class _HangingLookupRepo(_NoteRepo):
        async def _table_lookup(self):  # type: ignore[no-untyped-def]
            await asyncio.sleep(30)  # never resolves within the deadline

    repo = _HangingLookupRepo()

    # Every maintenance/write entry point must give up rather than park.
    with pytest.raises(VectorStoreBusyError):
        await repo.prune(dt.timedelta(seconds=60))
    with pytest.raises(VectorStoreBusyError):
        await repo.optimize()
    with pytest.raises(VectorStoreBusyError):
        await repo.add([_row(owner="u1", entry="e1")])

    # And the lock was never left held.
    assert not repo._write_lock(repo.table_name).locked()


def test_write_budgets_are_sized_from_measurements_not_guesses() -> None:
    """Write budgets must stay in the tens of seconds, not hundreds.

    The budget doubles as the detection latency for a wedged table: a stuck
    holder is invisible until its deadline expires. Measured write durations
    are 2-25ms (worst observed 63ms across 10k-100k row tables and 50-500 row
    batches), so tens of seconds is already ~10^3 headroom. A budget in the
    hundreds of seconds would mean minutes of blocked writers before anything
    is reported, which is what this whole change exists to prevent.
    """
    from everos.core.persistence.lancedb.repository import (
        _PRUNE_TIMEOUT_SECONDS,
        _REBUILD_TIMEOUT_SECONDS,
        _WRITE_TIMEOUT_SECONDS,
    )

    assert 5.0 <= _WRITE_TIMEOUT_SECONDS <= 30.0, (
        "row writes are millisecond operations; a budget outside this range is "
        "either too tight to survive a contended lock or too slack to detect a "
        "wedged table promptly"
    )
    # Rebuild is the one legitimately slow section, so it gets more — but the
    # ordering must hold: a rebuild budget below prune's would make the slowest
    # operation the most eagerly killed.
    assert _REBUILD_TIMEOUT_SECONDS > _PRUNE_TIMEOUT_SECONDS > _WRITE_TIMEOUT_SECONDS


def test_prune_timeout_is_well_below_the_prune_cadence() -> None:
    """The timeout is a hang-catcher, not a bound on normal runtime.

    A real cleanup is milliseconds even on a heavily churned table, so the
    value only matters when lance hangs — and then it must expire well before
    the next heavy beat is due, or the lock is held for most of every cadence
    (the ~97%-duty-cycle bug: timeout == cadence, so a hung prune was retried
    ~immediately after each expiry).
    """
    from everos.core.persistence.lancedb.repository import _PRUNE_TIMEOUT_SECONDS
    from everos.memory.cascade.worker import (
        DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS,
    )

    assert _PRUNE_TIMEOUT_SECONDS < DEFAULT_OPTIMIZE_PRUNE_INTERVAL_SECONDS / 2, (
        "prune timeout must leave a real write window before the next beat"
    )


async def test_hanging_reads_also_hit_a_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reads need a deadline too, for a different reason than writes.

    A read takes no lock, so a hung read blocks no writer — which is why the
    write-side deadline work skipped them. But the cascade drain loop reads on
    every batch (``handlers/_daily_log_base.py`` calls ``find_where``) and
    advances strictly one batch at a time, so a read that never returns stops
    the whole md -> LanceDB projection: claimed rows stay ``processing``
    forever, nothing new is indexed, and ``/health`` still reports healthy
    because a hang raises nothing and the drain-failure counter only counts
    exceptions.
    """
    from everos.core.errors import VectorStoreBusyError
    from everos.core.persistence.lancedb import repository as repo_mod

    monkeypatch.setattr(repo_mod, "_READ_TIMEOUT_SECONDS", 0.05)

    class _HangingLookupRepo(_NoteRepo):
        async def _table_lookup(self):  # type: ignore[no-untyped-def]
            await asyncio.sleep(30)

    repo = _HangingLookupRepo()

    with pytest.raises(VectorStoreBusyError):
        await repo.count()
    with pytest.raises(VectorStoreBusyError):
        await repo.get_by_id("u1_n1")
    with pytest.raises(VectorStoreBusyError):
        await repo.find_where("owner_id = 'u1'")
    with pytest.raises(VectorStoreBusyError):
        await repo.find_where_paginated("owner_id = 'u1'", sort_by="entry_id")
    with pytest.raises(VectorStoreBusyError):
        await repo.search(vector=None, where=None, limit=1)


def test_read_budget_is_generous_enough_never_to_fire_on_a_healthy_read() -> None:
    """The read deadline is a hang-catcher, not a latency SLO.

    everos builds no vector ANN index, so every read is a flat scan — measured
    ~62ms over 117k rows. The budget must stay far above any real scan so it
    cannot turn a large-but-working table into a stream of failures.
    """
    from everos.core.persistence.lancedb.repository import _READ_TIMEOUT_SECONDS

    measured_seconds_per_117k_rows = 0.062
    assert measured_seconds_per_117k_rows * 500 <= _READ_TIMEOUT_SECONDS, (
        "read budget must leave ~500x headroom over the measured flat scan"
    )


async def test_optimize_is_lock_free_compaction_only() -> None:
    """The light beat ``optimize`` compacts only — no ``delete_unverified`` —
    and must NOT hold the write lock, so it never stalls writers (a commit
    conflict against a concurrent write is benign and retried next beat)."""
    captured: dict = {}
    state = {"held": True}

    class _MockTable:
        async def optimize(self, **kwargs):
            state["held"] = repo._write_lock(repo.table_name).locked()
            captured.update(kwargs)

    repo = _NoteRepo(table=_MockTable())  # type: ignore[arg-type]
    await repo.optimize()

    assert state["held"] is False, "light optimize must be lock-free"
    assert captured == {}, "light optimize passes no cleanup/delete_unverified args"


async def test_rebuild_never_leaves_the_column_without_an_fts_index(
    tmp_path: Path,
) -> None:
    """A BM25 query must survive a rebuild — the index is replaced, not dropped.

    Vector search degrades to a flat scan when its index is missing; FTS does
    not. With no inverted index lance raises ``Cannot perform full text search
    unless an INVERTED index has been created``, and because the recall legs
    are gathered without ``return_exceptions`` that one failing leg 500s the
    whole search request. So the drop-then-create rebuild had a window where
    every keyword search on that kind failed. Hammering the table across a
    rebuild is the only way to catch a regression here — a unit double cannot
    reproduce it.
    """

    class _SearchRepo(LanceRepoBase[_SearchNote]):
        schema = _SearchNote

    mr = MemoryRoot(tmp_path)
    mr.ensure()
    conn = await open_lancedb_connection(mr.lancedb_dir, LanceDBSettings())
    table = await conn.create_table("_search_note", schema=_SearchNote)
    repo = _SearchRepo(table=table)
    await repo.add(
        [
            _SearchNote(
                id=f"n{i}",
                text=f"meeting notes alpha {i}",
                tokens=f"meeting notes alpha {i}",
                vector=[1.0, 0.0, 0.0, 0.0],
            )
            for i in range(200)
        ]
    )
    await _SearchNote.ensure_fts_indexes(table)

    failures: list[str] = []
    successes = 0
    stop = False

    async def _hammer() -> None:
        nonlocal successes
        while not stop:
            try:
                await table.query().nearest_to_text("alpha").limit(3).to_list()
                successes += 1
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
            await asyncio.sleep(0)

    hammer = asyncio.create_task(_hammer())
    try:
        for _ in range(3):
            await repo.rebuild_indexes()
    finally:
        stop = True
        await hammer

    assert successes > 0, "the hammer never ran; the test proves nothing"
    assert not failures, (
        f"{len(failures)} keyword queries failed across the rebuild — the index "
        f"went missing. First: {failures[0]}"
    )
    # And the rebuild still did its job: the column is indexed afterwards.
    indexed = {c for i in await table.list_indices() for c in (i.columns or [])}
    assert "tokens" in indexed


def _aged(path: Path, seconds: float) -> Path:
    """Backdate ``path`` so the sweep's age gate accepts it."""
    old = time.time() - seconds
    os.utime(path, (old, old))
    return path


class _SweepTable:
    """Prune-path double: no-op optimize, a uri(), and a live index list."""

    def __init__(self, uri: str, live: tuple[str, ...] = ()) -> None:
        self._uri = uri
        self._live = live

    async def optimize(self, **_kw):  # type: ignore[no-untyped-def]
        return None

    async def uri(self) -> str:
        return self._uri

    async def list_indices(self):  # type: ignore[no-untyped-def]
        return [SimpleNamespace(index_uuid=u) for u in self._live]


async def test_husk_sweep_only_takes_dead_empty_old_dirs(tmp_path: Path) -> None:
    """Every case the sweep must refuse, in one pass.

    lance's cleanup unlinks an index's files but never its directory (there is
    no ``rmdir`` anywhere in ``cleanup.rs`` — it targets object stores, where an
    empty directory is not a thing), so on a local filesystem the husks pile up:
    13061 dirs in one soak run, 98% empty. Sweeping them is ours to do, which
    means the refusals are what needs pinning down.
    """
    from everos.core.persistence.lancedb.repository import _HUSK_MIN_AGE_SECONDS

    indices = tmp_path / "_indices"
    old = _HUSK_MIN_AGE_SECONDS * 2

    dead = indices / "dead-uuid"
    dead.mkdir(parents=True)
    live = indices / "live-uuid"
    fresh = indices / "being-built-right-now"
    populated = indices / "has-files"
    for d in (live, fresh, populated):
        d.mkdir()
    (populated / "part_0").write_text("index data")
    for d in (dead, live, populated):
        _aged(d, old)  # only `fresh` keeps its current mtime

    repo = _NoteRepo(table=_SweepTable(str(tmp_path), live=("live-uuid",)))  # type: ignore[arg-type]
    await repo.prune(dt.timedelta(seconds=1))

    assert not dead.exists(), "a dead, empty, aged husk is the whole point"
    assert live.exists(), (
        "a UUID still in list_indices() must be spared even when its dir looks "
        "empty and old"
    )
    assert fresh.exists(), (
        "a dir younger than lance's own 7-day unverified threshold may be an "
        "index build in progress"
    )
    assert populated.exists() and (populated / "part_0").exists(), (
        "rmdir is refused by the kernel on a non-empty dir — no file can ever "
        "be lost here"
    )


def test_husk_age_gate_matches_lance_own_threshold() -> None:
    """The age bound is lance's number, not one we picked.

    ``cleanup.rs`` sets ``UNVERIFIED_THRESHOLD_DAYS = 7`` and applies it to
    exactly this judgement: an index UUID no manifest references is only assumed
    dead once it is that old, because until then it is indistinguishable from an
    in-progress build. Matching it means this sweep can never be more aggressive
    than lance itself. An earlier version used 300s — our own invention, and the
    reason the sweep was not defensible.
    """
    from everos.core.persistence.lancedb.repository import _HUSK_MIN_AGE_SECONDS

    assert _HUSK_MIN_AGE_SECONDS == 7 * 24 * 60 * 60.0


async def test_husk_sweep_timeout_must_not_fail_the_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep that blows its deadline is the sweep's problem, not prune's.

    By the time the sweep runs, the cleanup commit — the thing prune exists
    for — has already succeeded, and the sweep is best-effort by contract.
    Letting its timeout escape ``prune()`` bills the failure to the wrong
    account: the optimize scheduler counts a prune failure (feeding the
    fallback-rebuild threshold) and the prune-staleness clock stops
    advancing, so both alarms report a cleanup stall that did not happen.
    Not a theoretical path either — sweep time is proportional to dir count
    (~35us/dir measured), and the ceiling-load steady state sits right at
    the sweep budget.
    """
    from everos.core.persistence.lancedb import repository as repo_mod

    monkeypatch.setattr(repo_mod, "_HUSK_SWEEP_TIMEOUT_SECONDS", 0.05)

    def _slow_sweep(table_uri, *, live_uuids, min_age_seconds):  # type: ignore[no-untyped-def]
        time.sleep(0.5)  # to_thread cancellation cannot interrupt this
        return 0

    monkeypatch.setattr(repo_mod, "_remove_empty_index_dirs", _slow_sweep)

    repo = _NoteRepo(table=_SweepTable(str(tmp_path)))  # type: ignore[arg-type]
    await repo.prune(dt.timedelta(seconds=1))  # must not raise


async def test_a_broken_sweep_still_surfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the *timeout* is absorbed — a real fault in the sweep must escape.

    The sibling test pins that a deadline miss does not bill prune. The risk on
    the other side is the catch drifting wider: swallowing every exception
    would turn a genuine bug in ``_remove_empty_index_dirs`` (a TypeError after
    a signature change, a permission error on the index dir) into a silent
    ``removed = 0``, and nothing anywhere would say the sweep had stopped
    working. That is the exact failure shape this module keeps being audited
    for, so the narrowness of the catch is worth a test of its own — widening
    it to ``except Exception`` passes every other test in this file.
    """
    from everos.core.persistence.lancedb import repository as repo_mod

    def _broken_sweep(table_uri, *, live_uuids, min_age_seconds):  # type: ignore[no-untyped-def]
        raise TypeError("sweep signature drifted")

    monkeypatch.setattr(repo_mod, "_remove_empty_index_dirs", _broken_sweep)

    repo = _NoteRepo(table=_SweepTable(str(tmp_path)))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="signature drifted"):
        await repo.prune(dt.timedelta(seconds=1))
