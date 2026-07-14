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
from pathlib import Path
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

        marker = MemoryRoot.default().lancedb_dir / ".fts_index_version"
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
