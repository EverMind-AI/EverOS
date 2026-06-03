"""LanceDB IO toolkit — typical workflow demo.

End-to-end story for how to author + use a LanceDB-backed table in everos:

    1. Define a table schema by subclassing :class:`BaseLanceTable` and
       declaring a ``Vector(N)`` column for the embedding.
    2. ``open_lancedb_connection`` to get an :class:`AsyncConnection`.
    3. ``conn.create_table(name, schema=Cls)`` to create the table from
       the Pydantic schema.
    4. ``table.add(rows)`` to insert.
    5. ``table.query().nearest_to(vec).limit(k).to_list()`` for vector
       search (BM25 + scalar filter can chain in the same query).
    6. ``table.count_rows()`` for size.
    7. Mutate via :func:`touch` + :meth:`AsyncTable.update` (LanceDB has
       no SQL ``onupdate`` equivalent — the app must bump ``updated_at``).
    8. ``table.delete(predicate)`` to remove rows.
"""

from __future__ import annotations

from pathlib import Path

from everos.config import LanceDBSettings
from everos.core.persistence import (
    BaseLanceTable,
    MemoryRoot,
    Vector,
    open_lancedb_connection,
)


class _DemoNote(BaseLanceTable):
    """Demo table — used only by this test module."""

    text: str
    vector: Vector(4)  # 4-dim for the test fixture


async def test_lancedb_typical_workflow(tmp_path: Path) -> None:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    settings = LanceDBSettings()

    # 1. Open async connection rooted at <memory_root>/.index/lancedb/
    conn = await open_lancedb_connection(mr.lancedb_dir, settings)

    # 2. Create the table from the BaseLanceTable schema
    table = await conn.create_table("_demo_notes", schema=_DemoNote)

    # 3. Insert rows (Pydantic instances; created_at / updated_at filled in
    #    by BaseLanceTable's default_factory).
    rows = [
        _DemoNote(text="hello world", vector=[1.0, 0.0, 0.0, 0.0]),
        _DemoNote(text="goodbye cruel world", vector=[0.0, 1.0, 0.0, 0.0]),
        _DemoNote(text="welcome aboard", vector=[1.0, 0.5, 0.0, 0.0]),
    ]
    await table.add(rows)

    # 4. Count
    assert await table.count_rows() == 3

    # 5. Vector search — nearest_to picks rows by ANN distance.
    results = await table.query().nearest_to([0.95, 0.05, 0.0, 0.0]).limit(2).to_list()
    assert len(results) == 2
    # The closest row to [0.95, 0.05, 0, 0] is "hello world" [1, 0, 0, 0]
    # ahead of "welcome aboard" [1, 0.5, 0, 0].
    assert results[0]["text"] == "hello world"

    # 6. Filter (scalar predicate). LanceDB SQL-like predicate string.
    only_hello = await table.query().where("text = 'hello world'").to_list()
    assert len(only_hello) == 1
    assert only_hello[0]["text"] == "hello world"

    # 7. Delete by predicate
    await table.delete("text = 'goodbye cruel world'")
    assert await table.count_rows() == 2

    # 8. List tables on the connection
    tables_response = await conn.list_tables()
    assert "_demo_notes" in list(tables_response.tables)

    conn.close()
