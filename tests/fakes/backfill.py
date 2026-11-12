"""Doubles for the collaborators :class:`~everos.service.BackfillDeps` names.

:class:`InMemoryVectorTable` stands in for one index table's maintenance
handle (scan / update / optimize / prune). :func:`backfill_deps` assembles
the vectors phase's backfill over them, on top of
:func:`~tests.fakes.unused_backfill_deps`, so a test names only what the
phase reaches::

    deps = backfill_deps(embedding=FakeEmbeddingProvider(), episode=[row])
    ops = make_ops(backfill=deps)
"""

from __future__ import annotations

__all__ = [
    "InMemoryVectorTable",
    "backfill_deps",
    "episode_row",
]

import dataclasses
import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from everos.component.tokenizer import build_tokenizer
from everos.infra.persistence.index import ProjectionVectorBackfill
from everos.memory.ports import Filter
from everos.service import BackfillDeps

from ._filter import matches
from .application import FakeOmeLock, unused_backfill_deps


class _Record(BaseModel):
    """A raw index row as a record (what ``scan`` returns)."""

    model_config = ConfigDict(extra="allow")


class InMemoryVectorTable:
    """One index table's maintenance handle over a list of raw rows.

    ``updates`` records every partial update as ``(changes, where)``;
    ``optimized`` / ``pruned`` record housekeeping. ``fail_update`` makes
    :meth:`update` raise for the row ids it names.
    """

    def __init__(self, name: str, rows: Sequence[Mapping[str, Any]] = ()) -> None:
        self.table_name = name
        self.rows = [dict(r) for r in rows]
        self.updates: list[tuple[dict[str, Any], Filter]] = []
        self.optimized = 0
        self.pruned: list[dt.timedelta] = []
        self.fail_update: set[str] = set()
        self.fail_optimize = False

    async def scan(
        self, where: Filter | None = None, *, with_vectors: bool = False
    ) -> list[BaseModel]:
        """Like the real backends: vector fields come back only when asked."""
        return [
            _Record(
                **{k: v for k, v in row.items() if with_vectors or "vector" not in k}
            )
            for row in self.rows
            if where is None or matches(where, row)
        ]

    async def update(self, updates: dict[str, Any], *, where: Filter) -> None:
        targets = [row for row in self.rows if matches(where, row)]
        if any(row.get("id") in self.fail_update for row in targets):
            raise RuntimeError("simulated write failure")
        self.updates.append((updates, where))
        for row in targets:
            row.update(updates)

    async def optimize(self) -> None:
        if self.fail_optimize:
            raise RuntimeError("simulated optimize failure")
        self.optimized += 1

    async def prune(self, older_than: dt.timedelta) -> None:
        self.pruned.append(older_than)


def episode_row(entry_id: str = "ep_1", **overrides: Any) -> dict[str, Any]:
    """A raw ``episode`` index row from a pipeline memcell."""
    row: dict[str, Any] = {
        "id": f"alice_{entry_id}",
        "parent_type": "memcell",
        "parent_id": "mc_1",
        "entry_id": entry_id,
        "episode": "hello",
        "subject": None,
        "timestamp": dt.datetime(2023, 11, 14, tzinfo=dt.UTC),
        "owner_id": "alice",
        "session_id": None,
        "app_id": "default",
        "project_id": "default",
        "vector": None,
        "subject_vector": None,
    }
    return {**row, **overrides}


def backfill_deps(
    *,
    tables: Mapping[str, InMemoryVectorTable] | None = None,
    **overrides: Any,
) -> BackfillDeps:
    """Backfill wiring over in-memory tables; anything unnamed fails loudly.

    Every vector table starts empty; ``tables`` replaces some. The lock is
    free unless overridden.
    """
    vector_tables = {name: InMemoryVectorTable(name) for name in _VECTOR_TABLES}
    vector_tables.update(tables or {})
    fields: dict[str, Any] = {
        "vectors": ProjectionVectorBackfill(
            vector_tables,  # type: ignore[arg-type]  # in-memory tables
            tokenizer=build_tokenizer(),
            prune_retention=dt.timedelta(seconds=60),
        ),
        "ome_lock": FakeOmeLock(),
    }
    return dataclasses.replace(unused_backfill_deps(), **{**fields, **overrides})


_VECTOR_TABLES = ("episode", "image_memory")
"""The index tables that carry a vector column."""
