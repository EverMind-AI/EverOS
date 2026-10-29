"""Doubles for the collaborators :class:`~everos.service.BackfillDeps` names.

:class:`InMemoryVectorTable` stands in for one index table's maintenance
handle (scan / update / optimize / prune), :class:`ScriptedReplayEngine` for
the throw-away engine a replay phase builds, :class:`RecordingCascade` for
the skill-file sync and :class:`InMemorySkillFiles` for the ``SKILL.md``
probe. :func:`backfill_deps` assembles them over
:func:`~tests.fakes.unused_backfill_deps`, so a test names only what its
phase reaches::

    deps = backfill_deps(embedding=FakeEmbeddingProvider(), episode=[row])
    ops = make_ops(backfill=deps)
"""

from __future__ import annotations

__all__ = [
    "InMemorySkillFiles",
    "InMemoryVectorTable",
    "RecordingCascade",
    "ScriptedReplayEngine",
    "backfill_deps",
    "episode_row",
]

import dataclasses
import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from everos.component.tokenizer import build_tokenizer
from everos.infra.persistence.index import ProjectionVectorBackfill
from everos.memory.ports import Event, Filter, Scope, StrategySpec
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

    async def scan(self, where: Filter | None = None) -> list[BaseModel]:
        return [
            _Record(**row) for row in self.rows if where is None or matches(where, row)
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


class ScriptedReplayEngine:
    """A replay engine that records what it was given and never dispatches.

    ``start_error`` is raised by :meth:`start` (a server that took the lock
    after the probe); ``drains`` is what :meth:`wait_idle` answers.
    """

    def __init__(
        self, *, start_error: Exception | None = None, drains: bool = True
    ) -> None:
        self.start_error = start_error
        self.drains = drains
        self.specs: list[StrategySpec] = []
        self.emitted: list[Event] = []
        self.started = False
        self.stopped = False

    def build(self, specs: Sequence[StrategySpec]) -> ScriptedReplayEngine:
        self.specs = list(specs)
        return self

    async def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def emit(self, event: Event) -> None:
        self.emitted.append(event)

    async def wait_idle(self, *, timeout: float) -> bool:  # noqa: ASYNC109
        return self.drains


class RecordingCascade:
    """Records each ``sync_once`` call's ``kinds``."""

    def __init__(self, on_sync: Callable[[], None] | None = None) -> None:
        self.syncs: list[set[str] | None] = []
        self._on_sync = on_sync

    async def sync_once(self, *, kinds: set[str] | None = None) -> int:
        self.syncs.append(kinds)
        if self._on_sync is not None:
            self._on_sync()
        return 0


class InMemorySkillFiles:
    """``SKILL.md`` probe over a set of ``(owner_id, cluster_id)`` pairs."""

    def __init__(self, named: set[tuple[str, str]] | None = None) -> None:
        self.named = named or set()

    async def names_cluster(self, scope: Scope, cluster_id: str) -> bool:
        return (scope.owner_id, cluster_id) in self.named


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


async def _no_schema() -> None:
    return None


def backfill_deps(
    *,
    tables: Mapping[str, InMemoryVectorTable] | None = None,
    **overrides: Any,
) -> BackfillDeps:
    """Backfill wiring over in-memory tables; anything unnamed fails loudly.

    Every vector table starts empty; ``tables`` replaces some. The lock is
    free and the schema step is a no-op unless overridden.
    """
    base = unused_backfill_deps()
    vector_tables = {name: InMemoryVectorTable(name) for name in base.vector_tables}
    vector_tables.update(tables or {})
    fields: dict[str, Any] = {
        "vectors": ProjectionVectorBackfill(
            vector_tables,  # type: ignore[arg-type]  # in-memory tables
            tokenizer=build_tokenizer(),
            prune_retention=dt.timedelta(seconds=60),
        ),
        "vector_tables": vector_tables,
        "ensure_schema": _no_schema,
        "ome_lock": FakeOmeLock(),
    }
    return dataclasses.replace(base, **{**fields, **overrides})
