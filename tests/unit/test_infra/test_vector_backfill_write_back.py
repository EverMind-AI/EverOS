"""Write-back contract of the vector backfill.

Pins what the backfill embeds and where it writes: every update targets its
own row by the table's full primary key, and a row is picked up only for a
missing ``vector`` -- the retired ``episode.subject_vector`` column is never
backfilled, whatever it holds.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from everos.infra.persistence.index.vector_backfill import (
    _TABLE_SPECS,
    _backfill_table,
    _NullVectorRow,
    _scan_null_vector_backlog,
    _TableBacklog,
)
from everos.memory.ports.query import Filter
from tests.fakes.backfill import InMemoryVectorTable, episode_row

_RETENTION = dt.timedelta(seconds=60)


def _ignore_progress(table: str, done: int, total: int) -> None:
    return None


class _RecordingRepo:
    """Records every ``update`` as ``(values, where)``."""

    def __init__(self) -> None:
        self.updates: list[tuple[dict[str, Any], Filter]] = []

    async def update(self, values: dict[str, Any], *, where: Filter) -> None:
        self.updates.append((values, where))


class _DistinctProvider:
    """A vector that depends on the text, so a swapped payload is visible."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] * 3 for t in texts]


class _FakeTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()


def _episode_spec() -> Any:
    return next(spec for spec in _TABLE_SPECS if spec.table == "episode")


async def test_update_is_addressed_by_full_key() -> None:
    """Two spaces sharing an ``id``: each update targets its own row only."""
    repo = _RecordingRepo()
    rows = [
        _NullVectorRow(id="shared", text="alpha text", tokens=2, project_id="alpha"),
        _NullVectorRow(id="shared", text="beta text", tokens=2, project_id="beta"),
    ]
    backlog = _TableBacklog(spec=_episode_spec(), index=repo, rows=rows)  # type: ignore[arg-type]

    result = await _backfill_table(
        backlog,
        _DistinctProvider(),  # type: ignore[arg-type]
        prune_retention=_RETENTION,
        on_progress=_ignore_progress,
    )

    assert result.rows_failed == 0
    keys = [
        {c.field: str(c.value) for c in where.children} for _, where in repo.updates
    ]
    assert sorted(k["project_id"] for k in keys) == ["alpha", "beta"]
    assert all(k["id"] == "shared" and k["app_id"] == "default" for k in keys)
    by_project = {
        k["project_id"]: values
        for k, (values, _) in zip(keys, repo.updates, strict=True)
    }
    assert by_project["alpha"] == {"vector": [float(len("alpha text"))] * 3}
    assert by_project["beta"] == {"vector": [float(len("beta text"))] * 3}


async def test_an_episode_missing_only_its_subject_vector_is_not_backfilled() -> None:
    embedded = episode_row("ep_1", subject="Trip planning", vector=[0.1] * 4)
    unembedded = episode_row("ep_2", subject="Trip planning")
    tables = {s.table: InMemoryVectorTable(s.table) for s in _TABLE_SPECS}
    tables["episode"] = InMemoryVectorTable("episode", [embedded, unembedded])

    backlog = await _scan_null_vector_backlog(tables, _FakeTokenizer())  # type: ignore[arg-type]

    [episode] = [t for t in backlog.tables if t.table_name == "episode"]
    assert [(row.id, row.text) for row in episode.rows] == [("alice_ep_2", "hello")]
