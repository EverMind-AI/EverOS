"""Pins that the vector backfill covers every vector-carrying index table.

The tables the bootstrap hands backfill come from the index
(:meth:`IndexRepositories.with_vectors`), the per-table wiring (which text
each table embeds) from :data:`VECTOR_PROJECTIONS`. A business table with a
nullable vector column added on the index side without a projection would be
counted by the startup "unbackfilled rows" hint and silently skipped by
backfill, so building a :class:`ProjectionVectorBackfill` over mismatched
tables fails.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Any

import pytest

from everos.component.tokenizer import build_tokenizer
from everos.infra.persistence.index import (
    BUSINESS_SCHEMAS,
    ProjectionVectorBackfill,
)
from tests.fakes.storage import Storage


def test_the_index_and_the_projections_name_the_same_tables(
    storage: Storage,
) -> None:
    tables = storage.repos.with_vectors()

    assert set(tables) == {
        schema.TABLE_NAME for schema in BUSINESS_SCHEMAS if schema.vector_columns()
    }
    _backfill_over(tables)


def test_an_unwired_vector_table_refuses_to_build(storage: Storage) -> None:
    tables = {**storage.repos.with_vectors(), "synthetic_drift_kind": object()}

    with pytest.raises(RuntimeError, match="synthetic_drift_kind"):
        _backfill_over(tables)


def _backfill_over(tables: Mapping[str, Any]) -> ProjectionVectorBackfill:
    return ProjectionVectorBackfill(
        tables, tokenizer=build_tokenizer(), prune_retention=dt.timedelta(0)
    )
