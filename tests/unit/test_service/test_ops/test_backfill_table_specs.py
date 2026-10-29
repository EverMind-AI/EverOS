"""Pins that backfill's vectors phase covers every vector-carrying index table.

The tables the bootstrap hands backfill come from the index
(:meth:`IndexRepositories.with_vectors`), the per-table wiring (which text
each table embeds) lives in the vectors phase. A business table with a
nullable vector column added on the index side without wiring would be
counted by the startup "unbackfilled rows" hint and silently skipped by
backfill, so building an :class:`OpsService` over mismatched tables fails.
"""

from __future__ import annotations

import dataclasses

import pytest

from everos.infra.persistence.index import BUSINESS_SCHEMAS, schema_for
from everos.service import BackfillDeps
from tests.fakes import make_ops, unused_backfill_deps
from tests.fakes.storage import Storage


def test_the_index_and_the_vectors_phase_name_the_same_tables(
    storage: Storage,
) -> None:
    tables = storage.repos.with_vectors()

    assert set(tables) == {
        schema.TABLE_NAME
        for schema in BUSINESS_SCHEMAS
        if schema_for(schema).vector_fields
    }
    make_ops(backfill=_backfill_over(tables))


def test_an_unwired_vector_table_refuses_to_build() -> None:
    tables = {**unused_backfill_deps().vector_tables, "synthetic_drift_kind": object()}

    with pytest.raises(RuntimeError, match="synthetic_drift_kind"):
        make_ops(backfill=_backfill_over(tables))


def _backfill_over(tables: dict[str, object]) -> BackfillDeps:
    return dataclasses.replace(unused_backfill_deps(), vector_tables=tables)
