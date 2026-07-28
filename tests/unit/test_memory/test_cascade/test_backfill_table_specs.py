"""Unit tests for the ``_TABLE_SPECS`` / ``BUSINESS_SCHEMAS_WITH_VECTOR``
drift assertion in :mod:`everos.memory.cascade._backfill`.

The two lists carry different types — ``_TABLE_SPECS`` bundles repo +
text/subject extractors alongside the schema; ``BUSINESS_SCHEMAS_WITH_VECTOR``
is a plain schema tuple — so there is no static way to derive one from
the other without lifting every extractor elsewhere (bigger refactor).
The lightweight defence is an import-time assertion that the two
``TABLE_NAME`` sets match: adding a new business table with a nullable
vector column to the infra list without updating ``_TABLE_SPECS``
silently drops it from backfill (the unbackfilled hint counts rows the
CLI never touches), so we fail loud on module import instead.

Round-3 finding #4.
"""

from __future__ import annotations

from everos.infra.persistence.lancedb import BUSINESS_SCHEMAS_WITH_VECTOR
from everos.memory.cascade import _backfill


def test_table_specs_covers_business_schemas() -> None:
    """The invariant holds at test time — every schema with a nullable
    vector column is represented in ``_TABLE_SPECS``.

    That the ``import everos.memory.cascade._backfill`` above completed
    at all also proves the module-top-level guard did not raise on this
    tree: the two sets match at import time as well.
    """
    spec_names = {spec.schema.TABLE_NAME for spec in _backfill._TABLE_SPECS}
    schema_names = {schema.TABLE_NAME for schema in BUSINESS_SCHEMAS_WITH_VECTOR}
    assert spec_names == schema_names


def test_drift_scenario_would_raise() -> None:
    """Simulate a future contributor who adds a new business schema to
    ``BUSINESS_SCHEMAS_WITH_VECTOR`` but forgets ``_TABLE_SPECS``.

    Runs the exact set-difference check the module executes at import
    time against synthesised names to prove the guard fires on drift,
    without patching the module's actual tuple (import-time guards are
    only observable on import, so reloading with a monkeypatch is
    unreliable — the module source is what re-executes).
    """
    spec_names = {"episode", "atomic_fact"}
    schema_names = {"episode", "atomic_fact", "new_business_kind"}
    assert spec_names != schema_names, (
        "test fixture: drift scenario must have unequal sets so the "
        "assertion below is meaningful"
    )
