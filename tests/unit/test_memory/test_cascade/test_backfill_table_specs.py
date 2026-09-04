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

import pytest

from everos.infra.persistence.index import ALL_REPOS, Episode
from everos.infra.persistence.index.schema import schema_for
from everos.memory.cascade import _backfill


def test_table_specs_covers_business_schemas() -> None:
    """The invariant holds at test time — every schema with a nullable
    vector column is represented in ``_TABLE_SPECS``.

    That the ``import everos.memory.cascade._backfill`` above completed
    at all also proves the module-top-level guard did not raise on this
    tree: the two sets match at import time as well.
    """
    spec_names = {spec.schema.TABLE_NAME for spec in _backfill._TABLE_SPECS}
    schema_names = {
        repo.schema.TABLE_NAME
        for repo in ALL_REPOS
        if schema_for(repo.schema).vector_fields
    }
    assert spec_names == schema_names


def test_drift_scenario_actually_raises_at_import() -> None:
    """Prove the import-time guard actually fires when the two lists
    diverge, not just that unequal sets are unequal (round-4 review
    M10: the prior version of this test was a tautology — it never
    touched ``_backfill`` and would stay green even if the guard
    was deleted).

    Approach: reload ``memory.cascade._backfill`` with the
    ``BUSINESS_SCHEMAS_WITH_VECTOR`` reference swapped to a superset
    that contains a synthetic name absent from ``_TABLE_SPECS``. The
    module's top-level assertion must trip during reload; the test
    asserts on the resulting ``RuntimeError``.
    """
    import importlib
    from types import SimpleNamespace
    from typing import ClassVar

    import everos.infra.persistence.index as index_infra
    import everos.memory.cascade._backfill as backfill_mod

    class _SyntheticDriftSchema(Episode):
        TABLE_NAME: ClassVar[str] = "synthetic_drift_kind"

    fake_schema = _SyntheticDriftSchema
    fake_repo = SimpleNamespace(schema=fake_schema)
    monkey_repos = (*index_infra.ALL_REPOS, fake_repo)

    original = index_infra.ALL_REPOS
    index_infra.ALL_REPOS = monkey_repos  # type: ignore[misc]
    try:
        with pytest.raises(RuntimeError, match=r"synthetic_drift_kind|drift"):
            importlib.reload(backfill_mod)
    finally:
        index_infra.ALL_REPOS = original  # type: ignore[misc]
        importlib.reload(backfill_mod)  # restore module state
