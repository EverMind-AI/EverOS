"""Cover every neutral predicate node in the SeekDB SQL renderer."""

from __future__ import annotations

import datetime as dt

import pytest

from everos.infra.persistence import predicate as predicates
from everos.infra.persistence.seekdb.predicate import render_predicate


def test_comparison_in_and_datetime_rendering() -> None:
    moment = dt.datetime(1999, 1, 1, tzinfo=dt.UTC)
    rendered = render_predicate(
        predicates.all_of(
            predicates.eq("owner_id", "o'reilly"),
            predicates.gt("timestamp", moment),
            predicates.one_of("entry_id", ["a", "b"]),
        ),
        datetime_fields={"timestamp"},
    )
    assert "`owner_id` = 'o''reilly'" in rendered
    assert "`timestamp_ms` > 915148800000" in rendered
    assert "`entry_id` IN ('a', 'b')" in rendered


def test_contains_null_and_nested_or_rendering() -> None:
    rendered = render_predicate(
        predicates.any_of(
            predicates.contains("sender_ids", "user"),
            predicates.is_null("subject_vector"),
        ),
        vector_fields={"subject_vector"},
    )
    assert "JSON_CONTAINS(`sender_ids`, '\"user\"')" in rendered
    assert "`subject_vector` IS NULL" in rendered
    assert " OR " in rendered


def test_is_null_uses_physical_datetime_column() -> None:
    assert (
        render_predicate(predicates.is_null("timestamp"), datetime_fields={"timestamp"})
        == "`timestamp_ms` IS NULL"
    )


def test_invalid_fields_and_foreign_predicates_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid SQL identifier"):
        render_predicate(predicates.eq("x`; DELETE", "value"))
    with pytest.raises(TypeError, match="neutral Predicate AST"):
        render_predicate("owner_id = 1")  # type: ignore[arg-type]
