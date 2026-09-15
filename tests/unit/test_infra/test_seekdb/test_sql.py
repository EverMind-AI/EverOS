"""Pin SeekDB identifier and literal escaping against injection-shaped input."""

from __future__ import annotations

import datetime as dt

import pytest

from everos.infra.persistence.seekdb.sql import (
    json_literal,
    literal,
    quote_identifier,
)


def test_identifier_validation_is_allow_listed() -> None:
    assert quote_identifier("everos_episode") == "`everos_episode`"
    for invalid in ("a.b", "bad-name", "x`; DROP TABLE t", ""):
        with pytest.raises(ValueError, match="invalid SQL identifier"):
            quote_identifier(invalid)


def test_literals_escape_mysql_control_and_quote_characters() -> None:
    assert literal("o'reilly") == "'o''reilly'"
    assert literal("a\\b") == "'a\\\\b'"
    assert literal("'; DROP TABLE memories") == "'''; DROP TABLE memories'"
    assert literal(True) == "TRUE"
    assert literal(None) == "NULL"
    assert literal(dt.datetime(1970, 1, 1, tzinfo=dt.UTC)) == "0"


def test_json_literals_are_compact_and_finite() -> None:
    assert json_literal(["红 苹果", "quote'"]) == "'[\"红 苹果\",\"quote''\"]'"
    with pytest.raises(ValueError, match="finite"):
        literal(float("nan"))
