"""Safe SQL literal and identifier rendering for both SeekDB client modes.

Embedded pylibseekdb cursors do not support DB-API parameters, so every query
uses these renderers. Identifiers are allow-listed and values are escaped in
one place; repository code must never interpolate caller strings directly.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from typing import Any, Final

from everos.component.utils.datetime import ensure_utc, to_timestamp_ms

_IDENTIFIER: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_identifier(value: str) -> str:
    """Validate and quote one unqualified SQL identifier."""
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"invalid SQL identifier: {value!r}")
    return f"`{value}`"


def literal(value: Any) -> str:
    """Render a scalar as a SeekDB/MySQL SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, dt.datetime):
        aware = ensure_utc(value)
        assert aware is not None
        return str(to_timestamp_ms(aware))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("SQL numeric literals must be finite")
        return repr(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return f"'{_escape_string(value)}'"
    raise TypeError(f"unsupported SQL literal type: {type(value).__name__}")


def json_literal(value: Any) -> str:
    """Render a JSON value as a quoted UTF-8 JSON document."""
    document = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return literal(document)


def _escape_string(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\0", "\\0")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\x1a", "\\Z")
        .replace("'", "''")
    )


__all__ = ["json_literal", "literal", "quote_identifier"]
