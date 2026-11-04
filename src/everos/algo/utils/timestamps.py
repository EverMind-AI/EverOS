"""Time rendering and parsing for prompts, in UTC; every package goes through here rather than formatting its own.

Every instant a model reads has one shape, ``YYYY-MM-DD HH:MM UTC``, to the minute: it is the shape the Episode
prompt asks the model to write absolute times in, so the times the model reads and the times stored in narrative
text compare as equal strings. Dates the model writes and code computes are ``YYYY-MM-DD``.
"""

from __future__ import annotations

__all__ = [
    "format_utc_date",
    "format_utc_time",
    "parse_utc_date",
]

import re
from datetime import UTC, datetime

from everos.algo.utils.errors import InvalidInputError

_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}", re.ASCII)


def format_utc_time(timestamp_ms: int, *, weekday: bool = False) -> str:
    """Render an instant for a prompt, e.g. ``2024-04-10 14:00 UTC``.

    Seconds are dropped, not rounded: ``14:00:59`` renders as ``14:00``. The clock is 24-hour, since a 12-hour
    clock leaves noon and midnight ambiguous.

    Args:
        timestamp_ms: Milliseconds since the epoch.
        weekday: Append the UTC weekday, e.g. ``2024-04-10 14:00 UTC (Wednesday)``. Pass it where the model
            resolves relative times ("last Friday") against this instant; without it the model miscounts weeks.

    Returns:
        The instant in UTC, to the minute, with the ``UTC`` label.
    """
    pattern = "%Y-%m-%d %H:%M UTC (%A)" if weekday else "%Y-%m-%d %H:%M UTC"
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).strftime(pattern)


def format_utc_date(value: int | datetime) -> str:
    """Render a UTC calendar date, e.g. ``2024-04-10``; the inverse of :func:`parse_utc_date`.

    Args:
        value: Milliseconds since the epoch, or an aware datetime such as one :func:`parse_utc_date` returned
            or one computed from it.

    Returns:
        The date of ``value`` in UTC.

    Raises:
        InvalidInputError: If ``value`` is a naive datetime, which names no date in UTC.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise InvalidInputError(
                "format_utc_date got a naive datetime, which names no date in UTC: "
                "attach a timezone (for example tzinfo=UTC) before calling"
            )
        instant = value.astimezone(UTC)
    else:
        instant = datetime.fromtimestamp(value / 1000, tz=UTC)
    return instant.strftime("%Y-%m-%d")


def parse_utc_date(text: str) -> datetime | None:
    """Parse a strict ``YYYY-MM-DD`` date, e.g. one a model wrote.

    Args:
        text: The candidate date. Zero padding is required; time parts, surrounding whitespace and non-ASCII
            digits are rejected.

    Returns:
        Midnight of that date as an aware UTC datetime, or ``None`` when ``text`` is not of that shape or names
        no real date (``2024-02-30``).
    """
    if not _DATE_PATTERN.fullmatch(text):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None
