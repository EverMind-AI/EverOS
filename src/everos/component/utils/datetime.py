"""Timezone-aware datetime helpers.

EverOS follows a **two-zone discipline**:

* **Storage** (SQLite + LanceDB) is always UTC. Use :func:`get_utc_now`
  for any ``default_factory`` / write-path timestamp; if you accept a
  ``datetime`` from a caller, normalise with :func:`ensure_utc` before
  it crosses the persistence boundary.
* **Display** (markdown frontmatter, HTTP API response, date buckets for
  daily-log filenames) uses the configured "display timezone" from
  :attr:`everos.config.MemorySettings.timezone` (``EVEROS_MEMORY__TIMEZONE``).
  Use :func:`get_now_with_timezone` / :func:`today_with_timezone` /
  :func:`to_display_tz` here.

The display timezone also serves as the **fallback timezone for naive
input**: if a caller hands us a string / datetime without offset (e.g.
a hand-written ISO timestamp), :func:`from_iso_format` attaches the
display timezone before further processing — that matches a human's
intuition ("if I didn't say a zone, you should assume my zone").

Never call :func:`datetime.datetime.now` /
:func:`datetime.datetime.utcnow` directly — see
:doc:`.claude/rules/datetime-handling`.

Cache invalidation in tests::

    load_settings.cache_clear()
    _display_tz.cache_clear()
"""

from __future__ import annotations

import datetime as _dt
from functools import cache
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import AfterValidator

_MS_THRESHOLD = 1e12  # ts >= this is treated as milliseconds


@cache
def _display_tz() -> _dt.tzinfo:
    """Resolve the configured **display timezone** (cached).

    Reads :attr:`everos.config.MemorySettings.timezone`; that field
    validates the name with :class:`zoneinfo.ZoneInfo` at load time, so
    by the time we reach here the value is guaranteed valid. This
    timezone governs:

    1. ISO output rendered in markdown / API responses.
    2. The fallback zone attached to naive-input datetimes.

    It does **not** govern storage — see :func:`get_utc_now`.
    """
    # Lazy import to avoid pulling in pydantic-settings at module load.
    from everos.config import load_settings

    return ZoneInfo(load_settings().memory.timezone)


def get_utc_now() -> _dt.datetime:
    """Return the current time as a UTC-aware datetime.

    Use for any **storage** write-path (SQLite ``default_factory``,
    LanceDB row construction, OME event ``ts``, any internal "when
    did this happen" record). Independent of the display timezone — a
    new deployment that switches ``EVEROS_MEMORY__TIMEZONE`` will not
    misalign existing rows.

    Display-side code should use :func:`get_now_with_timezone` instead,
    or render via :func:`to_display_tz`.
    """
    return _dt.datetime.now(tz=_dt.UTC)


def get_now_with_timezone() -> _dt.datetime:
    """Return the current time in the **display timezone** (configured).

    Use for **display** write-paths only — markdown frontmatter values,
    daily-log date buckets, places where a human will see the literal
    string. The returned datetime carries the display timezone offset
    so ``.isoformat()`` produces something like
    ``2026-05-29T14:00:00+08:00``.

    For storage / internal "when did this happen" timestamps use
    :func:`get_utc_now` instead — display timezone must not bleed into
    persisted rows.
    """
    return _dt.datetime.now(tz=_display_tz())


def today_with_timezone() -> _dt.date:
    """Return today's date in the **display timezone**.

    Use this anywhere a *date bucket* is needed (e.g. daily-log file
    boundaries) — it normalises ``get_now_with_timezone().date()`` so
    the timezone fallback rules are applied consistently.
    """
    return get_now_with_timezone().date()


def ensure_utc(d: _dt.datetime | None) -> _dt.datetime | None:
    """Normalise any datetime to UTC at the **storage boundary**.

    Semantics:

    * ``None`` → ``None`` (nullable-column convenience: lets callers
      pipe ``ensure_utc(row.last_attempt_at)`` without an outer guard).
    * Aware input → ``astimezone(UTC)``.
    * **Naive input → assume UTC** (attach ``tzinfo=UTC``); no
      display-tz fallback.

    Why naive→UTC rather than naive→display→UTC? Every caller of this
    function sits at the storage boundary, and the dominant naive
    source is SQLite reads: SQLAlchemy strips tz on write so what
    comes back is a naive value whose bytes are UTC. Treating those
    naive reads as display-tz would drift by the configured offset on
    every round trip — exactly the bug Q2 prevents.

    Caller-supplied datetimes that may genuinely be naive in display
    tz (e.g. ISO strings from HTTP request bodies that omitted the
    offset) should be funnelled through :func:`from_iso_format` first,
    which encodes the "if you didn't say a zone, assume your zone"
    rule. The aware result then passes through ``ensure_utc`` as a
    pure ``astimezone(UTC)``.

    Use the :data:`UtcDatetime` ``Annotated`` type to apply this
    automatically on Pydantic model fields.
    """
    if d is None:
        return None
    if d.tzinfo is None:
        return d.replace(tzinfo=_dt.UTC)
    return d.astimezone(_dt.UTC)


def to_display_tz(d: _dt.datetime | None) -> _dt.datetime | None:
    """Convert a datetime to the **display timezone** (configured).

    Used at the **response render boundary**: any datetime leaving the
    system through an API response or markdown body passes through
    here so the user sees their wall-clock time with the matching
    ``+HH:MM`` offset.

    * ``None`` → ``None`` (nullable-column convenience).
    * Naive input is treated as already display-tz local (the fallback
      rule) — attach the zone and return as-is.
    * Aware input is ``astimezone(...)``-d to the display tz.
    """
    if d is None:
        return None
    if d.tzinfo is None:
        return d.replace(tzinfo=_display_tz())
    return d.astimezone(_display_tz())


UtcDatetime = Annotated[_dt.datetime, AfterValidator(ensure_utc)]
"""Pydantic-friendly ``datetime`` type that normalises to UTC.

Apply to any SQLModel / Pydantic ``datetime`` field that maps to a
storage column. Both INSERT default values and post-read values pass
through :func:`ensure_utc`, so SQLite's tz-stripping behaviour is
neutralised: rows go in as UTC and come out as UTC-aware.

Usage::

    from everos.component.utils.datetime import UtcDatetime, get_utc_now

    class MyRow(BaseTable, table=True):
        happened_at: UtcDatetime = Field(default_factory=get_utc_now)
"""


def from_timestamp(ts: int | float) -> _dt.datetime:
    """Parse a Unix timestamp into a timezone-aware datetime.

    Auto-detects seconds vs milliseconds: values ``>= 1e12`` are treated as
    milliseconds. Returned datetime is in the default timezone.
    """
    seconds = ts / 1000.0 if ts >= _MS_THRESHOLD else float(ts)
    return _dt.datetime.fromtimestamp(seconds, tz=_display_tz())


def from_iso_format(value: _dt.datetime | int | float | str) -> _dt.datetime:
    """Parse a value into a timezone-aware datetime (strict).

    Accepted inputs:
        * ``datetime`` — naive values get the default timezone attached.
        * ``int`` / ``float`` — Unix timestamp (auto-detect seconds vs ms).
        * ``str`` — ISO-8601, including ``"Z"`` suffix for UTC.

    Raises:
        TypeError: On unsupported input type.
        ValueError: On malformed string / negative timestamp.
    """
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_display_tz())
        return value
    if isinstance(value, bool):  # bool is an int subclass — reject explicitly
        raise TypeError("from_iso_format does not accept bool")
    if isinstance(value, int | float):
        return from_timestamp(value)
    if isinstance(value, str):
        s = value.strip()
        # Python's fromisoformat accepts "+HH:MM" but not the "Z" suffix; map it.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        parsed = _dt.datetime.fromisoformat(s)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_display_tz())
        return parsed
    raise TypeError(
        f"from_iso_format: unsupported type {type(value).__name__}; "
        "expected datetime / int / float / str"
    )


def to_iso_format(
    value: _dt.datetime | int | float | str | None,
) -> str | None:
    """Render a value as an ISO-8601 string (timezone-aware).

    Accepted inputs:
        * ``None`` — returns ``None`` (nullable column convenience).
        * ``datetime`` — rendered as-is (must already be tz-aware).
        * ``int`` / ``float`` — interpreted via :func:`from_timestamp`.
        * ``str`` — re-validated through :func:`from_iso_format`.
    """
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    if isinstance(value, bool):  # bool is an int subclass
        raise TypeError("to_iso_format does not accept bool")
    if isinstance(value, int | float):
        return from_timestamp(value).isoformat()
    if isinstance(value, str):
        if not value:
            return None
        return from_iso_format(value).isoformat()
    raise TypeError(
        f"to_iso_format: unsupported type {type(value).__name__}; "
        "expected datetime / int / float / str / None"
    )


def to_date_str(d: _dt.datetime | None) -> str | None:
    """Render the date portion of a datetime as ``YYYY-MM-DD``.

    Accepts ``None`` for nullable database columns. When the input is
    already a :class:`datetime.date`, call ``d.isoformat()`` directly.
    """
    if d is None:
        return None
    return d.date().isoformat()


def to_timestamp_ms(d: _dt.datetime) -> int:
    """Convert a datetime to a Unix timestamp in milliseconds."""
    return int(d.timestamp() * 1000)
