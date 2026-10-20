"""Laws of the datetime helpers, checked on generated instants.

``everos.component.utils.datetime`` is the one place the two-zone discipline
(UTC in storage, display zone at the edges) is implemented, and every
timestamp in SQLite, LanceDB and the markdown files passes through it. The
functions promise algebraic facts in their docstrings; this module states
each as a property over the whole input domain instead of a few picked
values:

1. ``from_timestamp_ms`` is the exact inverse of ``to_timestamp_ms``: for
   every epoch-millisecond value in the supported range, converting to a
   datetime and back returns the same integer.
2. ``to_timestamp_ms`` truncates (never rounds) a sub-millisecond instant
   and is monotone in the instant.
3. ``ensure_utc`` is idempotent and instant-preserving; a naive input is
   read as UTC (the storage-boundary contract: SQLite hands timestamps back
   naive, and they must not drift), whatever the display zone is.
4. ``to_iso_format`` / ``from_iso_format`` preserve the instant, and the
   rendered string carries an explicit offset so it can never be re-read
   as naive.
5. ``from_timestamp`` applies the documented heuristic: values at or above
   ``1e12`` are milliseconds, below are seconds. Instants in milliseconds
   before 2001-09-09 fall under the threshold and are read as seconds --
   that is a documented limitation of the heuristic, not a target of this
   test; ``from_timestamp_ms`` exists for callers who know the unit.

The first law failed before ``to_timestamp_ms`` switched from
``int(timestamp() * 1000)`` to integer arithmetic on the timedelta since the
epoch: the float product sat just below the integer for about 0.6 % of
values (e.g. 1095954883735 came back as 1095954883734).
"""

from __future__ import annotations

import datetime as dt

from hypothesis import given, settings
from hypothesis import strategies as st

from everos.component.utils import datetime as dtu

# 1970-01-01 .. 2100-01-01, in epoch milliseconds.
_MS_MAX = 4_102_444_800_000
_UTC = dt.UTC


def _ms(ms: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(ms / 1000, tz=_UTC)


aware_instants = st.integers(0, _MS_MAX * 1000).map(
    lambda us: dt.datetime(1970, 1, 1, tzinfo=_UTC) + dt.timedelta(microseconds=us)
)


@given(ms=st.integers(0, _MS_MAX))
@settings(max_examples=2000)
def test_ms_round_trip_is_exact(ms: int) -> None:
    assert dtu.to_timestamp_ms(dtu.from_timestamp_ms(ms)) == ms


@given(d=aware_instants)
def test_to_timestamp_ms_truncates_and_is_monotone(d: dt.datetime) -> None:
    ms = dtu.to_timestamp_ms(d)
    assert _ms(ms) <= d < _ms(ms + 1)
    assert dtu.to_timestamp_ms(d + dt.timedelta(milliseconds=1)) == ms + 1


@given(d=aware_instants, offset_minutes=st.integers(-14 * 60, 14 * 60))
def test_ensure_utc_is_idempotent_and_keeps_the_instant(
    d: dt.datetime, offset_minutes: int
) -> None:
    shifted = d.astimezone(dt.timezone(dt.timedelta(minutes=offset_minutes)))
    once = dtu.ensure_utc(shifted)
    assert once is not None
    assert once.tzinfo == _UTC
    assert once == d  # same instant
    assert dtu.ensure_utc(once) == once


@given(d=aware_instants)
def test_ensure_utc_reads_naive_input_as_utc(d: dt.datetime) -> None:
    naive = d.replace(tzinfo=None)
    got = dtu.ensure_utc(naive)
    assert got is not None and got.tzinfo == _UTC
    # Not the display zone: a naive value crossing the storage boundary is
    # already UTC (review on #521 caught the first version of this law,
    # which only held because the default display zone is UTC).
    assert got == naive.replace(tzinfo=_UTC)


@given(d=aware_instants, offset_minutes=st.integers(-14 * 60, 14 * 60))
def test_iso_round_trip_keeps_the_instant(d: dt.datetime, offset_minutes: int) -> None:
    shifted = d.astimezone(dt.timezone(dt.timedelta(minutes=offset_minutes)))
    text = dtu.to_iso_format(shifted)
    assert text is not None
    back = dtu.from_iso_format(text)
    assert back == d
    assert back.utcoffset() is not None, "the rendered form must carry an offset"


@given(seconds=st.integers(0, _MS_MAX // 1000))
def test_from_timestamp_reads_values_below_the_threshold_as_seconds(
    seconds: int,
) -> None:
    # Up to 2100: beyond that the platform's own fromtimestamp refuses on
    # Windows long before the 1e12 threshold matters.
    expected = dt.datetime(1970, 1, 1, tzinfo=_UTC) + dt.timedelta(seconds=seconds)
    assert dtu.from_timestamp(seconds) == expected


@given(ms=st.integers(10**12, _MS_MAX))
def test_from_timestamp_reads_values_at_or_above_the_threshold_as_ms(
    ms: int,
) -> None:
    assert dtu.from_timestamp(ms) == dtu.from_timestamp_ms(ms)
