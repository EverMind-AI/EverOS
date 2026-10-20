"""Two small laws: the idle store's string order and the cosine score.

1. ``IdleStore`` keeps ``last_activity_ts`` as an ISO string and its
   ``scan_idle`` compares strings in SQL so the index stays usable; the
   module docstring says the string order matches temporal order. That is
   only true when every stored and compared value is rendered in one zone:
   ``to_iso_format`` keeps the input's offset, so a ``touch`` or a
   ``scan_idle(now=...)`` called with a non-UTC aware datetime compared
   wrongly against UTC rows. The store now normalises to UTC before
   rendering, and the law holds for any aware input.
2. ``cosine_score_from_distance`` maps a distance to a score in ``[0, 1]``,
   never increasing with distance, and ``None`` to 0.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import shutil
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from everos.infra.ome._stores.idle import IdleStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.memory.search.recall.base import cosine_score_from_distance

_UTC = dt.UTC
_instants = st.integers(0, 4_102_444_800).map(
    lambda s: dt.datetime(1970, 1, 1, tzinfo=_UTC) + dt.timedelta(seconds=s)
)
_offsets = st.integers(-14 * 60, 14 * 60).map(
    lambda m: dt.timezone(dt.timedelta(minutes=m))
)


async def _idle_order(
    touches: list[tuple[str, dt.datetime, dt.timezone]],
    now: dt.datetime,
    now_zone: dt.timezone,
    idle_seconds: int,
) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="idle_laws_"))
    try:
        storage = OMEStorage(tmp / "ome.db")
        await storage.init()
        store = IdleStore(storage=storage)
        latest: dict[str, dt.datetime] = {}
        for bucket, at, zone in touches:
            await store.touch("s", bucket, at=at.astimezone(zone))
            latest[bucket] = at
        got = await store.scan_idle(
            "s", idle_seconds=idle_seconds, now=now.astimezone(now_zone)
        )
        cutoff = now - dt.timedelta(seconds=idle_seconds)
        want = sorted(
            (b for b, at in latest.items() if at <= cutoff), key=lambda b: latest[b]
        )
        assert got == want, (got, want)
        for bucket, at in latest.items():
            stored = await store.get_last_activity("s", bucket)
            assert stored == at
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@given(
    touches=st.lists(
        st.tuples(st.sampled_from(["a", "b", "c"]), _instants, _offsets), max_size=6
    ),
    now=_instants,
    now_zone=_offsets,
    idle_seconds=st.integers(1, 10**6),
)
@settings(max_examples=60, deadline=None)
def test_idle_store_orders_by_instant_whatever_the_zone(
    touches: list[tuple[str, dt.datetime, dt.timezone]],
    now: dt.datetime,
    now_zone: dt.timezone,
    idle_seconds: int,
) -> None:
    asyncio.run(_idle_order(touches, now, now_zone, idle_seconds))


@given(
    a=st.one_of(st.none(), st.floats(-1, 3, allow_nan=False)),
    b=st.floats(-1, 3, allow_nan=False),
)
def test_cosine_score_is_bounded_and_monotone(a: float | None, b: float) -> None:
    s = cosine_score_from_distance(a)
    assert 0.0 <= s <= 1.0
    if a is None:
        assert s == 0.0
        return
    lo, hi = sorted((a, b))
    assert cosine_score_from_distance(lo) >= cosine_score_from_distance(hi)
