"""Laws of the merge, rank and parse helpers on the memorize and search paths.

1. ``merge_dedupe_sort`` (boundary stage): the result is sorted by
   ``(timestamp, message_id)``, holds each id once, its id set is the union
   of both inputs, a buffered message wins over a new one with the same id,
   and merging again changes nothing. ``buffered`` comes from a table keyed
   by ``message_id``, so it never holds an id twice; the law is stated on
   that domain.
2. ``merge_by_id_max`` (search recall): each id once, with the highest score seen
   for it in either list; a tie keeps the primary candidate. A duplicate id
   inside ``primary`` kept the *last* occurrence before this law.
3. ``_coerce_core_indices``: the result is a sub-sequence of ``range(n)``
   in first-occurrence order with no repeats, and coercing the result again
   is the identity. The decider's reply model makes every index an ``int``;
   the strings and bools the parser once tolerated are rejected upstream.
4. ``_merge_status``: commutative, ``extracted`` absorbs.
5. ``reconcile``: every path appears at most once; ``added`` is exactly
   scan minus state, ``deleted`` exactly state minus scan minus rows whose
   last cycle was a completed delete; the decision set does not depend on
   scan order; and once the worker has marked every decision done, the
   next sweep over the same disk state emits nothing.
"""

from __future__ import annotations

import datetime as dt
import random

from everalgo.types import Candidate
from hypothesis import given
from hypothesis import strategies as st

from everos.infra.cascade.models import ScanInput
from everos.infra.cascade.reconciler import PriorState, reconcile
from everos.memory.extract.boundary.messages import merge_dedupe_sort
from everos.memory.models import CanonicalMessage
from everos.memory.search.llm_multiround import _coerce_core_indices
from everos.memory.search.recall import RecallResult, merge_by_id_max
from everos.service.memory._memorize import _merge_status

_UTC = dt.UTC


def _msg(mid: str, ts: int, text: str) -> CanonicalMessage:
    return CanonicalMessage(
        message_id=mid,
        session_id="s",
        sender_id="u",
        role="user",
        timestamp=dt.datetime.fromtimestamp(ts, tz=_UTC),
        text=text,
    )


_msgs = st.lists(
    st.builds(
        _msg,
        st.text("abc", min_size=1, max_size=3),
        st.integers(0, 50),
        st.text(max_size=5),
    ),
    max_size=8,
)


_buffered = st.lists(
    st.builds(
        _msg,
        st.text("abc", min_size=1, max_size=3),
        st.integers(0, 50),
        st.text(max_size=5),
    ),
    max_size=8,
    unique_by=lambda m: m.message_id,
)


@given(buffered=_buffered, new=_msgs)
def test_merge_dedupe_sort_laws(
    buffered: list[CanonicalMessage], new: list[CanonicalMessage]
) -> None:
    merged = merge_dedupe_sort(buffered, new)
    keys = [(m.timestamp, m.message_id) for m in merged]
    assert keys == sorted(keys)
    ids = [m.message_id for m in merged]
    assert len(ids) == len(set(ids))
    assert set(ids) == {m.message_id for m in buffered} | {m.message_id for m in new}
    by_id = {m.message_id: m for m in buffered}
    for m in merged:
        if m.message_id in by_id:
            assert m is by_id[m.message_id], "the buffered message wins"
    assert merge_dedupe_sort(merged, []) == merged
    assert merge_dedupe_sort(merged, new) == merged


_cands = st.lists(
    st.builds(
        lambda i, s: Candidate(id=i, score=s, source="vector", metadata={}),
        st.text("xyz", min_size=1, max_size=2),
        st.floats(0, 1, allow_nan=False),
    ),
    max_size=6,
)


@given(primary=_cands, extra=_cands)
def test_merge_by_id_max_keeps_the_highest_score(
    primary: list[Candidate], extra: list[Candidate]
) -> None:
    merged = merge_by_id_max(
        RecallResult(candidates=primary, stored_by_id={}),
        RecallResult(candidates=extra, stored_by_id={}),
    ).candidates
    ids = [c.id for c in merged]
    assert len(ids) == len(set(ids))
    assert set(ids) == {c.id for c in primary} | {c.id for c in extra}
    for c in merged:
        best = max(x.score for x in (*primary, *extra) if x.id == c.id)
        assert c.score == best
        if any(x.id == c.id and x.score == best for x in primary):
            assert any(x is c for x in primary), "a tie keeps the primary candidate"


@given(raw=st.lists(st.integers(-3, 12), max_size=10), n=st.integers(0, 8))
def test_coerce_core_indices_laws(raw: list[int], n: int) -> None:
    core = _coerce_core_indices(raw, n)
    assert len(core) == len(set(core))
    assert all(0 <= i < n for i in core)
    firsts: list[int] = []
    for idx in raw:
        if 0 <= idx < n and idx not in firsts:
            firsts.append(idx)
    assert core == firsts
    assert _coerce_core_indices(core, n) == core


@given(
    a=st.sampled_from(["accumulated", "extracted"]),
    b=st.sampled_from(["accumulated", "extracted"]),
)
def test_merge_status_is_commutative_and_extracted_absorbs(a: str, b: str) -> None:
    assert _merge_status(a, b) == _merge_status(b, a)  # type: ignore[arg-type]
    assert _merge_status(a, "extracted") == "extracted"  # type: ignore[arg-type]
    assert _merge_status(a, a) == a  # type: ignore[arg-type]


_paths = st.text("abc", min_size=1, max_size=2).map(lambda s: f"users/u/{s}.md")
_prior = st.builds(
    PriorState,
    md_path=_paths,
    kind=st.just("episode"),
    mtime=st.sampled_from([0.0, 1.0, 2.0]),
    status=st.sampled_from(["pending", "processing", "done", "failed"]),
    change_type=st.sampled_from(["added", "modified", "deleted"]),
    retryable=st.sampled_from([None, True, False]),
)
_scan = st.lists(
    st.builds(
        ScanInput,
        md_path=_paths,
        mtime=st.sampled_from([0.0, 1.0, 2.0]),
        kind=st.just("episode"),
    ),
    max_size=6,
    unique_by=lambda s: s.md_path,
)
_state = st.lists(_prior, max_size=6, unique_by=lambda p: p.md_path).map(
    lambda ps: {p.md_path: p for p in ps}
)


@given(scan=_scan, state=_state, seed=st.integers(0, 100))
def test_reconcile_laws(
    scan: list[ScanInput], state: dict[str, PriorState], seed: int
) -> None:
    decisions = reconcile(scan, state)
    paths = [d.md_path for d in decisions]
    assert len(paths) == len(set(paths))
    on_disk = {s.md_path for s in scan}
    assert {d.md_path for d in decisions if d.change_type == "added"} == on_disk - set(
        state
    )
    assert {d.md_path for d in decisions if d.change_type == "deleted"} == {
        p
        for p, prior in state.items()
        if p not in on_disk
        and not (prior.status == "done" and prior.change_type == "deleted")
    }
    shuffled = list(scan)
    random.Random(seed).shuffle(shuffled)
    assert set(reconcile(shuffled, state)) == set(decisions)
    # The worker marks every decision done; the next sweep is quiet.
    after = dict(state)
    for d in decisions:
        after[d.md_path] = PriorState(
            md_path=d.md_path,
            kind=d.kind,
            mtime=d.mtime,
            status="done",
            change_type=d.change_type,
        )
    assert reconcile(scan, after) == []
