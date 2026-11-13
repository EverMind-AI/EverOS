"""The forgetting curve's arithmetic, keys and per-search weighing.

Pins the 2.0 plan's numbers (retrieval section, "forgetting curve"):
``R = 1 / (1 + t / (9 S))`` is 0.9 at ``t = S`` and keeps a long tail (0.27 for
a two-year-old memory at ``S = 30``); a use multiplies ``S`` by
``1 + g (1 - R)`` (at most the cap) and restarts the clock, so a memory
recalled after a long gap gains much and one just used almost nothing; the
score factor is ``(1 - w + w R) * state``, so age takes at most ``w`` and a
past version is scaled by the state factor; one strength per memory, shared by
its versions and never across owners.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Collection, Mapping

import pytest
from hypothesis import given
from hypothesis import strategies as st

from everos.config.settings import GroupSearchSettings
from everos.memory.models import FORGOTTEN_MARK, ImageMemory
from everos.memory.ports import (
    MemoryRef,
    Reinforcement,
    Scope,
    Stored,
    Strength,
    StrengthKey,
)
from everos.memory.search.forgetting import (
    DAY_MS,
    Curve,
    CurveRanking,
    reinforced,
    reinforcements,
    retention,
    state_of,
    strength_key,
    strengthen,
)

from ._records import TS_MS, stored_episode

CURVE = Curve()


def _picture(entry_id: str, *, image_id: str, parent_id: str) -> Stored[ImageMemory]:
    return Stored(
        ref=MemoryRef(id=f"alice_{entry_id}", entry_id=entry_id, md_path="i.md"),
        scope=Scope(
            app_id="default", project_id="default", owner_id="alice", owner_type="user"
        ),
        item=ImageMemory(
            owner_id="alice",
            description="a picture",
            timestamp=TS_MS,
            image_id=image_id,
            parent_id=parent_id,
        ),
    )


# ── retention ──────────────────────────────────────────────────────────────────


def test_retention_is_one_now_and_nine_tenths_after_s_days() -> None:
    assert retention(0, 30) == 1.0
    assert retention(30, 30) == pytest.approx(0.9)
    assert retention(90, 90) == pytest.approx(0.9)


def test_retention_keeps_a_long_tail() -> None:
    """Two years at S = 30: the power law keeps 0.27, an exponential 0.08."""
    assert retention(730, 30) == pytest.approx(1 / (1 + 730 / 270))
    assert retention(730, 30) == pytest.approx(0.27, abs=0.005)
    assert math.exp(math.log(0.9) * 730 / 30) == pytest.approx(0.077, abs=0.005)


def test_a_clock_behind_the_memory_reads_zero_days() -> None:
    assert retention(-5, 30) == 1.0


@given(
    st.floats(min_value=0, max_value=5_000),
    st.floats(min_value=0, max_value=5_000),
    st.floats(min_value=0.5, max_value=3_650),
)
def test_retention_falls_with_time_and_never_reaches_zero(
    a: float, b: float, strength: float
) -> None:
    early, late = sorted((a, b))
    assert 0 < retention(late, strength) <= retention(early, strength) <= 1


# ── reinforcement ──────────────────────────────────────────────────────────────


def test_a_long_gap_gains_much_and_a_moment_almost_nothing() -> None:
    assert reinforced(30, 0, gain=2, cap=3650) == 30
    assert reinforced(30, 30, gain=2, cap=3650) == pytest.approx(30 * 1.2)
    after_a_year = reinforced(30, 365, gain=2, cap=3650)
    assert after_a_year == pytest.approx(30 * (1 + 2 * (1 - 1 / (1 + 365 / 270))))
    assert after_a_year > 60
    assert reinforced(30, 1 / 24, gain=2, cap=3650) < 30.01


def test_strength_never_passes_the_cap() -> None:
    assert reinforced(3000, 10_000, gain=2, cap=3650) == 3650
    assert reinforced(5000, 0, gain=2, cap=3650) == 3650


def _event(
    *, since: int = TS_MS, at: int = TS_MS, memory: str = "ep:memcell:mc_1"
) -> Reinforcement:
    key = StrengthKey("default", "default", "user", "alice", memory)
    return Reinforcement(key=key, since=since, at=at)


def test_a_first_use_grows_the_initial_strength_from_the_memorys_own_time() -> None:
    first = strengthen(CURVE, None, _event(at=TS_MS + 30 * DAY_MS))
    assert first.days == pytest.approx(30 * 1.2)
    assert first.reinforced_at == TS_MS + 30 * DAY_MS
    assert first.reinforcements == 1


def test_a_later_use_counts_from_the_last_one_and_restarts_the_clock() -> None:
    stored = Strength(days=36.0, reinforced_at=TS_MS + 30 * DAY_MS, reinforcements=1)
    again = strengthen(CURVE, stored, _event(at=TS_MS + 66 * DAY_MS))
    assert again.days == pytest.approx(36 * 1.2)
    assert again.reinforced_at == TS_MS + 66 * DAY_MS
    assert again.reinforcements == 2


def test_a_newer_version_restarts_the_clock_of_a_stored_strength() -> None:
    """A scene updated after its last use counts from the update."""
    stored = Strength(days=30.0, reinforced_at=TS_MS, reinforcements=1)
    update = TS_MS + 300 * DAY_MS
    used = strengthen(CURVE, stored, _event(since=update, at=update + 30 * DAY_MS))
    assert used.days == pytest.approx(30 * 1.2)


def test_the_clock_never_runs_backwards() -> None:
    stored = Strength(days=40.0, reinforced_at=TS_MS + 10 * DAY_MS, reinforcements=3)
    early = strengthen(CURVE, stored, _event(at=TS_MS))
    assert early.days == 40.0
    assert early.reinforced_at == TS_MS + 10 * DAY_MS
    assert early.reinforcements == 4


# ── keys and states ────────────────────────────────────────────────────────────


def test_versions_of_one_episode_share_its_strength() -> None:
    first = stored_episode("ep_1", parent_id="mc_9")
    edit = stored_episode("ep_7", parent_id="mc_9")
    other = stored_episode("ep_2", parent_id="mc_8")
    assert strength_key(first) == strength_key(edit)
    assert strength_key(first) != strength_key(other)
    assert strength_key(first).memory == "ep:memcell:mc_9"


def test_a_scene_is_keyed_by_its_scene_and_a_picture_by_file_and_segment() -> None:
    version_1 = stored_episode("sc_1", parent_id="scn_trip", parent_type="scene")
    version_2 = stored_episode("sc_2", parent_id="scn_trip", parent_type="scene")
    assert strength_key(version_1) == strength_key(version_2)
    assert strength_key(version_1).memory == "sc:scn_trip"
    picture = _picture("im_1", image_id="h" * 64, parent_id="mc_9")
    revised = _picture("im_2", image_id="h" * 64, parent_id="mc_9")
    elsewhere = _picture("im_3", image_id="h" * 64, parent_id="mc_4")
    assert strength_key(picture) == strength_key(revised)
    assert strength_key(picture) != strength_key(elsewhere)


def test_two_owners_never_share_a_strength() -> None:
    """Two people's copies of one segment, and a group's record of it."""
    ann = stored_episode("ep_1", owner_id="u_ann", parent_id="mc_9")
    bob = stored_episode("ep_1", owner_id="u_bob", parent_id="mc_9")
    group = stored_episode(
        "gep_1", owner_id="trip", owner_type="group", parent_id="mc_9"
    )
    assert len({strength_key(ann), strength_key(bob), strength_key(group)}) == 3


def test_a_record_without_a_parent_is_its_own_memory() -> None:
    orphan = stored_episode("ep_5", parent_id="")
    assert strength_key(orphan).memory == f"id:{orphan.ref.id}"


def test_the_state_comes_from_the_rows_mark() -> None:
    current = stored_episode("ep_1")
    assert state_of(current) == "current"
    assert state_of(dataclasses.replace(current, mark="ep_20260102_0002")) == (
        "superseded"
    )
    assert state_of(dataclasses.replace(current, mark=FORGOTTEN_MARK)) == "forgotten"


def test_one_reinforcement_per_memory_from_its_newest_version() -> None:
    older = stored_episode("ep_1", parent_id="mc_9", timestamp=TS_MS)
    newer = stored_episode("ep_7", parent_id="mc_9", timestamp=TS_MS + 5)
    other = stored_episode("ep_2", parent_id="mc_8")
    found = reinforcements([older, newer, other], at=TS_MS + 100)
    assert len(found) == 2
    by_memory = {e.key.memory: e for e in found}
    assert by_memory["ep:memcell:mc_9"].since == TS_MS + 5
    assert {e.at for e in found} == {TS_MS + 100}


# ── the weighing of one search ─────────────────────────────────────────────────


class _Store:
    def __init__(self, rows: Mapping[StrengthKey, Strength] | None = None) -> None:
        self.rows = dict(rows or {})
        self.reads: list[set[StrengthKey]] = []

    async def read(self, keys: Collection[StrengthKey]) -> dict[StrengthKey, Strength]:
        self.reads.append(set(keys))
        return {k: v for k, v in self.rows.items() if k in keys}

    async def write(self, rows: Mapping[StrengthKey, Strength]) -> None:
        self.rows.update(rows)


async def test_the_factor_is_retention_weighted_and_age_takes_at_most_w() -> None:
    now = TS_MS + 30 * DAY_MS
    ranking = CurveRanking(CURVE, now=now, retain=True, store=_Store())
    fresh = stored_episode("ep_1", parent_id="mc_1", timestamp=now)
    month = stored_episode("ep_2", parent_id="mc_2", timestamp=TS_MS)
    ancient = stored_episode("ep_3", parent_id="mc_3", timestamp=TS_MS - 10**13)
    await ranking.load([fresh, month, ancient])
    assert ranking.factor(fresh) == 1.0
    assert ranking.factor(month) == pytest.approx(0.7 + 0.3 * 0.9)
    assert 0.7 < ranking.factor(ancient) < 0.71


async def test_a_stored_strength_slows_the_forgetting() -> None:
    now = TS_MS + 300 * DAY_MS
    old = stored_episode("ep_1", parent_id="mc_1", timestamp=TS_MS)
    store = _Store(
        {strength_key(old): Strength(days=300.0, reinforced_at=TS_MS, reinforcements=4)}
    )
    plain = CurveRanking(CURVE, now=now, retain=True, store=None)
    strong = CurveRanking(CURVE, now=now, retain=True, store=store)
    await plain.load([old])
    await strong.load([old])
    assert plain.retention_of(old) == pytest.approx(1 / (1 + 300 / 270))
    assert strong.retention_of(old) == pytest.approx(0.9)


async def test_a_past_version_is_scaled_by_the_state_factor() -> None:
    current = stored_episode("ep_1", parent_id="mc_1")
    replaced = dataclasses.replace(current, mark="ep_20260105_0003")
    forgotten = dataclasses.replace(current, mark=FORGOTTEN_MARK)
    only_state = CurveRanking(CURVE, now=TS_MS, retain=False, store=None)
    assert only_state.factor(current) == 1.0
    assert only_state.factor(replaced) == pytest.approx(0.3)
    assert only_state.factor(forgotten) == pytest.approx(0.3)
    both = CurveRanking(CURVE, now=TS_MS + 30 * DAY_MS, retain=True, store=None)
    assert both.factor(replaced) == pytest.approx(0.3 * (0.7 + 0.3 * 0.9))


async def test_strengths_are_read_once_per_memory_per_search() -> None:
    store = _Store()
    ranking = CurveRanking(CURVE, now=TS_MS, retain=True, store=store)
    a, b = stored_episode("ep_1", parent_id="mc_1"), stored_episode("ep_2")
    await ranking.load([a, b])
    await ranking.load([a, b])
    await ranking.load([a])
    assert store.reads == [{strength_key(a), strength_key(b)}]


async def test_without_retention_nothing_is_read() -> None:
    store = _Store()
    ranking = CurveRanking(CURVE, now=TS_MS, retain=False, store=store)
    await ranking.load([stored_episode("ep_1")])
    assert store.reads == []


def test_the_curve_reads_its_parameters_from_search_settings() -> None:
    settings = GroupSearchSettings(
        initial_strength_days=10,
        max_strength_days=100,
        reinforce_gain=1.5,
        retention_weight=0.2,
        past_state_factor=0.5,
    )
    assert Curve.of(settings) == Curve(
        initial_days=10, max_days=100, gain=1.5, weight=0.2, past_factor=0.5
    )
    assert Curve.of(GroupSearchSettings()) == CURVE
