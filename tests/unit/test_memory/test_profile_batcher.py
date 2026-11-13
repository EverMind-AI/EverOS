"""Tests for :class:`ProfileBatcher`: the per-call cap and the folded-ahead marks.

A mark says another run folded the episode; its own run takes it once and
skips. Marks are per owner and capped, oldest dropped first.
"""

from __future__ import annotations

import pytest

from everos.memory import profile_batcher as batcher_mod
from everos.memory.ports import Scope
from everos.memory.profile_batcher import ProfileBatcher


def _scope(owner_id: str = "u_alice") -> Scope:
    return Scope(
        app_id="default", project_id="default", owner_id=owner_id, owner_type="user"
    )


def test_a_mark_is_taken_once() -> None:
    batcher = ProfileBatcher(10)
    batcher.mark(_scope(), ["ep_1"])

    assert batcher.is_marked(_scope(), "ep_1")
    assert batcher.take(_scope(), "ep_1") is True
    assert batcher.take(_scope(), "ep_1") is False
    assert not batcher.is_marked(_scope(), "ep_1")


def test_marks_are_per_owner() -> None:
    batcher = ProfileBatcher(10)
    batcher.mark(_scope("u_alice"), ["ep_1"])

    assert not batcher.is_marked(_scope("u_bob"), "ep_1")
    assert batcher.take(_scope("u_bob"), "ep_1") is False
    assert batcher.take(_scope("u_alice"), "ep_1") is True


def test_an_owner_keeps_only_the_newest_marks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batcher_mod, "_MAX_MARKS_PER_OWNER", 2)
    batcher = ProfileBatcher(10)
    batcher.mark(_scope(), ["ep_1", "ep_2"])
    batcher.mark(_scope(), ["ep_3"])

    assert [batcher.is_marked(_scope(), e) for e in ("ep_1", "ep_2", "ep_3")] == [
        False,
        True,
        True,
    ]


def test_marking_nothing_leaves_no_owner_behind() -> None:
    batcher = ProfileBatcher(10)
    batcher.mark(_scope(), [])

    assert batcher._marks == {}


@pytest.mark.parametrize("max_episodes", [0, -1])
def test_the_cap_counts_the_runs_own_episode(max_episodes: int) -> None:
    with pytest.raises(ValueError, match="max_episodes"):
        ProfileBatcher(max_episodes)
