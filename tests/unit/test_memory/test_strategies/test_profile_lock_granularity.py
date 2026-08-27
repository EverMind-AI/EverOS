"""Per-sender profiles lock per subject, not per owner.

The partition lock was owner-wide from when every owner had exactly one
``user.md``. Per-sender mode broke that assumption: each subject owns its own
file, so an owner-wide lock buys no extra safety while serialising every task
for the owner across all N subjects.

Cost of getting it wrong, measured on a 38-speaker owner: one subject stuck
inside the loop held the owner lock, 60 of the engine's 64 slots piled up behind
it, and the process stopped running any strategy -- including the extractors that
had nothing to do with profiles -- for 6.7 hours.

The fix is granularity, not removal: two tasks that reach the *same* speaker must
still serialise, because that path is read -> LLM merge -> overwrite on one file.
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

from everos.memory import _partition_locks

# import_module, not `from ... import extract_user_profile`: the decorator binds
# a Strategy object to that name in the package namespace, which shadows the
# module the source assertions below need to reach.
mod = importlib.import_module("everos.memory.strategies.extract_user_profile")


@pytest.fixture(autouse=True)
def _clean_locks() -> None:
    _partition_locks._reset_for_tests()


def _lock(owner: str, subject: str) -> asyncio.Lock:
    return _partition_locks.get_partition_lock(
        "extract_user_profile", f"app:proj:{owner}::{subject}"
    )


def test_different_subjects_of_one_owner_get_different_locks() -> None:
    """The property that keeps one stuck speaker from freezing the group."""
    assert _lock("group", "Lan Ye") is not _lock("group", "Bo Chen")


def test_the_same_subject_still_serialises() -> None:
    """Granularity, not removal -- the file still needs a single writer."""
    assert _lock("group", "Lan Ye") is _lock("group", "Lan Ye")


def test_the_owner_lock_and_a_subject_lock_are_distinct() -> None:
    """Owner mode and sender mode must not collide on one key.

    ``<owner>`` and ``<owner>::<subject>`` are different partitions; a subject
    named such that the two collide would reintroduce the stall on one owner.
    """
    owner_key = _partition_locks.get_partition_lock(
        "extract_user_profile", "app:proj:group"
    )
    assert owner_key is not _lock("group", "Lan Ye")


def test_sender_mode_takes_no_owner_wide_lock() -> None:
    """Per-sender runs must not hold an owner lock around the whole pass.

    Asserted on the source rather than by running the strategy because the stall
    was structural: the ``async with`` spanned the subject loop, so any owner-wide
    acquire there is the defect regardless of what the loop body does.
    """
    # The decorator returns a Strategy; the original coroutine is on `.meta.func`.
    code = mod.extract_user_profile.meta.func.__code__
    # The guard is chosen by mode; nullcontext is what makes sender mode lock-free
    # at the owner level.
    assert "nullcontext" in set(code.co_names)
    # And the acquire must be conditional, not unconditional-then-ignored.
    assert "PROFILE_SUBJECT" in set(code.co_names)


def test_per_subject_lock_wraps_the_body() -> None:
    """``_extract_one_subject`` must acquire before doing any of the work.

    The body lives in ``_extract_one_subject_locked`` precisely so the lock is
    unmissable at the boundary; if someone inlines it back, this fails.
    """
    assert asyncio.iscoroutinefunction(mod._extract_one_subject)
    assert asyncio.iscoroutinefunction(mod._extract_one_subject_locked)
    names = set(mod._extract_one_subject.__code__.co_names)
    assert "get_partition_lock" in names
    assert "_extract_one_subject_locked" in names
    # The wrapper must not do the work itself -- reading or writing before the
    # acquire is the race the lock exists to prevent.
    assert "_get_reader" not in names
    assert "_persist_profile" not in names


async def test_two_subjects_make_progress_while_a_third_is_stuck() -> None:
    """The end state the fix buys, expressed as a scheduling property."""
    stuck = asyncio.Event()

    async def work(subject: str) -> str:
        async with _lock("group", subject):
            if subject == "Lan Ye":
                await stuck.wait()  # never set: this one hangs forever
            return subject

    hung = asyncio.create_task(work("Lan Ye"))
    await asyncio.sleep(0)
    done = await asyncio.gather(work("Bo Chen"), work("Mei Zheng"))
    assert done == ["Bo Chen", "Mei Zheng"]
    assert not hung.done()
    hung.cancel()
    with pytest.raises(asyncio.CancelledError):
        await hung
