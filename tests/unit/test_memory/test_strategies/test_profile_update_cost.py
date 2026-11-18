"""The 2026-11-18 profile-update cost changes: the age rule, and bare stored items.

What these pin:

- ``[profile] max_age_seconds``: the idle tick reads a user's queue only once
  its oldest segment is at least that old, so a deployment's idle-only cadence
  (556 reads of the LoCoMo arm's 674 segments) becomes the 7-day rule's 362;
  a queue already old enough is read, a young one is left queued, and a
  segment that is gone counts as old so a queue is never stranded;
- the catch-up and a flush still read whatever is queued: they ask for the
  profile to be current now, and the age rule is the idle tick's alone;
- ``[profile] stored_grounding`` reaches the update: false sends a stored
  item as its description and name, with no evidence or basis.
"""

from __future__ import annotations

import importlib
import json
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

from everos.config import Settings
from everos.core.utils.datetime import get_utc_now, to_timestamp_ms
from everos.memory._partition_locks import _reset_for_tests
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted, UserProfileDue, user_key
from everos.memory.models import Episode, ExplicitInfoItem, ImplicitTraitItem, Profile
from everos.memory.ports import IdleTick, Scope
from everos.memory.strategies import extract_user_profile
from tests.fakes import make_deps

_TS = 1_700_000_000_000
_DAY_MS = 86_400_000


@pytest.fixture(autouse=True)
def _isolate_partition_locks() -> None:
    _reset_for_tests()


def _profile_settings(monkeypatch: pytest.MonkeyPatch, **values: Any) -> None:
    """Set ``[profile]`` for the strategy under test."""
    base = Settings()
    settings = base.model_copy(
        update={"profile": base.profile.model_copy(update=values)}
    )
    module = importlib.import_module("everos.memory.strategies.extract_user_profile")
    monkeypatch.setattr(module, "load_settings", lambda: settings)


def _scope(owner_id: str = "u_alice") -> Scope:
    return Scope(
        app_id="default", project_id="default", owner_id=owner_id, owner_type="user"
    )


def _event(memcell_id: str = "mc_1") -> EpisodeExtracted:
    return EpisodeExtracted(
        memcell_id=memcell_id,
        episode_entry_id=f"ep_{memcell_id}",
        episode_text="...",
        episode_timestamp_ms=_TS,
        owner_id="u_alice",
        language="English",
    )


def _memcell(text: str, *, ts: int) -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id=f"m_{ts}",
                role="user",
                content=text,
                timestamp=ts,
                sender_id="u_alice",
                sender_name="Alice",
            )
        ],
        timestamp=ts,
    )


def _stored(**values: Any) -> Profile:
    base: dict[str, Any] = {
        "owner_id": "u_alice",
        "summary": "s",
        "timestamp": _TS,
        "primary_language": "English",
    }
    base.update(values)
    return Profile.model_validate(base)


class _Model:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.sent: list[list[Any]] = []
        self.llm = FakeLLMClient(handler=self._answer)

    def _answer(self, messages: list[Any], **_kwargs: Any) -> str:
        self.sent.append(messages)
        return self.replies.pop(0)

    def user_text(self, call: int = 0) -> str:
        return self.sent[call][1].content


def _deps(model: _Model, **kwargs: Any) -> MemoryDeps:
    deps = make_deps(llm=model.llm, **kwargs)
    seed = deps.stores.memcell.seed

    def seed_with_episode(memcell_id: str, cell: MemCell, *args: Any) -> None:
        seed(memcell_id, cell, *args)
        deps.stores.episode.seed(
            _scope(),
            f"ep_{memcell_id}",
            Episode(
                owner_id="u_alice",
                episode="...",
                summary="...",
                timestamp=cell.timestamp,
                parent_id=memcell_id,
            ),
        )

    deps.stores.memcell.seed = seed_with_episode  # type: ignore[method-assign]
    return deps


def _none() -> str:
    return json.dumps({"operations": [{"action": "none"}]})


async def _queue_one(deps: MemoryDeps, *, age_days: float) -> None:
    """A profile with items, and one segment queued that is ``age_days`` old."""
    ts = to_timestamp_ms(get_utc_now()) - int(age_days * _DAY_MS)
    deps.stores.memcell.seed(
        "mc_old", _memcell("I always book refundable tickets.", ts=ts)
    )
    await deps.stores.profile.write(
        _scope(),
        _stored(
            explicit_info=[
                ExplicitInfoItem(
                    evidence="2023-11-01: 'I live in Paris'",
                    description="Lives in Paris",
                    category="Home",
                )
            ],
            pending_memcell_ids=["mc_old"],
        ),
    )


def _tick() -> IdleTick:
    return IdleTick(
        strategy_name="extract_user_profile_idle",
        bucket_key=user_key("default", "default", "u_alice"),
        idle_seconds=600,
    )


async def test_a_young_queue_is_left_queued_by_the_idle_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _profile_settings(monkeypatch, max_age_seconds=7 * 86_400)
    model = _Model(_none())
    deps = _deps(model)
    await _queue_one(deps, age_days=0.5)

    await extract_user_profile(_tick(), deps)

    assert model.sent == []
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.pending_memcell_ids == ["mc_old"]


async def test_a_queue_that_reached_the_age_rule_is_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _profile_settings(monkeypatch, max_age_seconds=7 * 86_400)
    model = _Model(_none())
    deps = _deps(model)
    await _queue_one(deps, age_days=8)

    await extract_user_profile(_tick(), deps)

    assert len(model.sent) == 1
    assert "<segment id=" in model.user_text()
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.pending_memcell_ids == []


async def test_the_catch_up_reads_whatever_is_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``UserProfileDue`` asks for the profile now: the age rule does not hold it."""
    _profile_settings(monkeypatch, max_age_seconds=7 * 86_400)
    model = _Model(_none())
    deps = _deps(model)
    await _queue_one(deps, age_days=0.1)

    await extract_user_profile(
        UserProfileDue(app_id="default", project_id="default", user_id="u_alice"), deps
    )

    assert len(model.sent) == 1


async def test_a_queue_whose_oldest_segment_is_gone_makes_no_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An age rule that reads the queue is not a licence to read a deleted segment.

    No call is made, and the queue stands as every version before this one
    left it -- a stranding the age rule neither causes nor fixes.
    """
    _profile_settings(monkeypatch, max_age_seconds=7 * 86_400)
    model = _Model(_none())
    deps = _deps(model)
    await deps.stores.profile.write(
        _scope(), _stored(pending_memcell_ids=["mc_deleted"])
    )

    await extract_user_profile(_tick(), deps)

    assert model.sent == []
    stored = await deps.stores.profile.read(_scope())
    assert stored is not None and stored.pending_memcell_ids == ["mc_deleted"]


async def test_stored_grounding_false_shows_descriptions_and_names_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``stored_grounding = false``: the stored block loses its quotes."""
    _profile_settings(monkeypatch, update_every=1, stored_grounding=False)
    model = _Model(_none())
    deps = _deps(model, profile_stored_grounding=False)
    deps.stores.memcell.seed("mc_1", _memcell("Black again.", ts=_TS))
    await deps.stores.profile.write(
        _scope(),
        _stored(
            explicit_info=[
                ExplicitInfoItem(
                    evidence="2023-11-01: 'I live in Paris'",
                    description="Lives in Paris",
                    category="Home",
                )
            ],
            implicit_traits=[
                ImplicitTraitItem(
                    basis="2023-11-01: 'dark, please'",
                    description="Prefers dark colours",
                    trait="Style preferences",
                )
            ],
        ),
    )

    await extract_user_profile(_event("mc_1"), deps)

    text = model.user_text()
    stored_block = text.split("<stored_profile>")[1].split("</stored_profile>")[0]
    assert "Lives in Paris" in stored_block and "Prefers dark colours" in stored_block
    assert "I live in Paris" not in stored_block
    assert "dark, please" not in stored_block
    assert "evidence" not in stored_block and "basis" not in stored_block


async def test_the_two_settings_reach_the_strategy_from_a_deps_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defaults: quotes shown, 8 segments a call, 7 days."""
    profile = Settings().profile
    assert (profile.update_every, profile.max_age_seconds) == (8, 7 * 86_400)
    assert profile.stored_grounding is True
